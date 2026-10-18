"""
YieldPredictionTool — expected crop yield from farm parameters.

Two paths:
  1. ML model (scripts/train_yield_model.py): gradient-boosted trees trained on
     the "Crop Yield in Indian States" dataset (1997–2020; crop, season, state,
     area, fertilizer, pesticide, rainfall → yield in tonnes/hectare). Used when
     the artifact exists AND the crop + state were seen in training.
  2. Rule-based fallback: typical tonnes/acre per crop × soil, adjusted for
     irrigation and fertilizer. Used otherwise, and labelled as such.

Every answer says which path produced it, so the farmer (and the interviewer)
can tell a learned estimate from a heuristic one.
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger("farm-tools")

DEFAULT_MODEL_DIR = "outputs/yield_model"
HA_PER_ACRE = 0.404686


class YieldInput(BaseModel):
    crop_type: str = Field(description="Crop, e.g. Wheat, Rice, Maize, Cotton, Potato")
    farm_area_acres: float = Field(description="Farm area in acres")
    season: str = Field(description="Season: Kharif, Rabi, Zaid/Summer, Autumn, Winter or Whole Year")
    state: Optional[str] = Field(default=None, description="Indian state, e.g. Punjab, Karnataka")
    soil_type: Optional[str] = Field(default=None, description="Soil type: Loamy, Sandy, Clay, Silty, Peaty")
    irrigation_type: Optional[str] = Field(default=None, description="Drip, Sprinkler, Flood, Rain-fed or Manual")
    fertilizer_kg: float = Field(default=0.0, description="Total fertilizer applied in kg")
    pesticide_kg: float = Field(default=0.0, description="Total pesticide applied in kg")
    annual_rainfall_mm: Optional[float] = Field(default=None, description="Annual rainfall in mm, if known")


class YieldPredictionTool:

    # Typical yields (tonnes/acre) by soil, for the rule-based fallback
    BASE_YIELDS = {
        "wheat": {"loamy": 2.1, "clay": 1.8, "silty": 2.0, "sandy": 1.5, "peaty": 1.6},
        "rice": {"loamy": 2.5, "clay": 2.8, "silty": 2.6, "sandy": 1.8, "peaty": 2.0},
        "tomato": {"loamy": 12.0, "clay": 10.0, "silty": 11.0, "sandy": 8.0, "peaty": 9.0},
        "cotton": {"loamy": 0.45, "clay": 0.5, "silty": 0.42, "sandy": 0.35, "peaty": 0.38},
        "sugarcane": {"loamy": 35.0, "clay": 32.0, "silty": 34.0, "sandy": 25.0, "peaty": 28.0},
        "potato": {"loamy": 8.0, "clay": 7.0, "silty": 7.5, "sandy": 6.0, "peaty": 6.5},
        "maize": {"loamy": 2.8, "clay": 2.5, "silty": 2.6, "sandy": 2.0, "peaty": 2.2},
        "soybean": {"loamy": 1.2, "clay": 1.0, "silty": 1.1, "sandy": 0.9, "peaty": 0.95},
    }
    CROP_ALIASES = {"corn": "maize", "paddy": "rice", "soyabean": "soybean"}
    IRRIGATION_MULTIPLIERS = {"drip": 1.30, "sprinkler": 1.15, "flood": 1.00, "rain-fed": 0.80, "manual": 0.90}
    WATER_M3_PER_ACRE = {"drip": 180, "sprinkler": 220, "flood": 300}

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._model = None
        self._meta: Optional[dict] = None
        self._load_attempted = False

    # ── ML path ───────────────────────────────────────────────────────────────

    def _load_model(self) -> bool:
        if not self._load_attempted:
            self._load_attempted = True
            model_file, meta_file = self.model_dir / "yield_model.joblib", self.model_dir / "metadata.json"
            if model_file.exists() and meta_file.exists():
                try:
                    import joblib
                    import sklearn

                    self._model = joblib.load(model_file)
                    self._meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    trained_with = self._meta.get("sklearn_version")
                    if trained_with and trained_with != sklearn.__version__:
                        logger.warning(
                            f"Yield model trained with scikit-learn {trained_with}, running {sklearn.__version__} — "
                            "pin the version in requirements.txt to avoid silent behaviour changes."
                        )
                    logger.info("Yield ML model loaded ✓")
                except Exception as e:
                    logger.error(f"Could not load yield model: {e}")
                    self._model = None
        return self._model is not None

    def _match_category(self, value: Optional[str], known: list[str]) -> Optional[str]:
        if not value:
            return None
        wanted = value.strip().lower()
        for k in known:
            if k.lower() == wanted:
                return k
        return None

    def _predict_ml(self, crop: str, state: Optional[str], season: str, area_ha: float,
                    fertilizer_kg: float, pesticide_kg: float, rainfall: Optional[float]) -> Optional[dict]:
        if not self._load_model():
            return None
        cats = self._meta["categories"]
        crop_c = self._match_category(crop, cats["Crop"])
        state_c = self._match_category(state, cats["State"])
        season_c = self._match_category(season, cats["Season"]) or self._match_category("Whole Year", cats["Season"])
        if not crop_c or not state_c or not season_c:
            return None   # unseen crop/state — don't extrapolate a tree model

        import pandas as pd

        if rainfall is None:
            rainfall = self._meta.get("state_rainfall_median", {}).get(state_c, self._meta["global_rainfall_median"])
        area_ha = max(area_ha, 0.01)
        row = pd.DataFrame([{
            "Crop": crop_c, "Season": season_c, "State": state_c,
            "log_area": math.log1p(area_ha),
            "fertilizer_per_ha": fertilizer_kg / area_ha,
            "pesticide_per_ha": pesticide_kg / area_ha,
            "Annual_Rainfall": float(rainfall),
        }])[self._meta["feature_order"]]
        log_pred = float(self._model.predict(row)[0])
        t_per_ha = max(math.expm1(log_pred), 0.0)
        return {"crop": crop_c, "state": state_c, "season": season_c, "t_per_ha": t_per_ha,
                "rainfall": rainfall, "test_mae": self._meta.get("metrics", {}).get("test", {}).get("mae_t_per_ha")}

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, crop_type: str, farm_area_acres: float, season: str, state: Optional[str] = None,
            soil_type: Optional[str] = None, irrigation_type: Optional[str] = None,
            fertilizer_kg: float = 0.0, pesticide_kg: float = 0.0,
            annual_rainfall_mm: Optional[float] = None) -> str:
        if farm_area_acres <= 0:
            return "Farm area must be greater than zero acres."
        crop = self.CROP_ALIASES.get(crop_type.lower().strip(), crop_type.lower().strip())
        area_ha = farm_area_acres * HA_PER_ACRE

        ml = self._predict_ml(crop, state, season, area_ha, fertilizer_kg, pesticide_kg, annual_rainfall_mm)
        if ml:
            t_per_acre = ml["t_per_ha"] * HA_PER_ACRE
            mae_line = f"\n- Typical error on unseen years: ±{ml['test_mae']:.2f} t/ha" if ml["test_mae"] else ""
            return (
                f"Yield prediction (ML model: gradient-boosted trees on Indian state crop data, 1997–2020)\n"
                f"Inputs: {ml['crop']} | {ml['state']} | {ml['season']} | {farm_area_acres} acres | "
                f"rainfall {ml['rainfall']:.0f} mm\n\n"
                f"- Yield: {ml['t_per_ha']:.2f} t/ha ({t_per_acre:.2f} t/acre)\n"
                f"- Total for the farm: {t_per_acre * farm_area_acres:.2f} tonnes"
                f"{mae_line}\n\n"
                "Note: trained on state-level averages, so it reflects typical farms in that state and season, "
                "not your specific field, seed variety, or this year's weather. Confirm with the nearest KVK."
            )
        return self._rule_based(crop, crop_type, farm_area_acres, season, soil_type, irrigation_type, fertilizer_kg)

    def _rule_based(self, crop: str, crop_type: str, acres: float, season: str,
                    soil_type: Optional[str], irrigation_type: Optional[str], fertilizer_kg: float) -> str:
        assumed = []
        soil = (soil_type or "").lower().strip()
        if soil not in {"loamy", "clay", "silty", "sandy", "peaty"}:
            soil = "loamy"
            assumed.append("loamy soil")
        irrigation = (irrigation_type or "").lower().strip().replace("rainfed", "rain-fed")
        if irrigation not in self.IRRIGATION_MULTIPLIERS:
            irrigation = "flood"
            assumed.append("flood irrigation")

        base = self.BASE_YIELDS.get(crop, {}).get(soil)
        if base is None:
            return (f"I don't have yield data for {crop_type}. Supported crops for estimates: "
                    f"{', '.join(sorted(self.BASE_YIELDS))}. Your nearest KVK can give local figures.")

        fert_t_per_acre = (fertilizer_kg / 1000) / acres
        yield_per_acre = base * self.IRRIGATION_MULTIPLIERS[irrigation] * (1.0 + min(fert_t_per_acre * 0.05, 0.25))
        water = acres * self.WATER_M3_PER_ACRE.get(irrigation, 150)

        recs = []
        if irrigation in ("flood", "rain-fed") and crop in ("tomato", "potato"):
            recs.append("Drip irrigation typically raises yield for this crop by 25–30% and saves water")
        if fert_t_per_acre < 0.3:
            recs.append("Fertilizer use looks low — a soil test (Soil Health Card) would show which nutrients to add")
        if not recs:
            recs.append("Farm parameters look reasonable for this crop")

        assumed_line = f"\nAssumed (not provided): {', '.join(assumed)}." if assumed else ""
        return (
            f"Yield estimate (rule-based: typical values, not a trained model)\n"
            f"Inputs: {crop_type} | {acres} acres | {season} | {soil} soil | {irrigation} irrigation{assumed_line}\n\n"
            f"- Yield per acre: {yield_per_acre:.2f} t/acre\n"
            f"- Total yield: {yield_per_acre * acres:.2f} tonnes\n"
            f"- Water use: ~{water:,.0f} m³\n\n"
            "Recommendations:\n" + "\n".join(f"- {r}" for r in recs) +
            "\n\nActual yield depends on weather, seed variety and local conditions — confirm with your nearest KVK."
        )

    def as_tool(self) -> BaseTool:
        yield_instance = self

        class _YieldTool(BaseTool):
            name: str = "yield_prediction"
            description: str = (
                "Predict crop yield from farm parameters. Use when a farmer asks about expected yield, "
                "production estimates, or wants to compare farming approaches. "
                "Requires crop type, farm area in acres and season; state, soil, irrigation, fertilizer, "
                "pesticide and rainfall improve the estimate. Ask the farmer for missing required values."
            )
            args_schema: Type[BaseModel] = YieldInput

            def _run(self, **kwargs) -> str:
                return yield_instance.run(**kwargs)

        return _YieldTool()
