"""YieldPredictionTool — estimates crop yield from farm parameters."""

import logging
from pathlib import Path
from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger("farm-tools")


class YieldInput(BaseModel):
    crop_type: str = Field(description="Type of crop e.g. Wheat, Rice, Tomato, Cotton")
    farm_area_acres: float = Field(description="Farm area in acres")
    soil_type: str = Field(description="Soil type e.g. Loamy, Sandy, Clay, Silty, Peaty")
    irrigation_type: str = Field(description="Irrigation type e.g. Drip, Sprinkler, Flood, Rain-fed, Manual")
    season: str = Field(description="Season e.g. Kharif, Rabi, Zaid")
    fertilizer_tons: float = Field(description="Fertilizer used in tons", default=0.0)
    pesticide_kg: float = Field(description="Pesticide used in kg", default=0.0)


class YieldPredictionTool:
    """
    ML-based yield prediction from tabular farm data.
    Uses a trained model if present at outputs/yield_model.pkl, otherwise a rule-based estimate.
    """

    # Rule-based yield estimates (tons/acre) when ML model not available
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

    IRRIGATION_MULTIPLIERS = {
        "drip": 1.30, "sprinkler": 1.15, "flood": 1.00,
        "rain-fed": 0.80, "manual": 0.90,
    }

    def run(self, crop_type: str, farm_area_acres: float, soil_type: str,
            irrigation_type: str, season: str,
            fertilizer_tons: float = 0.0, pesticide_kg: float = 0.0) -> str:

        crop = crop_type.lower().strip()
        soil = soil_type.lower().strip()
        irrigation = irrigation_type.lower().strip()

        ml_result = self._try_ml_model(
            crop, farm_area_acres, soil, irrigation, season, fertilizer_tons, pesticide_kg
        )
        if ml_result:
            return ml_result

        # Fallback: rule-based
        base = self.BASE_YIELDS.get(crop, {}).get(soil, 2.0)
        irr_mult = self.IRRIGATION_MULTIPLIERS.get(irrigation, 1.0)

        fert_per_acre = fertilizer_tons / max(farm_area_acres, 0.1)
        fert_mult = 1.0 + min(fert_per_acre * 0.05, 0.25)

        yield_per_acre = base * irr_mult * fert_mult
        total_yield = yield_per_acre * farm_area_acres
        water_usage = farm_area_acres * (
            180 if irrigation == "drip" else
            220 if irrigation == "sprinkler" else
            300 if irrigation == "flood" else 150
        )

        recs = []
        if irrigation in ["flood", "rain-fed"] and crop in ["tomato", "potato"]:
            recs.append("Switch to drip irrigation — could increase yield by 25–30%")
        if fert_per_acre < 0.3:
            recs.append("Fertilizer usage is low — consider soil testing for optimal NPK")
        if not recs:
            recs.append("Farm parameters look good for this crop")

        return f"""Yield Prediction ({crop_type} | {farm_area_acres} acres | {season}):

Estimates:
- Yield per acre: {yield_per_acre:.2f} tons/acre
- Total yield: {total_yield:.2f} tons
- Water usage: ~{water_usage:,.0f} cubic meters

Recommendations:
{chr(10).join(f"- {r}" for r in recs)}

Note: These are estimates based on typical values.
Actual yield depends on weather, seed variety, and local conditions.
Contact your nearest KVK for field-specific advice.
"""

    def _try_ml_model(self, *args) -> Optional[str]:
        """Try loading a trained model if available; returns None to use the rule-based fallback."""
        model_path = "outputs/yield_model.pkl"
        if not Path(model_path).exists():
            return None
        try:
            import pickle
            with open(model_path, "rb") as f:
                pickle.load(f)
            logger.info("ML yield model found but inference is not yet implemented — using rule-based estimate")
            return None
        except Exception:
            return None

    def as_tool(self) -> BaseTool:
        yield_instance = self

        class _YieldTool(BaseTool):
            name: str = "yield_prediction"
            description: str = (
                "Predict crop yield given farm parameters. "
                "Use when a farmer asks about expected yield, production estimates, "
                "or wants to compare different farming approaches. "
                "Requires: crop type, farm area, soil type, irrigation type, season."
            )
            args_schema: Type[BaseModel] = YieldInput

            def _run(self, crop_type: str, farm_area_acres: float, soil_type: str,
                     irrigation_type: str, season: str,
                     fertilizer_tons: float = 0.0, pesticide_kg: float = 0.0) -> str:
                return yield_instance.run(
                    crop_type, farm_area_acres, soil_type,
                    irrigation_type, season, fertilizer_tons, pesticide_kg
                )

        return _YieldTool()
