import json

import joblib
import numpy as np
import pandas as pd

from scripts.train_yield_model import FEATURE_ORDER, make_model
from src.tools.yield_tool import YieldPredictionTool


def test_rule_based_path_when_no_model(tmp_path):
    out = YieldPredictionTool(model_dir=str(tmp_path)).run(
        crop_type="Wheat", farm_area_acres=2, season="Rabi", soil_type="Loamy", irrigation_type="Drip")
    assert "rule-based" in out and "t/acre" in out


def test_missing_soil_and_irrigation_are_stated_as_assumptions(tmp_path):
    out = YieldPredictionTool(model_dir=str(tmp_path)).run(crop_type="paddy", farm_area_acres=1, season="Kharif")
    assert "Assumed (not provided): loamy soil, flood irrigation" in out


def test_invalid_area_and_unknown_crop(tmp_path):
    tool = YieldPredictionTool(model_dir=str(tmp_path))
    assert "greater than zero" in tool.run(crop_type="Wheat", farm_area_acres=0, season="Rabi")
    assert "don't have yield data" in tool.run(crop_type="Quinoa", farm_area_acres=1, season="Rabi")


def _train_tiny_model(model_dir):
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "Crop": rng.choice(["Wheat", "Rice"], n),
        "Season": rng.choice(["Rabi", "Kharif"], n),
        "State": rng.choice(["Punjab", "Karnataka"], n),
        "log_area": rng.uniform(5, 12, n),
        "fertilizer_per_ha": rng.uniform(50, 200, n),
        "pesticide_per_ha": rng.uniform(0.1, 1, n),
        "Annual_Rainfall": rng.uniform(500, 2500, n),
    })
    y = np.log1p(np.where(df.Crop == "Wheat", 4.0, 3.0) + rng.normal(0, 0.1, n))
    model = make_model().fit(df[FEATURE_ORDER], y)
    model_dir.mkdir()
    joblib.dump(model, model_dir / "yield_model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps({
        "feature_order": FEATURE_ORDER,
        "categories": {"Crop": ["Rice", "Wheat"], "Season": ["Kharif", "Rabi"], "State": ["Karnataka", "Punjab"]},
        "state_rainfall_median": {"Punjab": 650.0, "Karnataka": 1200.0},
        "global_rainfall_median": 1000.0,
        "metrics": {"test": {"mae_t_per_ha": 0.3}},
    }))


def test_ml_path_for_known_crop_and_state(tmp_path):
    model_dir = tmp_path / "yield"
    _train_tiny_model(model_dir)
    tool = YieldPredictionTool(model_dir=str(model_dir))
    out = tool.run(crop_type="wheat", farm_area_acres=5, season="rabi", state="punjab")
    assert "ML model" in out and "Wheat | Punjab | Rabi" in out and "±0.30 t/ha" in out

    # an unseen state must not be extrapolated by the tree model
    assert "rule-based" in tool.run(crop_type="wheat", farm_area_acres=5, season="rabi", state="Kerala")
