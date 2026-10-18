from pathlib import Path

import pytest

from src.tools.vision_tool import DEFAULT_MODEL_PATH, LABEL_METADATA, display_for, format_prediction

# The 27 class names stored in outputs/farm-vision/weights/best.pt
MODEL_CLASSES = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy", "pest_ants",
    "pest_bees", "pest_beetle", "pest_catterpillar", "pest_earthworms", "pest_earwig", "pest_grasshopper",
    "pest_moth", "pest_slug", "pest_snail", "pest_wasp", "pest_weevil",
]


def test_every_model_class_has_metadata():
    assert set(MODEL_CLASSES) == set(LABEL_METADATA)


@pytest.mark.parametrize("label", ["pest_bees", "pest_earthworms", "pest_wasp"])
def test_beneficial_organisms_are_never_called_pests(label):
    assert display_for(label)[1] == "beneficial"
    assert "do NOT spray" in format_prediction([(label, 0.95), ("pest_ants", 0.03), ("pest_moth", 0.02)])


def test_disease_is_typed_as_disease_not_pest():
    out = format_prediction([("Tomato_Early_blight", 0.91), ("Tomato_Late_blight", 0.05), ("Tomato_healthy", 0.04)])
    assert "Type: DISEASE" in out and "Tomato — Early blight" in out


def test_low_confidence_is_reported_as_uncertain_without_treatment():
    out = format_prediction([("Potato___Late_blight", 0.31), ("Tomato_Late_blight", 0.30), ("pest_slug", 0.10)])
    assert out.startswith("Vision analysis result: UNCERTAIN")
    assert "Do not recommend any treatment" in out


def test_unknown_label_gets_a_readable_fallback():
    name, kind, _ = display_for("Wheat___Brown_rust")
    assert name == "Wheat — brown rust" and kind == "disease"


def test_shipped_weights_match_the_metadata():
    weights = Path(DEFAULT_MODEL_PATH)
    if not weights.exists() or weights.stat().st_size < 1_000_000:
        pytest.skip("weights not present (or only a Git LFS pointer was checked out)")
    yolo = pytest.importorskip("ultralytics")
    names = yolo.YOLO(DEFAULT_MODEL_PATH).names
    assert set(names.values()) == set(MODEL_CLASSES)
