"""
FarmVisionTool — plant disease / insect photo classification with YOLOv8s-cls.

The classifier was trained on 27 classes: 15 PlantVillage leaf classes (bell
pepper, potato, tomato) and 12 classes from the Agricultural Pests image
dataset. Two things the output has to be honest about:

1. Closed set. A softmax classifier always picks one of its 27 classes — a
   wheat leaf, a cow, or a blurry photo still gets a label. So low-confidence
   predictions are reported as "uncertain" and the result always states which
   crops the model actually covers.
2. Not every insect is a pest. Bees are pollinators, earthworms improve soil,
   and most wasps are natural enemies of caterpillars. Telling a farmer to
   spray them would do real harm, so they are typed as "beneficial".

This tool is not exposed to the LLM. The agent graph runs it directly on the
uploaded file whenever an image is attached (see src/agent/graph.py), so the
model never chooses — or invents — a file path.
"""

import logging
from pathlib import Path

logger = logging.getLogger("farm-tools")

DEFAULT_MODEL_PATH = "outputs/farm-vision/weights/best.pt"

LOW_CONFIDENCE = 0.40     # below: "uncertain", ask for a better photo
HIGH_CONFIDENCE = 0.70    # below this but above LOW: "moderate"

COVERED_SCOPE = (
    "leaves of bell pepper, potato and tomato, plus 12 insect types "
    "(ants, bees, beetles, caterpillars, earthworms, earwigs, grasshoppers, "
    "moths, slugs, snails, wasps, weevils)"
)

# Exact class names stored in best.pt  →  (display name, type, severity)
# type ∈ disease | virus | healthy | pest | beneficial
LABEL_METADATA: dict[str, tuple[str, str, str]] = {
    "Pepper__bell___Bacterial_spot":               ("Bell pepper — Bacterial spot",          "disease", "medium"),
    "Pepper__bell___healthy":                      ("Bell pepper — Healthy",                 "healthy", "none"),
    "Potato___Early_blight":                       ("Potato — Early blight",                 "disease", "medium"),
    "Potato___Late_blight":                        ("Potato — Late blight",                  "disease", "high"),
    "Potato___healthy":                            ("Potato — Healthy",                      "healthy", "none"),
    "Tomato_Bacterial_spot":                       ("Tomato — Bacterial spot",               "disease", "medium"),
    "Tomato_Early_blight":                         ("Tomato — Early blight",                 "disease", "medium"),
    "Tomato_Late_blight":                          ("Tomato — Late blight",                  "disease", "high"),
    "Tomato_Leaf_Mold":                            ("Tomato — Leaf mold",                    "disease", "medium"),
    "Tomato_Septoria_leaf_spot":                   ("Tomato — Septoria leaf spot",           "disease", "medium"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Tomato — Two-spotted spider mite",      "pest",    "medium"),
    "Tomato__Target_Spot":                         ("Tomato — Target spot",                  "disease", "medium"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus":       ("Tomato — Yellow leaf curl virus",       "virus",   "high"),
    "Tomato__Tomato_mosaic_virus":                 ("Tomato — Mosaic virus",                 "virus",   "high"),
    "Tomato_healthy":                              ("Tomato — Healthy",                      "healthy", "none"),
    "pest_ants":                                   ("Ants",                                  "pest",    "low"),
    "pest_bees":                                   ("Bees (pollinators)",                    "beneficial", "none"),
    "pest_beetle":                                 ("Beetle",                                "pest",    "medium"),
    "pest_catterpillar":                           ("Caterpillar",                           "pest",    "medium"),
    "pest_earthworms":                             ("Earthworms (soil health)",              "beneficial", "none"),
    "pest_earwig":                                 ("Earwig",                                "pest",    "low"),
    "pest_grasshopper":                            ("Grasshopper",                           "pest",    "medium"),
    "pest_moth":                                   ("Moth (larvae can damage crops)",        "pest",    "medium"),
    "pest_slug":                                   ("Slug",                                  "pest",    "medium"),
    "pest_snail":                                  ("Snail",                                 "pest",    "medium"),
    "pest_wasp":                                   ("Wasp (mostly natural enemies of pests)", "beneficial", "none"),
    "pest_weevil":                                 ("Weevil",                                "pest",    "medium"),
}

NEXT_STEP = {
    "healthy":    "No disease detected. Keep monitoring, especially after rain or humid spells.",
    "disease":    "Look up treatment for {name} in the knowledge base (farm_rag) and confirm with the local KVK before spraying.",
    "virus":      "Viral diseases have no chemical cure: remove infected plants and control the insect vector (e.g. whitefly). Look up management for {name} in the knowledge base.",
    "pest":       "Look up integrated pest management for {name} in the knowledge base (farm_rag); spray only if damage crosses the economic threshold.",
    "beneficial": "This organism is beneficial to the farm — do NOT spray it. Avoid broad-spectrum insecticides that would kill it.",
}


def display_for(label: str) -> tuple[str, str, str]:
    """Metadata for a class label, with a readable fallback for unknown labels."""
    if label in LABEL_METADATA:
        return LABEL_METADATA[label]
    logger.warning(f"No metadata for vision class {label!r} — using a generic label")
    readable = label.replace("pest_", "").replace("___", " — ").replace("__", " ").replace("_", " ")
    return readable.strip().capitalize(), "pest" if label.startswith("pest_") else "disease", "medium"


def format_prediction(top: list[tuple[str, float]]) -> str:
    """Turn [(label, prob), …] (best first) into the text the agent reads."""
    label, conf = top[0]
    name, kind, severity = display_for(label)
    others = "\n".join(f"- {display_for(lbl)[0]}: {p * 100:.1f}%" for lbl, p in top[1:])

    if conf < LOW_CONFIDENCE:
        return (
            "Vision analysis result: UNCERTAIN\n"
            f"- Best guess: {name} ({conf * 100:.1f}%), which is too low to rely on.\n"
            f"- The model only recognises {COVERED_SCOPE}. The photo may show something outside that, "
            "or be too blurry, dark or far away.\n"
            "Next step: ask the farmer for a close, well-lit photo of a single affected leaf or insect, "
            "and which crop it is. Do not recommend any treatment from this result."
        )

    confidence_word = "high" if conf >= HIGH_CONFIDENCE else "moderate"
    return (
        "Vision analysis result:\n"
        f"- Detected: {name}\n"
        f"- Type: {kind.upper()}\n"
        f"- Confidence: {conf * 100:.1f}% ({confidence_word})\n"
        f"- Severity: {severity.upper()}\n"
        f"Other possibilities:\n{others}\n"
        f"Model scope: {COVERED_SCOPE}.\n"
        f"Next step: {NEXT_STEP[kind].format(name=name)}"
    )


class FarmVisionTool:
    """Wraps the trained YOLOv8 classifier."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self.unavailable_reason = ""

    def _get_model(self):
        if self._model is None and not self.unavailable_reason:
            try:
                from ultralytics import YOLO

                if Path(self.model_path).exists():
                    self._model = YOLO(self.model_path)
                    logger.info(f"Vision model loaded: {self.model_path} ({len(self._model.names)} classes) ✓")
                else:
                    self.unavailable_reason = f"weights not found at {self.model_path}"
                    logger.warning(f"Vision unavailable: {self.unavailable_reason}")
            except Exception as e:
                self.unavailable_reason = f"init error: {e}"
                logger.error(f"Vision model init error: {e}")
        return self._model

    def run(self, image_path: str) -> str:
        model = self._get_model()
        if model is None:
            return "Photo analysis is not available right now. Ask the farmer to describe the symptoms instead."
        if not Path(image_path).exists():
            return "The uploaded image could not be read. Ask the farmer to upload it again."
        try:
            probs = model(image_path, verbose=False)[0].probs
            k = min(3, len(model.names))
            top = [(model.names[int(probs.top5[i])], float(probs.top5conf[i])) for i in range(k)]
            return format_prediction(top)
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return "Photo analysis failed for this image. Ask the farmer to try a different photo."
