"""FarmVisionTool — plant/pest photo analysis via a YOLOv8 classifier."""

import logging
from pathlib import Path
from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger("farm-tools")


class VisionInput(BaseModel):
    image_path: str = Field(description="Path to the plant or pest image file to analyse")


class FarmVisionTool:
    """Wraps a trained YOLOv8 classifier for plant disease/pest identification."""

    LABEL_METADATA = {
        "Tomato___Early_blight":       ("Tomato — Early Blight",  "disease", "medium"),
        "Tomato___Late_blight":        ("Tomato — Late Blight",   "disease", "high"),
        "Tomato___healthy":            ("Tomato — Healthy",       "healthy",  "none"),
        "Potato___Early_blight":       ("Potato — Early Blight",  "disease", "medium"),
        "Potato___Late_blight":        ("Potato — Late Blight",   "disease", "high"),
        "Potato___healthy":            ("Potato — Healthy",       "healthy",  "none"),
        "Corn_(maize)___Common_rust_": ("Corn — Common Rust",     "disease", "medium"),
        "Corn_(maize)___healthy":      ("Corn — Healthy",         "healthy",  "none"),
        "Wheat___stripe_rust":         ("Wheat — Stripe Rust",    "disease", "high"),
        "Wheat___Brown_rust":          ("Wheat — Brown Rust",     "disease", "medium"),
    }

    def __init__(self, model_path: str = "outputs/farm-vision/weights/best.pt"):
        self.model_path = model_path
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                if Path(self.model_path).exists():
                    self._model = YOLO(self.model_path)
                    logger.info(f"Vision model loaded: {self.model_path} ✓")
                else:
                    logger.warning(f"Vision model not found at {self.model_path}")
                    self._model = "fallback"
            except Exception as e:
                logger.error(f"Vision model init error: {e}")
                self._model = "fallback"
        return self._model

    def run(self, image_path: str) -> str:
        model = self._get_model()
        if model == "fallback":
            return "Vision model not available. Please train the YOLOv8 model first."

        if not Path(image_path).exists():
            return f"Image not found at path: {image_path}"

        try:
            results = model(image_path, verbose=False)
            probs = results[0].probs
            top3 = [(model.names[int(probs.top5[i])], float(probs.top5conf[i])) for i in range(3)]

            top_class, top_conf = top3[0]
            meta = self.LABEL_METADATA.get(
                top_class,
                (top_class.replace("___", " — ").replace("pest_", ""), "pest", "medium")
            )
            display, itype, severity = meta

            output = f"""Vision Analysis Result:
- Detected: {display}
- Type: {itype.upper()}
- Confidence: {top_conf*100:.1f}%
- Severity: {severity.upper()}
- Healthy: {itype == 'healthy'}

Other possibilities:
- {top3[1][0].replace('___', ' — ')}: {top3[1][1]*100:.1f}%
- {top3[2][0].replace('___', ' — ')}: {top3[2][1]*100:.1f}%

Next step: {"No action needed — plant is healthy." if itype == "healthy" else f"Search knowledge base for treatment of {display}."}
"""
            return output
        except Exception as e:
            return f"Vision analysis error: {str(e)}"

    def as_tool(self) -> BaseTool:
        vision_instance = self

        class _VisionTool(BaseTool):
            name: str = "farm_vision"
            description: str = (
                "Analyse a plant leaf or pest photo to identify diseases or pests. "
                "Input must be a file path to an image (jpg/png). "
                "Use this whenever a user uploads or mentions a photo of their plant or crop."
            )
            args_schema: Type[BaseModel] = VisionInput

            def _run(self, image_path: str) -> str:
                return vision_instance.run(image_path)

        return _VisionTool()
