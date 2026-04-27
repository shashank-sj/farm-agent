"""
Farm Agent Tools
1. FarmRAGTool       — searches knowledge base (Project 1)
2. FarmVisionTool    — plant/pest image analysis (Project 3)
3. FarmWebSearchTool — real-time web search for prices/laws
4. YieldPredictionTool — ML-based yield prediction
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Type
from dataclasses import dataclass

from langchain.tools import BaseTool
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger("farm-tools")


# ══════════════════════════════════════════════════════════════════════
# Tool 1 — RAG Tool (from Project 1)
# ══════════════════════════════════════════════════════════════════════

class RAGInput(BaseModel):
    query: str = Field(description="Farming question to search the knowledge base for")

class FarmRAGTool:
    """Wraps the Project 1 RAG pipeline as a LangGraph tool."""

    def __init__(self, gemini_api_key: str, index_path: str = "data/faiss_index"):
        self.gemini_api_key = gemini_api_key
        self.index_path = index_path
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_community.retrievers import BM25Retriever
                from langchain.retrievers import EnsembleRetriever
                from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
                from langchain.chains import ConversationalRetrievalChain
                from langchain.memory import ConversationBufferWindowMemory
                from langchain.prompts import PromptTemplate

                os.environ["GOOGLE_API_KEY"] = self.gemini_api_key
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

                if Path(self.index_path).exists():
                    vectorstore = FAISS.load_local(
                        self.index_path, embeddings, allow_dangerous_deserialization=True
                    )
                    self._pipeline = vectorstore.as_retriever(search_kwargs={"k": 5})
                    logger.info("RAG index loaded ✓")
                else:
                    logger.warning(f"RAG index not found at {self.index_path}. Using fallback.")
                    self._pipeline = "fallback"
            except Exception as e:
                logger.error(f"RAG init error: {e}")
                self._pipeline = "fallback"
        return self._pipeline

    def run(self, query: str) -> str:
        pipeline = self._get_pipeline()
        if pipeline == "fallback":
            return f"Knowledge base not available. Please build the RAG index first by uploading farm documents."
        try:
            docs = pipeline.invoke(query)
            if not docs:
                return "No relevant information found in knowledge base for this query."
            context = "\n\n".join([d.page_content for d in docs[:3]])
            return f"From knowledge base:\n{context}"
        except Exception as e:
            return f"RAG search error: {str(e)}"

    def as_tool(self) -> BaseTool:
        rag_instance = self

        class _RAGTool(BaseTool):
            name: str = "farm_rag"
            description: str = (
                "Search the farm knowledge base for detailed information about: "
                "crop cultivation, soil health, fertilizers, irrigation, pest and disease "
                "management, organic farming, and government farming schemes. "
                "Use this for any farming how-to questions."
            )
            args_schema: Type[BaseModel] = RAGInput

            def _run(self, query: str) -> str:
                return rag_instance.run(query)

        return _RAGTool()


# ══════════════════════════════════════════════════════════════════════
# Tool 2 — Vision Tool (from Project 3)
# ══════════════════════════════════════════════════════════════════════

class VisionInput(BaseModel):
    image_path: str = Field(description="Path to the plant or pest image file to analyse")

class FarmVisionTool:
    """Wraps the Project 3 YOLOv8 model as a LangGraph tool."""

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

    def run(self, image_path: str) -> str:
        model = self._get_model()
        if model == "fallback":
            return "Vision model not available. Please train the YOLOv8 model first (Project 3)."

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


# ══════════════════════════════════════════════════════════════════════
# Tool 3 — Web Search Tool
# ══════════════════════════════════════════════════════════════════════

class SearchInput(BaseModel):
    query: str = Field(description="Search query for current farming laws, market prices, mandi rates, or government schemes")

class FarmWebSearchTool:
    """Real-time web search via Tavily API for prices, laws, market info."""

    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")

    def run(self, query: str) -> str:
        # Add farm context to query for better results
        farm_query = f"India farming agriculture {query}"

        if self.tavily_api_key:
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=self.tavily_api_key)
                response = client.search(
                    query=farm_query,
                    max_results=3,
                    include_domains=["agricoop.nic.in", "enam.gov.in", "farmer.gov.in",
                                     "krishijagran.com", "agrifarming.in"],
                )
                results = response.get("results", [])
                if results:
                    output = "Web search results:\n"
                    for r in results[:3]:
                        output += f"\n• {r.get('title', '')}\n  {r.get('content', '')[:300]}...\n"
                    return output
            except Exception as e:
                logger.error(f"Tavily search error: {e}")

        # Fallback: DuckDuckGo (no API key needed)
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            return search.run(farm_query)
        except Exception as e:
            return f"Web search unavailable: {str(e)}. Please check TAVILY_API_KEY."

    def as_tool(self) -> BaseTool:
        search_instance = self

        class _SearchTool(BaseTool):
            name: str = "farm_web_search"
            description: str = (
                "Search the web for current information about: "
                "crop market prices (mandi rates), eNAM platform, legal and illegal crops in India, "
                "where to buy/sell seeds and produce, APMC laws, government scheme updates, "
                "current weather advisories, and any time-sensitive farming information."
            )
            args_schema: Type[BaseModel] = SearchInput

            def _run(self, query: str) -> str:
                return search_instance.run(query)

        return _SearchTool()


# ══════════════════════════════════════════════════════════════════════
# Tool 4 — Yield Prediction Tool
# ══════════════════════════════════════════════════════════════════════

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
    Uses a trained XGBoost model (or rule-based fallback).
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

        # Try ML model first
        ml_result = self._try_ml_model(
            crop, farm_area_acres, soil, irrigation, season, fertilizer_tons, pesticide_kg
        )
        if ml_result:
            return ml_result

        # Fallback: rule-based
        base = self.BASE_YIELDS.get(crop, {}).get(soil, 2.0)
        irr_mult = self.IRRIGATION_MULTIPLIERS.get(irrigation, 1.0)

        # Fertilizer effect
        fert_per_acre = fertilizer_tons / max(farm_area_acres, 0.1)
        fert_mult = 1.0 + min(fert_per_acre * 0.05, 0.25)

        yield_per_acre = base * irr_mult * fert_mult
        total_yield = yield_per_acre * farm_area_acres
        water_usage = farm_area_acres * (
            180 if irrigation == "drip" else
            220 if irrigation == "sprinkler" else
            300 if irrigation == "flood" else 150
        )

        # Recommendations
        recs = []
        if irrigation in ["flood", "rain-fed"] and crop in ["tomato", "potato"]:
            recs.append("💧 Switch to drip irrigation — could increase yield by 25–30%")
        if fert_per_acre < 0.3:
            recs.append("🌱 Fertilizer usage is low — consider soil testing for optimal NPK")
        if not recs:
            recs.append("✅ Farm parameters look good for this crop")

        return f"""Yield Prediction ({crop_type} | {farm_area_acres} acres | {season}):

📊 Estimates:
- Yield per acre: {yield_per_acre:.2f} tons/acre
- Total yield: {total_yield:.2f} tons
- Water usage: ~{water_usage:,.0f} cubic meters

💡 Recommendations:
{chr(10).join(recs)}

⚠️ Note: These are estimates based on typical values.
Actual yield depends on weather, seed variety, and local conditions.
Contact your nearest KVK for field-specific advice.
"""

    def _try_ml_model(self, *args) -> Optional[str]:
        """Try loading a trained XGBoost model if available."""
        model_path = "outputs/yield_model.pkl"
        if not Path(model_path).exists():
            return None
        try:
            import pickle
            import numpy as np
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            # Run inference and return formatted result
            logger.info("ML yield model used")
            return None  # Extend with actual ML inference
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
