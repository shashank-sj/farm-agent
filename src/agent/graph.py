"""
Farm Agent — LangGraph State Machine
Routes user queries to the right tool automatically:
  - Text farming questions  → RAG tool
  - Legal / market queries  → Web search tool
  - Image uploads           → Vision tool
  - Yield / data questions  → ML prediction tool
"""

import os
import logging
from typing import Annotated, TypedDict, Optional, Literal
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.tools.rag_tool import FarmRAGTool
from src.tools.vision_tool import FarmVisionTool
from src.tools.search_tool import FarmWebSearchTool
from src.tools.yield_tool import YieldPredictionTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("farm-agent")


# ── Agent State ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    image_path: Optional[str]       # Set when user uploads an image
    tool_used: Optional[str]        # Track which tool was called
    num_steps: int                  # Prevent infinite loops


# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Farm Assistant AI helping Indian farmers with:
1. Crop cultivation, soil health, fertilizers, irrigation, pest/disease management
2. Legal crops, illegal crops, where to sell/buy crops and seeds (mandis, eNAM)
3. Government schemes (PM-KISAN, PMFBY, SMAM, soil health cards)
4. Plant disease and pest identification from photos

You have access to these tools:
- **farm_rag**: Search your knowledge base of farm documents for detailed answers
- **farm_vision**: Analyse a plant/pest photo for disease or pest identification
- **farm_web_search**: Search the web for current prices, laws, schemes, market info
- **yield_prediction**: Predict crop yield given farm parameters

Rules:
- ALWAYS use a tool — never answer from memory alone
- For image questions → use farm_vision first, then farm_rag for treatment details
- For legal/market/price questions → use farm_web_search
- For cultivation/soil/pest/disease questions → use farm_rag
- For yield/data questions → use yield_prediction
- Be practical, farmer-friendly, and mention local resources (KVK, eNAM, mandi)
- If unsure, say so — never hallucinate crop advice (it affects livelihoods)
- Answer in simple English; use Hindi terms where helpful (e.g., Kharif, Rabi, Zaid)
"""


# ── Farm Agent ─────────────────────────────────────────────────────────────────

class FarmAgent:
    """
    LangGraph-powered multi-tool farm assistant.
    Automatically routes queries to the right tool.
    """

    def __init__(
        self,
        gemini_api_key: str,
        rag_index_path: str = "data/faiss_index",
        vision_model_path: str = "outputs/farm-vision/weights/best.pt",
        llm_model_path: str = "outputs/gemma-farm-qlora/final",
        use_local_llm: bool = False,
        max_steps: int = 10,
    ):
        self.max_steps = max_steps
        os.environ["GOOGLE_API_KEY"] = gemini_api_key

        # LLM — use fine-tuned Gemma if available, else Gemini fallback
        if use_local_llm and Path(llm_model_path).exists():
            logger.info(f"Loading fine-tuned Gemma from {llm_model_path}")
            self.llm = self._load_local_llm(llm_model_path)
        else:
            logger.info("Using Gemini 2.0 Flash as agent LLM")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.2,
                convert_system_message_to_human=True,
            )

        # Tools
        self.rag_tool = FarmRAGTool(
            gemini_api_key=gemini_api_key,
            index_path=rag_index_path,
        )
        self.vision_tool = FarmVisionTool(model_path=vision_model_path)
        self.search_tool = FarmWebSearchTool()
        self.yield_tool = YieldPredictionTool()

        self.tools = [
            self.rag_tool.as_tool(),
            self.vision_tool.as_tool(),
            self.search_tool.as_tool(),
            self.yield_tool.as_tool(),
        ]

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()
        logger.info("Farm Agent ready ✓")

    def _load_local_llm(self, model_path: str):
        """Load fine-tuned Gemma-2B as LangChain LLM."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from peft import PeftModel
        from langchain_community.llms import HuggingFacePipeline

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b-it",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, model_path)
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=512, temperature=0.2, do_sample=True,
        )
        return HuggingFacePipeline(pipeline=pipe)

    # ── Graph Nodes ────────────────────────────────────────────────────────────

    def _agent_node(self, state: AgentState) -> AgentState:
        """Main agent node — decides which tool to call."""
        messages = state["messages"]

        # Inject image context if present
        if state.get("image_path") and not any(
            "image uploaded" in str(m.content).lower() for m in messages
        ):
            messages = messages + [
                HumanMessage(content=f"[Image uploaded: {state['image_path']}] Please analyse this image using the farm_vision tool.")
            ]

        # Add system prompt as first message if not present
        if not any(hasattr(m, "type") and m.type == "system" for m in messages):
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = self.llm_with_tools.invoke(messages)
        logger.info(f"Agent response type: {type(response).__name__}")

        return {
            "messages": [response],
            "num_steps": state.get("num_steps", 0) + 1,
        }

    def _should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """Decide whether to call a tool or return final answer."""
        messages = state["messages"]
        last = messages[-1]

        # Safety: stop if too many steps
        if state.get("num_steps", 0) >= self.max_steps:
            logger.warning("Max steps reached — forcing end")
            return "end"

        # If last message has tool calls → go to tools
        if hasattr(last, "tool_calls") and last.tool_calls:
            tool_name = last.tool_calls[0].get("name", "unknown")
            logger.info(f"Calling tool: {tool_name}")
            return "tools"

        return "end"

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        tool_node = ToolNode(self.tools)

        graph = StateGraph(AgentState)

        # Nodes
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", tool_node)

        # Edges
        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")   # After tool → back to agent

        return graph.compile()

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, message: str, image_path: str = None, history: list = None) -> str:
        """
        Main entry point for the agent.
        Returns: final answer string
        """
        # Build message history
        messages = []
        if history:
            for entry in history:
                if isinstance(entry, dict):
                    # New Gradio 5.x format: {"role": "user"/"assistant", "content": "..."}
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                    if role == "user" and content:
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant" and content:
                        messages.append(AIMessage(content=content))
                else:
                    # Legacy tuple format: (human, ai)
                    human, ai = entry
                    if human:
                        messages.append(HumanMessage(content=human))
                    if ai:
                        messages.append(AIMessage(content=ai))

        messages.append(HumanMessage(content=message))

        initial_state = AgentState(
            messages=messages,
            image_path=image_path,
            tool_used=None,
            num_steps=0,
        )

        # Run graph
        final_state = self.graph.invoke(initial_state)

        # Extract final answer
        last_message = final_state["messages"][-1]
        answer = last_message.content if hasattr(last_message, "content") else str(last_message)

        logger.info(f"Steps taken: {final_state.get('num_steps', 0)}")
        return answer

    def stream(self, message: str, image_path: str = None, history: list = None):
        """Stream the agent's response token by token."""
        messages = []
        if history:
            for entry in history:
                if isinstance(entry, dict):
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                    if role == "user" and content:
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant" and content:
                        messages.append(AIMessage(content=content))
                else:
                    human, ai = entry
                    if human:
                        messages.append(HumanMessage(content=human))
                    if ai:
                        messages.append(AIMessage(content=ai))
        messages.append(HumanMessage(content=message))

        initial_state = AgentState(
            messages=messages,
            image_path=image_path,
            tool_used=None,
            num_steps=0,
        )

        for chunk in self.graph.stream(initial_state):
            if "agent" in chunk:
                msgs = chunk["agent"].get("messages", [])
                for msg in msgs:
                    if hasattr(msg, "content") and msg.content:
                        yield msg.content
