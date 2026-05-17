"""
Farm Agent — Gradio UI
HuggingFace Spaces ready
"""

import os
import tempfile
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Load Agent ─────────────────────────────────────────────────────────────────

agent = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def load_agent():
    global agent
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env file.")
    from src.agent.graph import FarmAgent
    agent = FarmAgent(
        gemini_api_key=GEMINI_API_KEY,
        use_local_llm=False,   # Set True if Gemma weights available
    )

# Auto-initialize agent on startup
try:
    load_agent()
except Exception as _e:
    print(f"[WARNING] Could not auto-load agent: {_e}")


# ── Chat Function ──────────────────────────────────────────────────────────────

def chat(message: str, image, history: list):
    global agent

    if agent is None:
        try:
            load_agent()
        except Exception as e:
            return history, history, f"Failed to load agent: {e}"

    # Handle image
    image_path = None
    if image is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            if hasattr(image, "save"):
                image.save(tmp.name)
            else:
                import shutil
                shutil.copy(image, tmp.name)
            image_path = tmp.name

    if not message and image_path:
        message = "Please analyse this image and tell me what's wrong with this plant."

    if not message:
        return history, history, ""

    try:
        response = agent.chat(
            message=message,
            image_path=image_path,
            history=history,
        )
    except Exception as e:
        response = f"Error: {str(e)}. Please try again."

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return history, history, ""


def clear_chat():
    global agent
    agent = None
    return [], [], ""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1000px; margin: auto; }
.title { text-align: center; color: #2d5a1b; }
footer { display: none !important; }
"""

EXAMPLE_QUESTIONS = [
    ["What fertilizer should I use for wheat on black soil?", None],
    ["How do I control stem borer in paddy organically?", None],
    ["What is PM-KISAN and how do I apply?", None],
    ["Is cannabis cultivation legal in India?", None],
    ["My tomato farm is 5 acres, loamy soil, drip irrigation, Rabi season. What yield can I expect?", None],
    ["Where can I sell my wheat at the best price?", None],
]

with gr.Blocks(title="🌾 Farm Assistant AI") as demo:

    gr.Markdown("""
    # 🌾 Farm Assistant AI
    *Your intelligent farming companion — powered by RAG + Vision + Web Search*
    
    Ask anything about **crops, soil, pests, diseases, laws, markets, and schemes**.
    Upload a **plant photo** for instant disease/pest identification.
    """)

    with gr.Row():
        # ── Left: Chat ──────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Farm Assistant",
                height=520,
                show_label=True,
                avatar_images=("👨‍🌾", "🌾"),
                type="messages",
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask about crops, soil, pests, laws, market prices...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send 🌱", variant="primary", scale=1)

            image_input = gr.Image(
                type="pil",
                label="📷 Upload plant/pest photo (optional)",
                height=180,
            )

            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=[msg_input, image_input],
                label="💡 Example questions",
            )

        # ── Right: Info Panel ────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Status")
            agent_status = "✅ Agent ready" if agent is not None else "⚠️ Agent not loaded — check GEMINI_API_KEY in .env"
            gr.Markdown(agent_status)

            gr.Markdown("---")
            gr.Markdown("### 🛠️ Tools Available")
            gr.Markdown("""
            - 📚 **RAG** — Farm knowledge base
            - 👁️ **Vision** — Disease & pest detection  
            - 🔍 **Web Search** — Live prices & laws
            - 📊 **Yield Predictor** — Crop estimates
            """)

            gr.Markdown("---")
            gr.Markdown("### 📋 Supported Crops")
            gr.Markdown("""
            Wheat, Rice, Tomato, Potato,  
            Cotton, Sugarcane, Corn, Onion,  
            Soybean, Mustard, and more
            """)

            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

    # State
    history_state = gr.State([])

    send_btn.click(
        fn=chat,
        inputs=[msg_input, image_input, history_state],
        outputs=[chatbot, history_state, msg_input],
    )

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, image_input, history_state],
        outputs=[chatbot, history_state, msg_input],
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, history_state, msg_input],
    )

    gr.Markdown("""
    ---
    <center>
    Built with LangGraph + YOLOv8 + FAISS + Gemini | 
    Trained on PlantVillage + Agricultural Pests datasets
    </center>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"),
        css=CSS,
    )
