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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # used for RAG embeddings

def load_agent():
    global agent
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Add it to HF Space secrets.")
    from src.agent.graph import FarmAgent
    agent = FarmAgent(
        groq_api_key=GROQ_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
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
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            response = "⚠️ Too many requests. Please wait a moment and try again."
        else:
            response = f"Something went wrong. Please try again."

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
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

/* ── Palette ──────────────────────────────────────────────────
   bg:      #1a1209   (dark warm brown)
   surface: #2b1f0e   (card background)
   cream:   #f5e6c8   (primary text)
   orange:  #ff8c00   (accent)
   amber:   #ffb347   (secondary accent)
   rust:    #cc4e00   (danger/secondary)
──────────────────────────────────────────────────────────────── */

* { image-rendering: pixelated; }

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    background: #1a1209 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px !important;
}

body, .dark { background: #1a1209 !important; }

/* ── Blocks ───────────────────────────────────────────────── */
.block, .gr-box, .gr-panel, .gr-form {
    background: #2b1f0e !important;
    border: 4px solid #ff8c00 !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0px #7a3d00, inset 0 0 20px rgba(255,140,0,0.04) !important;
}

/* ── Headings ─────────────────────────────────────────────── */
h1, h2, h3, h4, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    font-family: 'Press Start 2P', monospace !important;
    color: #ff8c00 !important;
    text-shadow: 0 0 10px #ff8c00, 0 0 20px #ffb347 !important;
    letter-spacing: 2px !important;
}

/* ── Body text ────────────────────────────────────────────── */
p, span, label, .gr-markdown p, .gr-markdown li {
    font-family: 'Press Start 2P', monospace !important;
    color: #f5e6c8 !important;
    font-size: 9px !important;
    line-height: 1.8 !important;
}

/* ── Chatbot ──────────────────────────────────────────────── */
.chatbot, .gr-chatbot {
    background: #120d06 !important;
    border: 4px solid #ff8c00 !important;
    border-radius: 0 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
}

/* User bubble */
.message.user, [data-testid="user"] .message {
    background: #3d2200 !important;
    border: 3px solid #ff8c00 !important;
    border-radius: 0 !important;
    color: #ff8c00 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    box-shadow: 3px 3px 0 #7a3d00 !important;
}

/* Bot bubble */
.message.bot, [data-testid="bot"] .message {
    background: #261600 !important;
    border: 3px solid #ffb347 !important;
    border-radius: 0 !important;
    color: #f5e6c8 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    box-shadow: 3px 3px 0 #7a3d00 !important;
}

/* ── Inputs ───────────────────────────────────────────────── */
input, textarea, .gr-textbox textarea, .gr-textbox input {
    background: #120d06 !important;
    border: 3px solid #ff8c00 !important;
    border-radius: 0 !important;
    color: #f5e6c8 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    caret-color: #ff8c00 !important;
}

input:focus, textarea:focus {
    box-shadow: 0 0 0 2px #ff8c00, 0 0 15px rgba(255,140,0,0.4) !important;
    outline: none !important;
}

input::placeholder, textarea::placeholder {
    color: #7a4d1a !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
button, .gr-button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    border-radius: 0 !important;
    border: 3px solid #ff8c00 !important;
    background: #3d2200 !important;
    color: #ff8c00 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    box-shadow: 4px 4px 0 #7a3d00 !important;
    transition: none !important;
    cursor: pointer !important;
    padding: 10px 14px !important;
}

button:hover, .gr-button:hover {
    background: #ff8c00 !important;
    color: #1a1209 !important;
    box-shadow: 2px 2px 0 #7a3d00 !important;
    transform: translate(2px, 2px) !important;
}

button:active, .gr-button:active {
    transform: translate(4px, 4px) !important;
    box-shadow: none !important;
}

/* Primary buttons */
button.primary, .gr-button-primary {
    background: #4d2900 !important;
    border-color: #ff8c00 !important;
    color: #ff8c00 !important;
    box-shadow: 4px 4px 0 #ff8c00 !important;
}

/* Secondary buttons */
button.secondary, .gr-button-secondary {
    background: #2b0a00 !important;
    border-color: #cc4e00 !important;
    color: #cc4e00 !important;
    box-shadow: 4px 4px 0 #7a1a00 !important;
}
button.secondary:hover, .gr-button-secondary:hover {
    background: #cc4e00 !important;
    color: #1a1209 !important;
}

/* ── Labels ───────────────────────────────────────────────── */
.gr-block-label, .label-wrap, label {
    color: #ff8c00 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border-bottom: 2px solid #ff8c00 !important;
    padding-bottom: 4px !important;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; background: #1a1209; }
::-webkit-scrollbar-thumb { background: #ff8c00; border: 2px solid #1a1209; }
::-webkit-scrollbar-thumb:hover { background: #ffb347; }

/* ── Image upload ─────────────────────────────────────────── */
.gr-image, .image-container {
    border: 4px dashed #ff8c00 !important;
    border-radius: 0 !important;
    background: #120d06 !important;
}

/* ── Examples ─────────────────────────────────────────────── */
.gr-examples .gr-sample-textbox {
    background: #261600 !important;
    border: 2px solid #7a3d00 !important;
    border-radius: 0 !important;
    color: #f5e6c8 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
}
.gr-examples .gr-sample-textbox:hover {
    border-color: #ff8c00 !important;
    background: #3d2200 !important;
}

/* ── Dividers ─────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 3px solid #ff8c00 !important;
    box-shadow: 0 0 8px rgba(255,140,0,0.5) !important;
    margin: 12px 0 !important;
}

/* ── Footer ───────────────────────────────────────────────── */
footer { display: none !important; }

/* ── Scanlines ────────────────────────────────────────────── */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.12) 2px,
        rgba(0,0,0,0.12) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── CRT warm glow ────────────────────────────────────────── */
.gradio-container::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    box-shadow: inset 0 0 120px rgba(255,140,0,0.05);
    pointer-events: none;
    z-index: 9998;
}

/* ── Info text ────────────────────────────────────────────── */
.gr-info, .gr-form .gr-info {
    color: #7a4d1a !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 7px !important;
}
"""

EXAMPLE_QUESTIONS = [
    ["What fertilizer should I use for wheat on black soil?", None],
    ["How do I control stem borer in paddy organically?", None],
    ["What is PM-KISAN and how do I apply?", None],
    ["Is cannabis cultivation legal in India?", None],
    ["My tomato farm is 5 acres, loamy soil, drip irrigation, Rabi season. What yield can I expect?", None],
    ["Where can I sell my wheat at the best price?", None],
]

with gr.Blocks(title="🌾 FARM QUEST AI", theme=gr.themes.Base(), css=CSS) as demo:

    gr.Markdown("""
    # 🌾 FARM QUEST AI
    ### >>> LEVEL 1: CROP MASTER <<<
    *RAG + VISION + WEB SEARCH ENGINE LOADED*

    > INSERT QUESTION TO CONTINUE...

    QUERY crops · soil · pests · diseases · laws · markets
    """)

    with gr.Row():
        # ── Left: Chat ──────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Farm Assistant",
                height=520,
                show_label=True,
                avatar_images=("👨‍🌾", "🌾"),
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask about crops, soil, pests, laws, market prices...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("[ SEND >> ]", variant="primary", scale=1)

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
            gr.Markdown("### [ STATUS ]")
            agent_status = "✅ Agent ready" if agent is not None else "⚠️ Agent not loaded — check GEMINI_API_KEY in .env"
            gr.Markdown(agent_status)

            gr.Markdown("---")
            gr.Markdown("### [ TOOLS ]")
            gr.Markdown("""
            > RAG    — Knowledge base

            > VISION — Pest scanner

            > SEARCH — Live prices

            > YIELD  — Crop calc
            """)

            gr.Markdown("---")
            gr.Markdown("### [ CROPS ]")
            gr.Markdown("""
            WHEAT · RICE · TOMATO

            POTATO · COTTON · CORN

            SUGARCANE · ONION · MORE
            """)

            clear_btn = gr.Button("[ RESET GAME ]", variant="secondary")

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
    © FARM QUEST AI v1.0 | ENGINE: LangGraph+YOLOv8+FAISS+Gemini | DATASET: PlantVillage
    </center>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
