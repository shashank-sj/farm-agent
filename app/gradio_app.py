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

def load_agent(api_key: str):
    global agent
    from src.agent.graph import FarmAgent
    agent = FarmAgent(
        gemini_api_key=api_key,
        use_local_llm=False,   # Set True if Gemma weights available
    )
    return "✅ Farm Assistant ready! Ask me anything about farming."


# ── Chat Function ──────────────────────────────────────────────────────────────

def chat(message: str, image, history: list, api_key: str):
    global agent

    if not api_key:
        return history, history, "Please enter your Gemini API key in the sidebar."

    if agent is None:
        try:
            load_agent(api_key)
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

/* ── Global Reset ─────────────────────────────────────────── */
* { image-rendering: pixelated; }

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    background: #0a0a0a !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px !important;
}

body, .dark {
    background: #0a0a0a !important;
}

/* ── Pixel border mixin via box-shadow ────────────────────── */
.block, .gr-box, .gr-panel, .gr-form {
    background: #111 !important;
    border: 4px solid #39ff14 !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0px #1a7a00, inset 0 0 20px rgba(57,255,20,0.05) !important;
}

/* ── Headings ─────────────────────────────────────────────── */
h1, h2, h3, h4, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    font-family: 'Press Start 2P', monospace !important;
    color: #39ff14 !important;
    text-shadow: 0 0 10px #39ff14, 0 0 20px #39ff14 !important;
    letter-spacing: 2px !important;
}

/* ── Body text ────────────────────────────────────────────── */
p, span, label, .gr-markdown p, .gr-markdown li {
    font-family: 'Press Start 2P', monospace !important;
    color: #a0ff70 !important;
    font-size: 9px !important;
    line-height: 1.8 !important;
}

/* ── Chatbot ──────────────────────────────────────────────── */
.chatbot, .gr-chatbot {
    background: #060f04 !important;
    border: 4px solid #39ff14 !important;
    border-radius: 0 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
}

/* User bubble */
.message.user, [data-testid="user"] .message {
    background: #003300 !important;
    border: 3px solid #39ff14 !important;
    border-radius: 0 !important;
    color: #39ff14 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    box-shadow: 3px 3px 0 #1a7a00 !important;
}

/* Bot bubble */
.message.bot, [data-testid="bot"] .message {
    background: #001a00 !important;
    border: 3px solid #00cc44 !important;
    border-radius: 0 !important;
    color: #a0ff70 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    box-shadow: 3px 3px 0 #005522 !important;
}

/* ── Inputs ───────────────────────────────────────────────── */
input, textarea, .gr-textbox textarea, .gr-textbox input {
    background: #050f05 !important;
    border: 3px solid #39ff14 !important;
    border-radius: 0 !important;
    color: #39ff14 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    caret-color: #39ff14 !important;
}

input:focus, textarea:focus {
    box-shadow: 0 0 0 2px #39ff14, 0 0 15px #39ff14 !important;
    outline: none !important;
}

input::placeholder, textarea::placeholder {
    color: #2a6600 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
button, .gr-button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    border-radius: 0 !important;
    border: 3px solid #39ff14 !important;
    background: #003300 !important;
    color: #39ff14 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    box-shadow: 4px 4px 0 #1a7a00 !important;
    transition: none !important;
    cursor: pointer !important;
    padding: 10px 14px !important;
}

button:hover, .gr-button:hover {
    background: #39ff14 !important;
    color: #000 !important;
    box-shadow: 2px 2px 0 #1a7a00 !important;
    transform: translate(2px, 2px) !important;
}

button:active, .gr-button:active {
    transform: translate(4px, 4px) !important;
    box-shadow: none !important;
}

/* Primary buttons */
button.primary, .gr-button-primary {
    background: #004400 !important;
    border-color: #39ff14 !important;
    color: #39ff14 !important;
    box-shadow: 4px 4px 0 #39ff14 !important;
}

/* Secondary buttons */
button.secondary, .gr-button-secondary {
    background: #1a0000 !important;
    border-color: #ff4444 !important;
    color: #ff4444 !important;
    box-shadow: 4px 4px 0 #880000 !important;
}
button.secondary:hover, .gr-button-secondary:hover {
    background: #ff4444 !important;
    color: #000 !important;
}

/* ── Labels ───────────────────────────────────────────────── */
.gr-block-label, .label-wrap, label {
    color: #39ff14 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border-bottom: 2px solid #39ff14 !important;
    padding-bottom: 4px !important;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #39ff14; border: 2px solid #0a0a0a; }
::-webkit-scrollbar-thumb:hover { background: #a0ff70; }

/* ── Image upload ─────────────────────────────────────────── */
.gr-image, .image-container {
    border: 4px dashed #39ff14 !important;
    border-radius: 0 !important;
    background: #060f04 !important;
}

/* ── Examples ─────────────────────────────────────────────── */
.gr-examples .gr-sample-textbox {
    background: #001a00 !important;
    border: 2px solid #1a7a00 !important;
    border-radius: 0 !important;
    color: #a0ff70 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
}

.gr-examples .gr-sample-textbox:hover {
    border-color: #39ff14 !important;
    background: #003300 !important;
}

/* ── Dividers ─────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 3px solid #39ff14 !important;
    box-shadow: 0 0 8px #39ff14 !important;
    margin: 12px 0 !important;
}

/* ── Footer ───────────────────────────────────────────────── */
footer { display: none !important; }

/* ── Scanline overlay effect ──────────────────────────────── */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.15) 2px,
        rgba(0,0,0,0.15) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── CRT glow on container ────────────────────────────────── */
.gradio-container::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    box-shadow: inset 0 0 100px rgba(57,255,20,0.04);
    pointer-events: none;
    z-index: 9998;
}

/* ── Info text ────────────────────────────────────────────── */
.gr-info, .gr-form .gr-info {
    color: #2a6600 !important;
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

with gr.Blocks(
    title="🌾 FARM QUEST AI",
    theme=gr.themes.Base(),
    css=CSS,
) as demo:

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

        # ── Right: Settings ─────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### [ CONFIG ]")
            api_key = gr.Textbox(
                label="Gemini API Key",
                type="password",
                value=os.getenv("GEMINI_API_KEY", ""),
                info="Get free key at aistudio.google.com",
            )
            init_btn = gr.Button("[ BOOT AGENT ]", variant="primary")
            status = gr.Textbox(label="Status", interactive=False, lines=2)

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

    # Events
    init_btn.click(
        fn=load_agent,
        inputs=[api_key],
        outputs=[status],
    )

    send_btn.click(
        fn=chat,
        inputs=[msg_input, image_input, history_state, api_key],
        outputs=[chatbot, history_state, msg_input],
    )

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, image_input, history_state, api_key],
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
