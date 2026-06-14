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

/* ── Palette ──────────────────────────────────────────────────
   bg:      #fdf6ec   (light cream)
   surface: #fff8f0   (card background)
   orange:  #e86500   (primary text & accent)
   amber:   #ff9500   (secondary accent)
   rust:    #c94a00   (danger/secondary)
──────────────────────────────────────────────────────────────── */

* { image-rendering: pixelated; }

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 8px !important;
    background: #fdf6ec !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px !important;
    box-sizing: border-box !important;
}

body, html { background: #fdf6ec !important; }

/* ── Blocks ───────────────────────────────────────────────── */
.block, .gr-box, .gr-panel, .gr-form {
    background: #fff8f0 !important;
    border: 3px solid #e86500 !important;
    border-radius: 4px !important;
    box-shadow: 3px 3px 0px #c94a00 !important;
}

/* ── Headings ─────────────────────────────────────────────── */
h1, h2, h3, h4, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    font-family: 'Press Start 2P', monospace !important;
    color: #e86500 !important;
    letter-spacing: 2px !important;
}

/* ── Body text ────────────────────────────────────────────── */
p, span, label, .gr-markdown p, .gr-markdown li {
    font-family: 'Press Start 2P', monospace !important;
    color: #e86500 !important;
    font-size: 9px !important;
    line-height: 1.8 !important;
}

/* ── Chatbot ──────────────────────────────────────────────── */
.chatbot, .gr-chatbot {
    background: #fff8f0 !important;
    border: 3px solid #e86500 !important;
    border-radius: 4px !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
}

/* User bubble */
.message.user, [data-testid="user"] .message {
    background: #ffe0b2 !important;
    border: 2px solid #e86500 !important;
    border-radius: 4px !important;
    color: #7a3000 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    box-shadow: 2px 2px 0 #c94a00 !important;
}

/* Bot bubble */
.message.bot, [data-testid="bot"] .message {
    background: #fff3e0 !important;
    border: 2px solid #ff9500 !important;
    border-radius: 4px !important;
    color: #7a3000 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    box-shadow: 2px 2px 0 #c94a00 !important;
}

/* ── Inputs ───────────────────────────────────────────────── */
input, textarea, .gr-textbox textarea, .gr-textbox input {
    background: #fff8f0 !important;
    border: 2px solid #e86500 !important;
    border-radius: 4px !important;
    color: #7a3000 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    caret-color: #e86500 !important;
}

input:focus, textarea:focus {
    box-shadow: 0 0 0 2px #e86500 !important;
    outline: none !important;
}

input::placeholder, textarea::placeholder {
    color: #c9956a !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
button, .gr-button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    border-radius: 4px !important;
    border: 2px solid #e86500 !important;
    background: #fff3e0 !important;
    color: #e86500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    box-shadow: 3px 3px 0 #c94a00 !important;
    transition: none !important;
    cursor: pointer !important;
    padding: 10px 14px !important;
}

button:hover, .gr-button:hover {
    background: #e86500 !important;
    color: #fdf6ec !important;
    box-shadow: 1px 1px 0 #c94a00 !important;
    transform: translate(2px, 2px) !important;
}

button:active, .gr-button:active {
    transform: translate(3px, 3px) !important;
    box-shadow: none !important;
}

/* Primary buttons */
button.primary, .gr-button-primary {
    background: #e86500 !important;
    border-color: #c94a00 !important;
    color: #fdf6ec !important;
    box-shadow: 3px 3px 0 #7a3000 !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: #c94a00 !important;
    color: #fdf6ec !important;
}

/* Secondary buttons */
button.secondary, .gr-button-secondary {
    background: #fff3e0 !important;
    border-color: #c94a00 !important;
    color: #c94a00 !important;
    box-shadow: 3px 3px 0 #7a3000 !important;
}
button.secondary:hover, .gr-button-secondary:hover {
    background: #c94a00 !important;
    color: #fdf6ec !important;
}

/* ── Labels ───────────────────────────────────────────────── */
.gr-block-label, .label-wrap, label {
    color: #e86500 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border-bottom: 2px solid #e86500 !important;
    padding-bottom: 4px !important;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; background: #fdf6ec; }
::-webkit-scrollbar-thumb { background: #e86500; border: 2px solid #fdf6ec; }
::-webkit-scrollbar-thumb:hover { background: #ff9500; }

/* ── Image upload ─────────────────────────────────────────── */
.gr-image, .image-container {
    border: 3px dashed #e86500 !important;
    border-radius: 4px !important;
    background: #fff8f0 !important;
}

/* ── Examples ─────────────────────────────────────────────── */
.gr-examples .gr-sample-textbox {
    background: #fff3e0 !important;
    border: 2px solid #e86500 !important;
    border-radius: 4px !important;
    color: #7a3000 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
}
.gr-examples .gr-sample-textbox:hover {
    border-color: #c94a00 !important;
    background: #ffe0b2 !important;
}

/* ── Dividers ─────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 2px solid #e86500 !important;
    margin: 12px 0 !important;
}

/* ── Footer ───────────────────────────────────────────────── */
footer { display: none !important; }

/* ── Info text ────────────────────────────────────────────── */
.gr-info, .gr-form .gr-info {
    color: #c9956a !important;
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
