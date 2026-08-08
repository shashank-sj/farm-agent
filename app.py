"""
Farm Agent — Gradio UI
HuggingFace Spaces ready
"""

import os
import tempfile
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ── Load Agent ─────────────────────────────────────────────────────────────────

agent = None
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # used for RAG embeddings


def load_agent():
    global agent
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env or HF Space secrets.")
    from src.agent import FarmAgent
    agent = FarmAgent(
        groq_api_key=GROQ_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
    )


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
            response = "Too many requests right now — please wait a moment and try again."
        else:
            response = "Something went wrong. Please try again."

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return history, history, "", None


def clear_chat():
    return [], [], "", None


# ── Gradio UI ──────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --fa-bg: #f6f8f6;
    --fa-surface: #ffffff;
    --fa-border: #e4e9e4;
    --fa-text: #1f2a20;
    --fa-muted: #6b7a6c;
    --fa-primary: #1f8a4c;
    --fa-primary-dark: #166a3b;
    --fa-primary-soft: #e7f5ec;
    --fa-radius: 14px;
    --fa-shadow: 0 1px 2px rgba(16, 24, 16, 0.04), 0 4px 16px rgba(16, 24, 16, 0.06);
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important; }

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background: var(--fa-bg) !important;
}

body, html { background: var(--fa-bg) !important; }

/* ── Cards / blocks ───────────────────────────────────────── */
.block, .gr-box, .gr-panel, .gr-form {
    background: var(--fa-surface) !important;
    border: 1px solid var(--fa-border) !important;
    border-radius: var(--fa-radius) !important;
    box-shadow: var(--fa-shadow) !important;
}

/* ── Header ───────────────────────────────────────────────── */
.fa-header h1 {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: var(--fa-text) !important;
    margin-bottom: 2px !important;
}
.fa-header p {
    color: var(--fa-muted) !important;
    font-size: 14px !important;
}

/* ── Headings ─────────────────────────────────────────────── */
h1, h2, h3, h4 { color: var(--fa-text) !important; font-weight: 600 !important; }

/* ── Body text ────────────────────────────────────────────── */
p, span, label, li { color: var(--fa-text) !important; font-size: 14px !important; line-height: 1.6 !important; }

/* ── Chatbot ──────────────────────────────────────────────── */
.chatbot, .gr-chatbot {
    background: var(--fa-surface) !important;
    border: 1px solid var(--fa-border) !important;
    border-radius: var(--fa-radius) !important;
}

.message.user, [data-testid="user"] .message {
    background: var(--fa-primary) !important;
    border: none !important;
    border-radius: 16px 16px 4px 16px !important;
    color: #ffffff !important;
}
.message.user p, .message.user span { color: #ffffff !important; }

.message.bot, [data-testid="bot"] .message {
    background: #f1f4f1 !important;
    border: none !important;
    border-radius: 16px 16px 16px 4px !important;
    color: var(--fa-text) !important;
}

/* ── Inputs ───────────────────────────────────────────────── */
input, textarea, .gr-textbox textarea, .gr-textbox input {
    background: var(--fa-surface) !important;
    border: 1px solid var(--fa-border) !important;
    border-radius: 10px !important;
    color: var(--fa-text) !important;
}
input:focus, textarea:focus {
    box-shadow: 0 0 0 3px var(--fa-primary-soft) !important;
    border-color: var(--fa-primary) !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder { color: var(--fa-muted) !important; }

/* ── Buttons ──────────────────────────────────────────────── */
button, .gr-button {
    border-radius: 10px !important;
    border: 1px solid var(--fa-border) !important;
    background: var(--fa-surface) !important;
    color: var(--fa-text) !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    transition: background 0.15s ease, transform 0.05s ease !important;
}
button:hover, .gr-button:hover { background: #f1f4f1 !important; }
button:active, .gr-button:active { transform: scale(0.98) !important; }

button.primary, .gr-button-primary {
    background: var(--fa-primary) !important;
    border-color: var(--fa-primary) !important;
    color: #ffffff !important;
}
button.primary:hover, .gr-button-primary:hover { background: var(--fa-primary-dark) !important; }

button.secondary, .gr-button-secondary {
    background: var(--fa-surface) !important;
    border-color: var(--fa-border) !important;
    color: var(--fa-muted) !important;
}
button.secondary:hover, .gr-button-secondary:hover { background: #f1f4f1 !important; color: var(--fa-text) !important; }

/* ── Labels ───────────────────────────────────────────────── */
.gr-block-label, .label-wrap, label {
    color: var(--fa-muted) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    border-bottom: none !important;
}

/* ── Sidebar cards ────────────────────────────────────────── */
.fa-card { padding: 4px 2px; }
.fa-tool-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; }
.fa-status-ok { color: var(--fa-primary) !important; font-weight: 600 !important; }
.fa-status-bad { color: #b3401f !important; font-weight: 600 !important; }
.fa-chip {
    display: inline-block;
    background: var(--fa-primary-soft);
    color: var(--fa-primary-dark) !important;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 12px !important;
    margin: 2px 4px 2px 0;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; background: var(--fa-bg); }
::-webkit-scrollbar-thumb { background: #c7d1c8; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #a9b6aa; }

/* ── Image upload ─────────────────────────────────────────── */
.gr-image, .image-container {
    border: 1.5px dashed var(--fa-border) !important;
    border-radius: var(--fa-radius) !important;
    background: var(--fa-surface) !important;
}

/* ── Examples ─────────────────────────────────────────────── */
.gr-examples .gr-sample-textbox {
    background: var(--fa-surface) !important;
    border: 1px solid var(--fa-border) !important;
    border-radius: 10px !important;
    color: var(--fa-text) !important;
}
.gr-examples .gr-sample-textbox:hover {
    border-color: var(--fa-primary) !important;
    background: var(--fa-primary-soft) !important;
}

/* ── Footer ───────────────────────────────────────────────── */
footer { display: none !important; }
.fa-footer { text-align: center; color: var(--fa-muted) !important; font-size: 12px !important; padding-top: 4px; }
"""

EXAMPLE_QUESTIONS = [
    ["What fertilizer should I use for wheat on black soil?", None],
    ["How do I control stem borer in paddy organically?", None],
    ["What is PM-KISAN and how do I apply?", None],
    ["Is cannabis cultivation legal in India?", None],
    ["My tomato farm is 5 acres, loamy soil, drip irrigation, Rabi season. What yield can I expect?", None],
    ["Where can I sell my wheat at the best price?", None],
]

with gr.Blocks(title="Farm Assistant") as demo:

    gr.Markdown(
        """
        # 🌾 Farm Assistant
        AI-powered guidance for crops, soil, pests, schemes and markets — with photo diagnosis and yield estimates.
        """,
        elem_classes="fa-header",
    )

    with gr.Row():
        # ── Left: Chat ──────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=520,
                show_label=False,
                avatar_images=("🧑‍🌾", "🌾"),
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask about crops, soil, pests, laws, market prices...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Accordion("📷 Attach a plant/pest photo (optional)", open=False):
                image_input = gr.Image(type="pil", show_label=False, height=180)

            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=[msg_input, image_input],
                label="Try asking",
            )

        # ── Right: Info Panel ────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("**Status**")
            if agent is not None:
                gr.Markdown("🟢 Agent ready", elem_classes="fa-status-ok")
            else:
                gr.Markdown("🔴 Agent not loaded — check GROQ_API_KEY", elem_classes="fa-status-bad")

            gr.Markdown("**Tools**")
            gr.Markdown(
                """
                <div class="fa-card">
                <div class="fa-tool-row">📚 <b>Knowledge base</b> — cultivation, soil, schemes</div>
                <div class="fa-tool-row">👁️ <b>Vision</b> — pest &amp; disease photo scan</div>
                <div class="fa-tool-row">🔍 <b>Web search</b> — live prices &amp; laws</div>
                <div class="fa-tool-row">📊 <b>Yield estimator</b> — crop &amp; farm data</div>
                </div>
                """
            )

            gr.Markdown("**Crops covered**")
            gr.Markdown(
                """
                <span class="fa-chip">Wheat</span><span class="fa-chip">Rice</span><span class="fa-chip">Tomato</span>
                <span class="fa-chip">Potato</span><span class="fa-chip">Cotton</span><span class="fa-chip">Corn</span>
                <span class="fa-chip">Sugarcane</span><span class="fa-chip">Onion</span><span class="fa-chip">+more</span>
                """
            )

            clear_btn = gr.Button("Clear chat", variant="secondary")

    # State
    history_state = gr.State([])

    send_btn.click(
        fn=chat,
        inputs=[msg_input, image_input, history_state],
        outputs=[chatbot, history_state, msg_input, image_input],
    )

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, image_input, history_state],
        outputs=[chatbot, history_state, msg_input, image_input],
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, history_state, msg_input, image_input],
    )

    gr.Markdown(
        "Farm Assistant · LangGraph + YOLOv8 + FAISS + Groq · Not a substitute for advice from your local KVK",
        elem_classes="fa-footer",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="emerald"),
        css=CSS,
    )
