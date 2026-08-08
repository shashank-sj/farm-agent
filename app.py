"""
Farm Agent — Gradio UI
HuggingFace Spaces ready

Two screens in one Blocks app: a portal-style welcome screen and a chat screen,
toggled by show/hide (no server-side routing needed).
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


def enter_chat():
    return gr.update(visible=False), gr.update(visible=True)


def go_home():
    return gr.update(visible=True), gr.update(visible=False)


# ── Content ──────────────────────────────────────────────────────────────────

FEATURES = [
    ("📚", "Knowledge base", "Cultivation, soil health, fertilizers and government schemes — grounded in real farm documents."),
    ("👁️", "Photo diagnosis", "Upload a leaf or pest photo and get an instant disease or pest identification."),
    ("🔍", "Live market info", "Current mandi prices, crop laws, and scheme updates, pulled from the web."),
    ("📊", "Yield estimator", "Expected yield and water use from your crop, soil, and irrigation details."),
]

CROPS = ["Wheat", "Rice", "Tomato", "Potato", "Cotton", "Corn", "Sugarcane", "Onion", "+more"]

STATUS_OK = agent is not None
STATUS_HTML = (
    '<span class="status-dot"></span> Assistant online'
    if STATUS_OK else
    '<span class="status-dot"></span> Assistant offline — set GROQ_API_KEY'
)

HERO_HTML = """
<div class="hero">
  <span class="eyebrow">Farm Assistant</span>
  <h1>Smarter farming decisions,<br>one question away.</h1>
  <p class="dek">
    Ask about crops, soil, pests, government schemes or market prices —
    get grounded, well-structured answers, not guesses.
  </p>
</div>
"""

def feature_card_html(icon: str, title: str, desc: str) -> str:
    return f"""
    <div class="feature-card">
      <div class="feature-icon">{icon}</div>
      <div class="feature-title">{title}</div>
      <div class="feature-desc">{desc}</div>
    </div>
    """

CROPS_HTML = "".join(f'<span class="crop-chip">{c}</span>' for c in CROPS)

FOOTER_TEXT = "Farm Assistant · LangGraph + YOLOv8 + FAISS + Groq · Not a substitute for advice from your local KVK"


# ── Styling ──────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#f5f7f2",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f0f3ec",
    block_background_fill="#ffffff",
    block_border_color="#e1e7d9",
    block_border_width="1px",
    block_radius="16px",
    block_shadow="0 1px 2px rgba(20,30,20,.04), 0 6px 20px rgba(20,30,20,.05)",
    block_label_text_color="#5b6455",
    input_background_fill="#ffffff",
    input_border_color="#e1e7d9",
    input_radius="12px",
    button_primary_background_fill="#1f7a4d",
    button_primary_background_fill_hover="#186139",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_border_color="#e1e7d9",
    button_secondary_text_color="#3a453a",
    body_background_fill_dark="#10140d",
    background_fill_primary_dark="#171c13",
    background_fill_secondary_dark="#1c2217",
    block_background_fill_dark="#171c13",
    block_border_color_dark="#2a3322",
    button_secondary_background_fill_dark="#1c2217",
    button_secondary_border_color_dark="#2a3322",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');

.gradio-container { max-width: 980px !important; margin: 0 auto !important; }

/* ── Top bar ──────────────────────────────────────────────── */
.topbar {
    display: flex !important;
    align-items: center;
    justify-content: space-between;
    padding: 6px 2px 18px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.topbar .brand { flex: 0 0 auto; }
.topbar .brand p { font-size: 18px !important; font-weight: 700; margin: 0 !important; }

.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; font-weight: 600;
    padding: 5px 12px; border-radius: 999px;
    border: 1px solid #e1e7d9; background: #ffffff;
    white-space: nowrap;
}
.status-pill p { margin: 0 !important; display: flex; align-items: center; gap: 6px; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; background: #d34a34; }
.is-ok .status-dot { background: #1f8a4c; }
.is-ok { color: #186139; }
.is-bad { color: #a1402c; }

/* ── Welcome screen ───────────────────────────────────────── */
.welcome-screen { border: none !important; background: transparent !important; box-shadow: none !important; }

.hero { text-align: center; padding: 36px 12px 8px; }
.hero .eyebrow {
    font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    color: #1f8a4c;
}
.hero h1 {
    font-family: 'Fraunces', Georgia, serif;
    font-size: clamp(28px, 4.4vw, 42px);
    line-height: 1.15;
    margin: 10px 0 14px;
    color: #1c2318;
}
.hero .dek { font-size: 16px; color: #5b6455; max-width: 46ch; margin: 0 auto; }

.feature-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px !important;
    padding: 28px 4px 8px !important;
    border: none !important;
    background: transparent !important;
}
.feature-card {
    background: #ffffff;
    border: 1px solid #e1e7d9;
    border-radius: 16px;
    padding: 18px 16px;
    height: 100%;
    box-shadow: 0 1px 2px rgba(20,30,20,.04);
}
.feature-icon {
    width: 38px; height: 38px; border-radius: 10px;
    background: #e7f5ec; display: flex; align-items: center; justify-content: center;
    font-size: 18px; margin-bottom: 10px;
}
.feature-title { font-weight: 700; font-size: 14px; margin-bottom: 4px; color: #1c2318; }
.feature-desc { font-size: 13px; color: #5b6455; line-height: 1.5; }

.cta-row { display: flex !important; justify-content: center; padding: 8px 0 4px !important; border: none !important; background: transparent !important; }
.cta-btn { font-size: 15px !important; padding: 12px 28px !important; border-radius: 999px !important; }

.crops-line { text-align: center; padding: 18px 0 8px !important; border: none !important; background: transparent !important; }
.crop-chip {
    display: inline-block; font-size: 12px; color: #5b6455;
    background: #f0f3ec; border-radius: 999px; padding: 3px 11px; margin: 3px;
}

/* ── Chat screen ──────────────────────────────────────────── */
.chat-screen { border: none !important; background: transparent !important; box-shadow: none !important; }
.chat-topbar {
    display: flex !important; align-items: center; justify-content: space-between;
    padding: 4px 2px 14px !important; border: none !important; background: transparent !important; box-shadow: none !important;
}
.chat-title p { font-weight: 700; font-size: 16px; margin: 0 !important; }
.ghost-btn {
    background: transparent !important; border: 1px solid #e1e7d9 !important;
    box-shadow: none !important; font-size: 13px !important; padding: 6px 14px !important;
}

footer { display: none !important; }
.fa-footer { text-align: center; padding-top: 6px !important; border: none !important; background: transparent !important; box-shadow: none !important; }
.fa-footer p { color: #8a9282 !important; font-size: 12px !important; margin: 0 !important; }

/* ── Dark mode ────────────────────────────────────────────── */
.dark .status-pill { background: #171c13; border-color: #2a3322; }
.dark .hero h1 { color: #e9ece1; }
.dark .hero .dek { color: #9aa38f; }
.dark .feature-card { background: #171c13; border-color: #2a3322; }
.dark .feature-icon { background: #1e2b21; }
.dark .feature-title { color: #e9ece1; }
.dark .feature-desc, .dark .crop-chip { color: #9aa38f; }
.dark .crop-chip { background: #1c2217; }
.dark .fa-footer p { color: #6b7462 !important; }
"""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Farm Assistant") as demo:

    with gr.Row(elem_classes="topbar"):
        gr.Markdown("🌾 **Farm Assistant**", elem_classes="brand")
        gr.Markdown(STATUS_HTML, elem_classes=["status-pill", "is-ok" if STATUS_OK else "is-bad"], sanitize_html=False)

    # ── Screen 1: Welcome / portal ───────────────────────────────────────────
    with gr.Column(elem_classes="welcome-screen", visible=True) as welcome_screen:
        gr.Markdown(HERO_HTML, sanitize_html=False)

        with gr.Row(elem_classes="feature-grid"):
            for icon, title, desc in FEATURES:
                gr.Markdown(feature_card_html(icon, title, desc), sanitize_html=False)

        with gr.Row(elem_classes="cta-row"):
            start_btn = gr.Button("Start chatting →", variant="primary", elem_classes="cta-btn", scale=0)

        gr.Markdown(CROPS_HTML, elem_classes="crops-line", sanitize_html=False)

    # ── Screen 2: Chat ────────────────────────────────────────────────────────
    with gr.Column(elem_classes="chat-screen", visible=False) as chat_screen:
        with gr.Row(elem_classes="chat-topbar"):
            back_btn = gr.Button("← Home", elem_classes="ghost-btn", scale=0)
            gr.Markdown("Chat with Farm Assistant", elem_classes="chat-title")
            clear_btn = gr.Button("Clear", elem_classes="ghost-btn", scale=0)

        chatbot = gr.Chatbot(
            label="Chat",
            height=480,
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

    # ── State & wiring ───────────────────────────────────────────────────────
    history_state = gr.State([])

    start_btn.click(fn=enter_chat, outputs=[welcome_screen, chat_screen])
    back_btn.click(fn=go_home, outputs=[welcome_screen, chat_screen])

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

    gr.Markdown(FOOTER_TEXT, elem_classes="fa-footer")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=THEME,
        css=CSS,
    )
