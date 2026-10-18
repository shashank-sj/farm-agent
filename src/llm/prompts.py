"""Prompts shared by fine-tuning (finetune/) and serving (src/llm/local_llm.py).

Train/serve consistency: the fine-tuned model only ever saw this exact system
prompt and chat template during training, so serving must use the same ones —
a different prompt at inference is a distribution shift the model never saw.
"""

FT_SYSTEM_PROMPT = (
    "You are a farm assistant for Indian farmers. Answer the farming question "
    "accurately, practically and concisely in English."
)

RAG_USER_TEMPLATE = (
    "Use the reference notes below if they are relevant; ignore them if not.\n\n"
    "Reference notes:\n{context}\n\n"
    "Question: {question}"
)


def build_user_message(question: str, context: str = "") -> str:
    return RAG_USER_TEMPLATE.format(context=context, question=question) if context else question
