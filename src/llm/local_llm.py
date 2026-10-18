"""
LocalFarmLLM — the fine-tuned small model (see finetune/), used as a fallback.

The main agent runs on a hosted LLM (Groq). When that API is rate-limited or
down, the app used to just say "something went wrong". With LOCAL_LLM_MODEL
set, it instead answers with this model, grounded on the same RAG passages.

It is a fallback, not the primary path, on purpose: a 0.5B model can't do
reliable multi-step tool calling, but it can answer a single farming question
from retrieved context — which is exactly the degraded mode we want.

Set LOCAL_LLM_MODEL to the merged model produced by finetune/merge_adapter.py
(a Hub id like "your-name/qwen2.5-0.5b-farm" or a local directory).
"""

import logging
import os
import time
from typing import Optional

from src.llm.prompts import FT_SYSTEM_PROMPT, build_user_message

logger = logging.getLogger("farm-llm")


class LocalFarmLLM:
    def __init__(self, model_id: str, max_new_tokens: int = 256, device: Optional[str] = None):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._model = None
        self._tokenizer = None

    @classmethod
    def from_env(cls) -> Optional["LocalFarmLLM"]:
        model_id = os.getenv("LOCAL_LLM_MODEL", "").strip()
        return cls(model_id) if model_id else None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            # fp32 on CPU: fp16 matmuls are slow or unsupported on most CPUs.
            dtype = torch.float16 if device == "cuda" else torch.float32
            start = time.time()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype).to(device)
            self._model.eval()
            self.device = device
            logger.info(f"Local fallback LLM loaded: {self.model_id} on {device} in {time.time() - start:.1f}s")
        return self._model, self._tokenizer

    def answer(self, question: str, context: str = "") -> str:
        import torch

        model, tokenizer = self._load()
        messages = [
            {"role": "system", "content": FT_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, context)},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,              # deterministic: same question → same advice
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
