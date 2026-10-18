import pytest

from finetune.data import IGNORE_INDEX, CompletionCollator, build_example
from finetune.metrics import lcs_length, rouge_l_f1, token_f1


class CharTokenizer:
    """Character-level tokenizer with a ChatML-like template."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class BrokenTemplateTokenizer(CharTokenizer):
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        text = super().apply_chat_template(messages, add_generation_prompt, tokenize)
        return text if add_generation_prompt else "<bos>" + text


def decode(ids):
    return "".join(chr(i) for i in ids)


def test_only_answer_tokens_are_scored():
    ex = build_example(CharTokenizer(), "Why rotate crops?", "To break pest cycles.", max_len=1000)
    scored = [t for t in ex["labels"] if t != IGNORE_INDEX]
    assert decode(scored) == "To break pest cycles.<|end|>"     # answer + end-of-turn, nothing else
    n_masked = ex["labels"].index(scored[0])
    assert decode(ex["input_ids"][:n_masked]).endswith("<|assistant|>")
    assert len(ex["input_ids"]) == len(ex["labels"])


def test_long_answers_are_truncated_and_long_prompts_skipped():
    tok = CharTokenizer()
    ex = build_example(tok, "q?", "a" * 5000, max_len=300)
    assert len(ex["input_ids"]) == 300 and ex["truncated"]
    assert build_example(tok, "q" * 400, "answer", max_len=300) is None


def test_template_that_is_not_prefix_consistent_is_rejected():
    with pytest.raises(ValueError):
        build_example(BrokenTemplateTokenizer(), "q?", "a", max_len=1000)


def test_collator_pads_with_ignored_labels():
    pytest.importorskip("torch")
    batch = CompletionCollator(pad_token_id=0)([
        {"input_ids": [5, 6, 7], "labels": [IGNORE_INDEX, 6, 7]},
        {"input_ids": [5], "labels": [5]},
    ])
    assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 6, 7], [5, IGNORE_INDEX, IGNORE_INDEX]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_overlap_metrics():
    assert lcs_length("a b c d".split(), "a c d".split()) == 3
    assert rouge_l_f1("apply urea in two splits", "apply urea in two splits") == pytest.approx(1.0)
    assert rouge_l_f1("completely different words", "apply urea") == 0.0
    assert token_f1("urea apply", "apply urea") == pytest.approx(1.0)       # order-insensitive
    assert rouge_l_f1("urea apply", "apply urea") == pytest.approx(0.5)     # order-sensitive
