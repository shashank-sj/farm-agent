"""Text-overlap metrics, written from scratch (no `evaluate` / `rouge_score` dependency).

Both reward word overlap with the reference answer, so a correct paraphrase
scores lower than it should. That's why finetune/evaluate.py also reports
answer perplexity and (optionally) an LLM-judge score.
"""

import re
from collections import Counter

_WORD_RE = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length, O(len(a)·len(b)) time, O(len(b)) memory."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, 1):
            curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    """ROUGE-L F1: precision/recall of the longest in-order word overlap."""
    p, r = words(prediction), words(reference)
    lcs = lcs_length(p, r)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style bag-of-words F1 (order-insensitive)."""
    p, r = words(prediction), words(reference)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)
