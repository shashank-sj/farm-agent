"""
Agricultural Q&A dataset: loading, cleaning, and a leakage-safe split.

Source: KisanVaani/agriculture-qa-english-only on the Hugging Face Hub
(22,615 rows, columns `question` / `answers`, Apache-2.0). It is noisy: answers
as short as 2 characters ("Cassava "), truncated answers, leading/trailing
whitespace, and many near-duplicate questions ("what is crop rotation" vs
"What is crop rotation?").

Used by both
  - finetune/   (train / val / test for supervised fine-tuning), and
  - scripts/eval_retrieval.py (questions as queries, answers as the corpus),
so both see exactly the same cleaning and the same split.

Leakage: a random row split would put "what is crop rotation" in train and
"What is crop rotation?" in test, so test scores would measure memorisation.
Instead, rows are grouped with union-find — two rows land in the same group if
their questions normalise to the same key OR their answers are identical — and
whole groups are assigned to a split. Assignment uses a hash of the group key,
not a shuffle, so it is deterministic and stable: adding new rows later never
moves an existing row to a different split.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from src.retrieval.bm25 import tokenize

DATASET_ID = "KisanVaani/agriculture-qa-english-only"

MIN_ANSWER_CHARS = 20
MIN_ANSWER_WORDS = 4
MAX_ANSWER_CHARS = 2000


@dataclass
class QAPair:
    question: str
    answer: str
    group: str = ""     # filled in by assign_groups()
    split: str = ""     # "train" / "val" / "test", filled in by group_split()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def question_key(question: str) -> str:
    """Order-insensitive bag of content words: 'What is crop rotation?' and
    'crop rotation — what is it' map to the same key. Over-grouping is safe
    (it only keeps more rows together); under-grouping would leak."""
    key = " ".join(sorted(set(tokenize(question))))
    # A question made only of stopwords would get an empty key and be merged
    # with every other such question — fall back to its lowercased text.
    return key or clean_text(question).lower()


def answer_key(answer: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()


def load_raw(dataset_id: str = DATASET_ID) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    return [{"question": r["question"], "answer": r["answers"]} for r in ds]


def clean_pairs(rows: list[dict]) -> tuple[list[QAPair], dict]:
    """Normalise whitespace, drop unusable answers, drop exact duplicates."""
    stats = Counter(raw=len(rows))
    seen = set()
    pairs = []
    for row in rows:
        q, a = clean_text(row.get("question", "")), clean_text(row.get("answer", ""))
        if not q or not a:
            stats["dropped_empty"] += 1
            continue
        if len(a) < MIN_ANSWER_CHARS or len(a.split()) < MIN_ANSWER_WORDS:
            stats["dropped_short_answer"] += 1      # "Cassava", "Crop Rotation"
            continue
        if len(a) > MAX_ANSWER_CHARS:
            stats["dropped_long_answer"] += 1
            continue
        if answer_key(a) == answer_key(q):
            stats["dropped_answer_equals_question"] += 1
            continue
        dedupe_key = (question_key(q), answer_key(a))
        if dedupe_key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(dedupe_key)
        pairs.append(QAPair(q, a))
    stats["kept"] = len(pairs)
    return pairs, dict(stats)


def assign_groups(pairs: list[QAPair]) -> int:
    """Union-find over shared question keys and shared answers. Sets each
    pair's .group to a stable key (the smallest question key in its group).
    Returns the number of groups."""
    parent = list(range(len(pairs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]   # path halving
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    first_with_key: dict[tuple[str, str], int] = {}
    for i, p in enumerate(pairs):
        for key in (("q", question_key(p.question)), ("a", answer_key(p.answer))):
            if key in first_with_key:
                union(i, first_with_key[key])
            else:
                first_with_key[key] = i

    group_key: dict[int, str] = {}
    for i, p in enumerate(pairs):
        root = find(i)
        qk = question_key(p.question)
        if root not in group_key or qk < group_key[root]:
            group_key[root] = qk
    for i, p in enumerate(pairs):
        p.group = group_key[find(i)]
    return len(group_key)


def _bucket(key: str) -> float:
    """Deterministic pseudo-random number in [0, 1) from a string."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def group_split(pairs: list[QAPair], val_frac: float = 0.05, test_frac: float = 0.05) -> dict[str, list[QAPair]]:
    if not all(p.group for p in pairs):
        assign_groups(pairs)
    splits: dict[str, list[QAPair]] = {"train": [], "val": [], "test": []}
    for p in pairs:
        u = _bucket(p.group)
        p.split = "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"
        splits[p.split].append(p)
    return splits


def leakage_report(splits: dict[str, list[QAPair]]) -> dict:
    """Counts of question keys / answers shared between splits. Should be all 0."""
    qkeys = {name: {question_key(p.question) for p in rows} for name, rows in splits.items()}
    akeys = {name: {answer_key(p.answer) for p in rows} for name, rows in splits.items()}
    report = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            report[f"{a}/{b}_shared_questions"] = len(qkeys[a] & qkeys[b])
            report[f"{a}/{b}_shared_answers"] = len(akeys[a] & akeys[b])
    return report
