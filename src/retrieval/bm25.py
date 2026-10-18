"""
Okapi BM25, implemented from scratch (no rank_bm25 dependency).

    score(q, d) = Σ_t  IDF(t) · tf(t,d)·(k1 + 1) / (tf(t,d) + k1·(1 − b + b·|d| / avgdl))
    IDF(t)      = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

The "+1 inside the log" IDF (the Lucene variant) never goes negative, so a term
that appears in more than half the corpus still adds a small positive score
instead of penalising documents that contain it.

k1 controls term-frequency saturation (the 5th mention of "urea" adds much less
than the 1st); b controls length normalisation (b=1 fully normalises by
document length, b=0 ignores it).
"""

import heapq
import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small, deliberately conservative list: farming terms like "how much", "when"
# carry little signal, but words like "not" or "no" are kept on purpose.
STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i in is it its
    me my of on or our should that the their this to was we what when where which
    who why will with you your""".split()
)


def _stem(token: str) -> str:
    """Minimal plural stripping so 'fertilizers' matches 'fertilizer'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed corpus, with an inverted index so a query only
    touches documents that share at least one term with it."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.tf = [Counter(doc) for doc in corpus_tokens]

        df = Counter(term for doc in corpus_tokens for term in set(doc))
        self.idf = {
            term: math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }

        self.postings: dict[str, list[int]] = defaultdict(list)
        for doc_id, counts in enumerate(self.tf):
            for term in counts:
                self.postings[term].append(doc_id)

    @classmethod
    def from_texts(cls, texts: list[str], **kwargs) -> "BM25":
        return cls([tokenize(t) for t in texts], **kwargs)

    def scores(self, query: str) -> dict[int, float]:
        """BM25 score for every document sharing a term with the query."""
        out: dict[int, float] = defaultdict(float)
        avgdl = max(self.avgdl, 1e-9)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id in self.postings[term]:
                tf = self.tf[doc_id][term]
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_id] / avgdl)
                out[doc_id] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return out

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """(doc_id, score) pairs for the k best-scoring documents, best first."""
        return heapq.nlargest(k, self.scores(query).items(), key=lambda kv: kv[1])
