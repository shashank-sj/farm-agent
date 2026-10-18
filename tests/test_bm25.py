from src.retrieval.bm25 import BM25, tokenize

DOCS = [
    "Apply urea as a top dressing for nitrogen deficiency in wheat.",
    "Drip irrigation saves water for tomato and sugarcane.",
    "PMFBY is the crop insurance scheme for farmers.",
    "Wheat rust is controlled with a propiconazole spray.",
]


def test_tokenize_lowercases_drops_stopwords_and_plurals():
    assert tokenize("What are the Fertilizers for WHEAT?") == ["fertilizer", "wheat"]


def test_exact_rare_term_ranks_first():
    assert BM25.from_texts(DOCS).top_k("PMFBY insurance", 2)[0][0] == 2


def test_idf_is_never_negative_even_for_very_common_terms():
    bm25 = BM25.from_texts(["wheat"] * 5 + ["rice"])
    assert all(v >= 0 for v in bm25.idf.values())


def test_query_with_no_known_terms_returns_nothing():
    assert BM25.from_texts(DOCS).top_k("zzzz qqqq", 3) == []


def test_length_normalisation_prefers_shorter_document():
    long_doc = "urea nitrogen " + "filler words here " * 20
    short_doc = "urea nitrogen"
    assert BM25.from_texts([long_doc, short_doc]).top_k("urea", 1)[0][0] == 1


def test_term_frequency_saturates():
    bm25 = BM25.from_texts(["urea " * 1, "urea " * 50, "rice"])
    s = bm25.scores("urea")
    # 50 mentions must score higher than 1, but far less than 50×
    assert s[0] < s[1] < 3 * s[0]
