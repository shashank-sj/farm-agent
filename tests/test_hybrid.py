import pytest

from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.index import Chunk, KnowledgeIndex
from tests.conftest import KeywordReranker

TEXTS = [
    "Nitrogen deficiency in wheat shows as yellowing of older leaves; apply urea.",
    "Stem borer in paddy is managed with Trichogramma egg parasitoids and light traps.",
    "PMFBY crop insurance premium is 2% for Kharif crops.",
    "Drip irrigation saves 40 to 60 percent water compared with flood irrigation.",
    "Late blight of potato spreads fast in cool, humid weather.",
]


@pytest.fixture
def index(embedder):
    chunks = [Chunk(id=i, text=t, source="test") for i, t in enumerate(TEXTS)]
    return KnowledgeIndex.build(chunks, embedder)


def test_rrf_rewards_documents_ranked_by_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 4]], k=60)
    assert fused[0][0] == 2
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_every_mode_ranks_the_matching_passage_first(index, embedder):
    retriever = HybridRetriever(index, embedder, KeywordReranker("insurance"))
    for mode in ("dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"):
        hits = retriever.retrieve("PMFBY insurance premium", top_n=2, mode=mode)
        assert hits[0].chunk.id == 2, mode
        # BM25 only returns passages sharing a query term (here just one);
        # the dense list always has candidates, so fused modes fill top_n.
        assert len(hits) == (1 if mode == "bm25" else 2), mode


def test_dense_rerank_requires_a_reranker(index, embedder):
    with pytest.raises(ValueError):
        HybridRetriever(index, embedder).retrieve("urea", mode="dense_rerank")


def test_reranker_reorders_the_fused_pool(index, embedder):
    retriever = HybridRetriever(index, embedder, KeywordReranker("potato"))
    hits = retriever.retrieve("crop disease and water", top_n=1, mode="hybrid_rerank")
    assert "potato" in hits[0].chunk.text.lower()


def test_rerank_mode_without_reranker_falls_back_to_dense(index, embedder):
    retriever = HybridRetriever(index, embedder, reranker=None)
    hits = retriever.retrieve("urea wheat", top_n=1, mode="hybrid_rerank")
    assert hits[0].chunk.id == 0
    assert hits[0].sparse_rank is None   # dense results carry no BM25 rank


def test_unknown_mode_is_rejected(index, embedder):
    with pytest.raises(ValueError):
        HybridRetriever(index, embedder).retrieve("x", mode="magic")


def test_index_round_trip_and_model_mismatch(index, tmp_path):
    index.save(tmp_path)
    assert KnowledgeIndex.exists(tmp_path)
    loaded = KnowledgeIndex.load(tmp_path, expected_embedding_model="hash-test")
    assert [c.text for c in loaded.chunks] == TEXTS
    with pytest.raises(ValueError):
        KnowledgeIndex.load(tmp_path, expected_embedding_model="BAAI/bge-small-en-v1.5")
