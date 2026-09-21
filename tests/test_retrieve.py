from unittest.mock import MagicMock
from langchain_core.documents import Document
from app import rag_chain
from app.rag_chain import retrieve_node, RAGState


def test_retrieve_按父去重且最多两块(monkeypatch):
    fake_docs = [
        Document(page_content="块1", metadata={"parent_id": "p1", "parent_text": "父块A" * 50}),
        Document(page_content="块2", metadata={"parent_id": "p1", "parent_text": "父块A" * 50}),
        Document(page_content="块3", metadata={"parent_id": "p2", "parent_text": "父块B" * 50}),
        Document(page_content="块4", metadata={"parent_id": "p3", "parent_text": "父块C" * 50}),
    ]
    fake_vs = MagicMock()
    fake_vs.similarity_search.return_value = fake_docs
    monkeypatch.setattr(rag_chain, "ensure_vector_store", lambda **kw: fake_vs)

    # 新增：mock BM25，返回空列表，只验证向量那一路 + 父块去重
    fake_bm25 = MagicMock()
    fake_bm25.invoke.return_value = []
    monkeypatch.setattr(rag_chain, "get_bm25_retriever", lambda k: fake_bm25)

    result = retrieve_node(RAGState(question="问题"))

    assert "父块A" in result["context"]
    assert "父块B" in result["context"]
    assert "父块C" not in result["context"]