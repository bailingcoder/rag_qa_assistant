from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from app import rag_chain
from app.rag_chain import retrieve_node, RAGState


def test_retrieve_按父去重且最多两块(monkeypatch):
    fake_docs = [
        Document(page_content="块1", metadata={"parent_id": "p1", "parent_text": "父块A" * 50}),
        Document(page_content="块2", metadata={"parent_id": "p1", "parent_text": "父块A" * 50}),  # 同父
        Document(page_content="块3", metadata={"parent_id": "p2", "parent_text": "父块B" * 50}),
        Document(page_content="块4", metadata={"parent_id": "p3", "parent_text": "父块C" * 50}),
    ]
    fake_vs = MagicMock()
    fake_vs.similarity_search.return_value = fake_docs
    monkeypatch.setattr(rag_chain, "ensure_vector_store", lambda **kw: fake_vs)

    result = retrieve_node(RAGState(messages=[HumanMessage(content="问题")]))

    assert "父块A" in result["context"]          # 同父的两个块只保留一次
    assert "父块B" in result["context"]
    assert "父块C" not in result["context"]      # 超过 2 块被截断