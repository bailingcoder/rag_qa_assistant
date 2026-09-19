from langchain_core.documents import Document
from app.splitter import split_sections, split_documents_with_parent


def test_split_sections_按数字标题切分():
    text = "1. 介绍\n这是介绍内容\n2. 方法\n这是方法内容"
    sections = split_sections(text)
    assert len(sections) == 2
    assert sections[0].startswith("1. 介绍")


def test_split_sections_无结构整篇算一段():
    text = "没有任何数字标题的纯文本"
    assert len(split_sections(text)) == 1


def test_split_with_parent_挂上三个元数据():
    docs = [Document(page_content="1. 标题\n" + "很长的内容" * 200, metadata={"source": "test.txt"})]
    chunks = split_documents_with_parent(docs, chunk_size=50, chunk_overlap=10)
    assert chunks, "应该切出至少一个块"
    for c in chunks:
        assert "doc_id" in c.metadata
        assert "parent_id" in c.metadata
        assert "parent_text" in c.metadata