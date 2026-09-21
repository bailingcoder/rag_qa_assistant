from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.logger import setup_logger
import re
import hashlib


logger=setup_logger()

def split_documents(documents,chunk_size=500,chunk_overlap=50):
    """把文档切成小块。chunk_size 是每块大小，overlap 是块之间重叠（防止切断语义）"""
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks=splitter.split_documents(documents)
    return chunks


def split_sections(text: str) -> list[str]:
    """按 '数字. 标题' 切成段落（父单元）；没有结构时整篇算一段"""
    parts = re.split(r"(?m)(?=^\s*\d+\.\s)", text)
    return [p.strip() for p in parts if p.strip()] or [text.strip()]


def _doc_id(doc) -> str:
    """文档级唯一 ID：用 source（文件路径）哈希生成，同一文件所有块共享，用于去重"""
    source = doc.metadata.get("source", "")
    raw = source if source else doc.page_content
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def split_documents_with_parent(documents, chunk_size=500, chunk_overlap=50) -> list:
    """逐文件→逐文档→逐段落切块，每块挂上 doc_id、chunk_id 和父段落全文"""
    all_chunks = []
    for doc in documents:
        doc_id = _doc_id(doc)
        page = doc.metadata.get("page", -1)
        sections = split_sections(doc.page_content)
        for i, section in enumerate(sections):
            sec_doc = type(doc)(page_content=section, metadata=dict(doc.metadata))
            chunks = split_documents([sec_doc], chunk_size, chunk_overlap)
            for j, c in enumerate(chunks):
                c.metadata["doc_id"] = doc_id                  # 所属文档（去重用）
                c.metadata["parent_text"] = section            # 父块全文
                c.metadata["parent_id"] = f"{doc_id}:{page}:{i}"       # 按父去重
            all_chunks.extend(chunks)

    logger.info("已切分 %d 个文档为 %d 个块", len(documents), len(all_chunks))
    return all_chunks