from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.logger import setup_logger

logger=setup_logger()

def split_documents(documents,chunk_size=500,chunk_overlap=50):
    """把文档切成小块。chunk_size 是每块大小，overlap 是块之间重叠（防止切断语义）"""
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks=splitter.split_documents(documents)
    logger.info(f"切分完成，共{len(chunks)}块")
    return chunks