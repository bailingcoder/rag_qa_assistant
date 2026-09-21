import pickle
from pathlib import Path
import jieba
from langchain_community.retrievers import BM25Retriever

from app.config import INDEX_DIR
from app.logger import setup_logger

logger = setup_logger()

_bm25 = None
_CHUNKS_FILE = INDEX_DIR / "chunks.pkl"


def _tokenize(text: str) -> list[str]:
    """中文分词：jieba 切词，去掉空白 token"""
    return [w for w in jieba.lcut(text) if w.strip()]


def save_chunks(chunks: list):
    """全量覆盖保存 chunks，并让 BM25 缓存失效"""
    global _bm25
    with open(_CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    _bm25 = None
    logger.info("已保存 %d 个 chunk 供 BM25 检索", len(chunks))


def append_chunks(new_chunks: list):
    """追加新 chunk（与磁盘已有合并后全量保存），并失效缓存"""
    existing = []
    if _CHUNKS_FILE.exists():
        with open(_CHUNKS_FILE, "rb") as f:
            existing = pickle.load(f)
    save_chunks(existing + new_chunks)


def get_bm25_retriever(k: int = 9) -> BM25Retriever:
    """获取缓存的 BM25 检索器；首次调用从磁盘加载 chunks 构建"""
    global _bm25
    if _bm25 is None:
        with open(_CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        _bm25 = BM25Retriever.from_documents(chunks, preprocess_func=_tokenize, k=k)
        logger.info("BM25 索引已构建，共 %d 个 chunk", len(chunks))
    return _bm25