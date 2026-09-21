from pathlib import Path
from langchain_community.vectorstores import FAISS
from app.embeddings import get_embeddings
from app.logger import setup_logger
from app.loader import load_documents, list_data_files
from app.splitter import split_documents_with_parent
from app.bm25_store import save_chunks, append_chunks
from app.config import load_config, INDEX_DIR, DATA_DIR

logger = setup_logger()

_vector_store = None
_MANIFEST = INDEX_DIR / "indexed_files.txt"


def _read_manifest() -> set[str]:
    """读取已索引清单。清单不存在时，用当前文件补写一份（视为已索引）。"""
    if _MANIFEST.exists():
        return set(_MANIFEST.read_text(encoding="utf-8").splitlines())

    # 清单缺失：说明是升级前建的库。把当前文件视为已索引，避免整库重加。
    logger.warning("未找到索引清单，将当前文件视为已索引。如有遗漏请删除 faiss_index 重建。")
    current = {f.name for f in list_data_files(str(DATA_DIR))}
    _write_manifest(current)
    return current

def _write_manifest(names: set[str]):
    _MANIFEST.write_text("\n".join(sorted(names)), encoding="utf-8")


def _new_files() -> list[Path]:
    """找出 data 目录里、但还没索引过的文件"""
    indexed = _read_manifest()
    current = list_data_files(str(DATA_DIR))
    return [f for f in current if f.name not in indexed]




def _add_files(vector_store, files: list[Path]):
    """把新文件向量化后追加进向量库，并更新清单"""
    config = load_config()
    sc = config["splitter"]

    documents = load_documents(str(DATA_DIR),files)
    chunks = split_documents_with_parent(documents, sc["chunk_size"], sc["chunk_overlap"])

    if chunks:
        vector_store.add_documents(chunks)  # 关键：追加，不重建
        vector_store.save_local(str(INDEX_DIR))
        append_chunks(chunks)  # 新增：同步 chunks 供 BM25


    # 更新清单
    indexed = _read_manifest()
    indexed.update(f.name for f in files)
    _write_manifest(indexed)
    logger.info("已增量添加 %d 个文件，共 %d 个片段", len(files), len(chunks))


def ensure_vector_store(index_dir=str(INDEX_DIR), embeddings=None):
    global _vector_store
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    if _vector_store is not None:
        # 缓存命中，但仍检查有无新文件（有则追加）
        new = _new_files()
        if new:
            logger.info("发现 %d 个新文件，增量追加", len(new))
            _add_files(_vector_store, new)
        return _vector_store

    if embeddings is None:
        embeddings = get_embeddings()

    # 磁盘已有向量库 → 加载后检查增量
    if Path(index_dir).exists() and (Path(index_dir) / "index.faiss").exists():
        logger.info("从磁盘加载向量库: %s", index_dir)
        _vector_store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        new = _new_files()
        if new:
            logger.info("发现 %d 个新文件，增量追加", len(new))
            _add_files(_vector_store, new)
        return _vector_store

    # 首次构建：处理全部文件
    logger.warning("向量库不存在，开始全量构建（首次较慢）...")
    config = load_config()
    sc = config["splitter"]
    all_files = list_data_files(str(DATA_DIR))
    documents = load_documents(str(DATA_DIR), all_files)
    chunks = split_documents_with_parent(documents, sc["chunk_size"], sc["chunk_overlap"])

    _vector_store = FAISS.from_documents(chunks, embeddings)
    _vector_store.save_local(index_dir)
    _write_manifest({f.name for f in all_files})
    save_chunks(chunks)
    logger.info("向量库已构建并保存: %s", index_dir)
    return _vector_store