from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from app.logger import setup_logger
from app.config import DATA_DIR

logger=setup_logger()


def list_data_files(data_dir: str = str(DATA_DIR)) -> list[Path]:
    """列出 data 目录下所有可处理的文档文件"""
    data_path = Path(data_dir)
    files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.txt")) + list(data_path.glob("*.md"))
    return files

def load_documents(data_dir: str = str(DATA_DIR), files: list[Path] | None = None):
    """加载文档。files 为 None 时加载全部；否则只加载指定文件。"""
    data_path = Path(data_dir)
    documents = []

    targets = files if files is not None else list_data_files(data_dir)

    for f in targets:
        f = Path(f)
        if f.suffix.lower() == ".pdf":
            documents.extend(PyPDFLoader(str(f)).load())
        elif f.suffix.lower() in (".txt", ".md"):
            documents.extend(TextLoader(str(f), encoding="utf-8").load())
        logger.info(f"已加载: {f.name}")

    logger.info(f"共加载 {len(documents)} 个文档")
    return documents
