import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "faiss_index"
SESSION_DIR = BASE_DIR / "sessions"


def load_config():
    load_dotenv(BASE_DIR / ".env")
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def get_api_key(env_name):
    key = os.getenv(env_name)
    if not key:
        raise ValueError(f"缺少环境变量 {env_name}，请检查 .env 文件")
    return key

def get_logging_config():
    """读取 config.yaml 的 logging 段，并把 filename 转成绝对路径"""
    config = load_config()
    logging_config = config.get("logging",{})
    for handler in logging_config.get("handlers",{}).values():
        if "filename" in handler:
            file_path=BASE_DIR / handler["filename"]
            file_path.parent.mkdir(parents=True, exist_ok=True)  # 关键：自动建 logs/ 目录
            handler["filename"] = str(file_path)
    return logging_config