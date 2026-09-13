import logging
import logging.config

from app.config import get_logging_config

_configured = False


def setup_logger(name="rag_assistant"):
    """用 dictConfig 统一初始化日志，之后各模块 getLogger 即可"""
    global _configured
    if not _configured:
        logging.config.dictConfig(get_logging_config())
        _configured = True
    return logging.getLogger(name)