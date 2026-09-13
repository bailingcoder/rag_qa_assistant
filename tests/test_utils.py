import pytest
from langchain_core.messages import HumanMessage, AIMessage

from app.rag_chain import (
    get_last_question, _is_greeting, _llm_json,
    _extract_json, route_condition, after_generate, should_retry, RAGState,
)
from app.config import get_logging_config


class TestGetLastQuestion:
    def test_空消息返回空字符串(self):
        assert get_last_question([]) == ""

    def test_只有用户消息(self):
        msgs = [HumanMessage(content="退货流程是什么？")]
        assert get_last_question(msgs) == "退货流程是什么？"

    def test_最后是AI消息时跳过(self):
        # 模拟：生成节点已经往 messages 里追加了 AIMessage
        msgs = [
            HumanMessage(content="退货流程是什么？"),
            AIMessage(content="退货需要 3 天"),
        ]
        # 应该拿到用户的问题，而不是 AI 的回答
        assert get_last_question(msgs) == "退货流程是什么？"


class TestIsGreeting:
    @pytest.mark.parametrize("text", [
        "你好", "你好！", "你好呀！！！", "谢谢", "再见", "在吗",
        "hi", "hello", "Hi!!",
    ])
    def test_寒暄命中(self, text):
        assert _is_greeting(text) is True

    @pytest.mark.parametrize("text", [
        "退货流程是什么", "怎么退款", "产品有哪些型号", "介绍一下公司",
    ])
    def test_非寒暄不命中(self, text):
        assert _is_greeting(text) is False


class TestExtractJson:
    def test_纯JSON(self):
        assert _extract_json('{"score": 8.5, "is_pass": true}') == '{"score": 8.5, "is_pass": true}'

    def test_代码块包裹(self):
        raw = '```json\n{"score": 8.5}\n```'
        assert _extract_json(raw) == '{"score": 8.5}'

    def test_前后有多余文字(self):
        raw = '好的，结果如下：{"score": 8.5, "reason": "很好"}以上是评审。'
        assert _extract_json(raw) == '{"score": 8.5, "reason": "很好"}'

    def test_空内容返回空(self):
        assert _extract_json("") == ""


class TestRouteCondition:
    def test_需要检索(self):
        state = RAGState(need_retrieve=True)
        assert route_condition(state) == "retrieve"

    def test_不需要检索(self):
        state = RAGState(need_retrieve=False)
        assert route_condition(state) == "chat"


class TestAfterGenerate:
    def test_有资料进评分(self):
        state = RAGState(context="退货需要3天")
        assert after_generate(state) == "judge"

    def test_无资料直接结束(self):
        state = RAGState(context="")
        assert after_generate(state) == "end"


class TestShouldRetry:
    def test_高分通过(self):
        state = RAGState(score=8.5, retry_count=0)
        assert should_retry(state) == "end"

    def test_低分未超限重试(self):
        state = RAGState(score=5.0, retry_count=0)
        assert should_retry(state) == "generate"

    def test_低分但已超限放弃(self):
        # max_retry=2，retry_count 已到 2，不能再重试
        state = RAGState(score=5.0, retry_count=2)
        assert should_retry(state) == "end"


class TestLoggingConfig:
    def test_返回字典(self):
        cfg = get_logging_config()
        assert isinstance(cfg, dict)

    def test_包含关键段(self):
        cfg = get_logging_config()
        assert "handlers" in cfg
        assert "formatters" in cfg

    def test_文件处理器已轮转(self):
        cfg = get_logging_config()
        file_handler = cfg["handlers"]["file"]
        assert file_handler["class"] == "logging.handlers.RotatingFileHandler"
        assert file_handler["maxBytes"] > 0
        assert file_handler["backupCount"] > 0

    def test_filename是绝对路径(self):
        from pathlib import Path
        cfg = get_logging_config()
        for h in cfg["handlers"].values():
            if "filename" in h:
                assert Path(h["filename"]).is_absolute(), f"应为绝对路径: {h['filename']}"