from app.rag_chain import _is_greeting, _extract_json, should_retry, RAGState


def test_is_greeting():
    assert _is_greeting("你好") is True
    assert _is_greeting("你好呀") is True
    assert _is_greeting("米彩是谁") is False


def test_extract_json_去代码块包裹():
    raw = '```json\n{"score": 8.5, "reason": "好"}\n```'
    assert _extract_json(raw) == '{"score": 8.5, "reason": "好"}'


def test_should_retry_达标结束():
    assert should_retry(RAGState(score=8.5, retry_count=0)) == "end"


def test_should_retry_未达标未到上限重试():
    assert should_retry(RAGState(score=5.0, retry_count=0)) == "generate"


def test_should_retry_到上限放弃():
    assert should_retry(RAGState(score=5.0, retry_count=2)) == "end"