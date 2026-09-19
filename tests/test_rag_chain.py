from app.rag_chain import _is_greeting, _extract_json, RAGState


def test_is_greeting():
    assert _is_greeting("你好") is True
    assert _is_greeting("你好呀") is True
    assert _is_greeting("米彩是谁") is False


def test_extract_json_去代码块包裹():
    raw = '```json\n{"score": 8.5, "reason": "好"}\n```'
    assert _extract_json(raw) == '{"score": 8.5, "reason": "好"}'
