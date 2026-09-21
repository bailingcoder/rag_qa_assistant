from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Any
from pydantic import BaseModel, Field
import json
import re
from openai import APITimeoutError, APIConnectionError, RateLimitError, AuthenticationError
import uuid
import time
from contextvars import ContextVar

from app.logger import setup_logger
from app.config import get_api_key, load_config, INDEX_DIR, MAX_CONTEXT_CHARS, MAX_PARENTS,MAX_TURNS, RRF_K
from app.embeddings import get_embeddings
from app.vector_store import ensure_vector_store
from app.prompts import (ROUTE_PROMPT, SYSTEM_PROMPT_WITH_CONTEXT, SYSTEM_PROMPT_NO_CONTEXT,
                         REWRITE_PROMPT,CLARIFY_PROMPT)
from app.bm25_store import get_bm25_retriever


logger = setup_logger()


class RAGError(Exception):
    """RAG 应用的自定义异常基类"""


class RAGState(BaseModel):
    """问答 Agent 的状态模型。Pydantic 会在运行时校验字段类型。"""

    question: str = Field(default="", description="当前用户问题")
    context: str = Field(default="", description="检索到的资料片段（拼接后）")
    need_retrieve: bool = Field(default=True, description="是否需要检索知识库")
    history: list[BaseMessage] = Field(default_factory=list, description="最近几轮历史，供查询重写使用")
    clarify_question: str = Field(default="", description="需要反问用户的问题")


_llm: ChatOpenAI | None = None
_graph: Any = None
_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


def get_llm() -> ChatOpenAI:
    """获取 LLM 实例（模块级缓存，避免重复创建连接）"""
    global _llm
    if _llm is None:
        config = load_config()
        llm_config = config["llm"]
        _llm = ChatOpenAI(
            model=llm_config["model"],
            api_key=get_api_key(llm_config["api_key_env"]),
            base_url=llm_config["base_url"],
            temperature=0.3,
        )
    return _llm


def _is_greeting(text: str) -> bool:
    """判断是否是寒暄：去掉标点/空格后，看是否命中常见寒暄词"""
    cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", text).lower()

    # 精确匹配：完整的寒暄词
    if cleaned in {"你好", "您好", "谢谢", "再见", "拜拜", "在吗", "在不在",
                   "hi", "hello", "thanks", "thankyou", "bye"}:
        return True

    # 前缀匹配：处理"你好呀""你好啊""谢谢啦"这类变体
    for g in ("你好", "您好", "谢谢", "再见", "拜拜"):
        if cleaned.startswith(g):
            return True
    return False


def _extract_json(text: str) -> str:
    """清洗 LLM 输出：去代码块标记，只留最外层 {} 内容"""
    text = text.strip()
    if text.startswith("```"):                 # 去掉 ```json ... ``` 包裹
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:            # 提取最外层 {}
        text = text[start:end + 1]
    return text


def _llm_json(message: str) -> dict:
    """直接解析 LLM 输出的 JSON；解析失败就让 LLM 重答一次"""
    llm = get_llm()
    content = llm.invoke(message).content.strip()
    try:
        return json.loads(_extract_json(content))
    except json.JSONDecodeError:
        content = llm.invoke(
            message + "\n\n注意：上次你没输出合法 JSON。这次请只输出一个 JSON 对象，不要任何其他文字。"
        ).content.strip()
        return json.loads(_extract_json(content))


def _format_history(history: list[BaseMessage], max_turns: int = MAX_TURNS) -> str:
    """把最近几轮消息转成可读文本，供重写 prompt 使用"""
    lines = []
    for m in history[-max_turns:]:
        role = "用户" if m.type == "human" else "助手"
        lines.append(f"{role}：{m.content}")
    return "\n".join(lines)


def route_node(state: RAGState) -> dict:
    """路由节点：让 LLM 判断是否需要检索"""
    question = state.question

    if _is_greeting(question) or len(re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", question)) <= 2:
        logger.info("路由决策: %s\n原因: %s", "直接回答", "命中寒暄/短文本规则，跳过知识库检索")
        return {"need_retrieve": False}

    try:
        data = _llm_json(ROUTE_PROMPT.format(question=question))
        need_retrieve = bool(data["need_retrieve"])
        reason = str(data["reason"])
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("路由解析失败，默认检索")
        reason = "路由解析失败，默认检索"
        need_retrieve = True

    logger.info("路由决策: %s\n原因: %s", "检索" if need_retrieve else "直接回答", reason)
    return {"need_retrieve": need_retrieve}

def rewrite_node(state: RAGState) -> dict:
    """查询重写节点：把依赖上文的问题改写成独立完整的问题，便于检索"""
    question = state.question
    history = state.history
    if not history:
        return {"question": question}
    try:
        history = _format_history(history)
        rewritten = get_llm().invoke(
            REWRITE_PROMPT.format(question=question, history=history)
        ).content.strip()
        logger.info("查询重写: %s\n重写后: %s", question, rewritten)
        return {"question": rewritten}
    except Exception as exc:
        logger.warning("查询重写失败，默认使用原问题")
        logger.debug(exc)
        return {"question": question}


def clarify_node(state: RAGState) -> dict:
    """澄清节点：判断重写后的问题信息是否足够，不足则生成反问"""
    question = state.question
    history = state.history
    try:
        data = _llm_json(CLARIFY_PROMPT.format(question=question, history=history))
        clarify_question = str(data.get("clarify_question", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("澄清判断失败，默认不反问")
        clarify_question = ""

    if clarify_question:
        logger.info("需要反问澄清: %s", clarify_question)
    return {"clarify_question": clarify_question}


def clarify_condition(state: RAGState) -> str:
    return "retrieve" if not state.clarify_question else "end"

def route_condition(state: RAGState) -> str:
    return "rewrite" if state.need_retrieve else "reset_context"



def _rrf_fusion(docs_a: list, docs_b: list) -> list:
    """RRF 倒数排名融合：两路排名累加分数，返回按分数降序的 doc 列表"""
    scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}

    for rank, doc in enumerate(docs_a, start=1):
        key = doc.page_content
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)

    for rank, doc in enumerate(docs_b, start=1):
        key = doc.page_content
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]




def retrieve_node(state: RAGState) -> dict:
    """检索节点：向量检索 + BM25 关键词检索，RRF 融合后父块去重"""
    config = load_config()
    question = state.question
    vector_store = ensure_vector_store(
        index_dir=str(INDEX_DIR),
        embeddings=get_embeddings(),
    )

    top_k = config["retriever"]["top_k"]
    search_k = top_k * 3          # 每路多召回，给 RRF 融合留余量

    # 向量检索
    docs_vec = vector_store.similarity_search(question, k=search_k)

    # BM25 关键词检索
    bm25 = get_bm25_retriever(k=search_k)
    docs_bm25 = bm25.invoke(question)

    # RRF 融合
    fused = _rrf_fusion(docs_vec, docs_bm25)

    # 父块去重拼接（沿用你之前的逻辑）
    parents: dict[str, str] = {}
    total = 0
    for doc in fused:
        pid = doc.metadata.get("parent_id", "")
        parent = doc.metadata.get("parent_text", "")
        if not parent or pid in parents:
            continue
        if len(parents) >= MAX_PARENTS:
            break
        if total + len(parent) > MAX_CONTEXT_CHARS:
            continue
        parents[pid] = parent
        total += len(parent)

    context = "\n\n".join(parents.values()) if parents else "\n\n".join(d.page_content for d in fused[:top_k])
    logger.info("向量 %d + BM25 %d，RRF 融合后 %d，去重 %d 父块",
                len(docs_vec), len(docs_bm25), len(fused), len(parents))
    return {"context": context}


def reset_context_node(state: RAGState) -> dict:
    """寒暄分支专用：清空 context，避免残留上一轮的资料"""
    return {"context": ""}



def _stream_answer(context: str, history: list[BaseMessage], question: str):
    """流式生成答案。history 是完整历史（不含当前问题）。"""
    if context:
        system = SYSTEM_PROMPT_WITH_CONTEXT.format(context=context)
    else:
        system = SYSTEM_PROMPT_NO_CONTEXT.format(context=context)

    messages = [system] + history + [HumanMessage(content=question)]
    for chunk in get_llm().stream(messages):
        if chunk.content:
            yield chunk.content


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reset_context", reset_context_node)
    graph.add_node("route", route_node)
    graph.add_node("clarify", clarify_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        route_condition,
        {"rewrite": "rewrite", "reset_context": "reset_context"},
    )
    graph.add_edge("rewrite", "clarify")
    graph.add_conditional_edges(
        "clarify",
        clarify_condition,
        {"retrieve": "retrieve", "end": END},
    )
    graph.add_edge("retrieve", END)
    graph.add_edge("reset_context", END)

    return graph.compile()


def ask(question: str, history: list[BaseMessage]) -> dict:
    """问答入口。history 是完整历史（不含当前问题）。
    返回 {"ok": True, "stream": generator} 或 {"ok": False, "error": str}。"""
    global _graph
    if _graph is None:
        _graph = build_graph()

    trace = uuid.uuid4().hex[:8]
    token = _trace_id.set(trace)
    start = time.perf_counter()
    logger.info("[%s] 开始问答，问题: %s", trace, question)

    try:
        result = _graph.invoke({"question": question, "history": history})
        clarify_question = result.get("clarify_question", "")
        if clarify_question:
            logger.info("[%s] 反问澄清: %s", trace, clarify_question)
            return {"ok": True, "clarify": clarify_question}

        context = result.get("context", "")
        logger.info("[%s] 检索完成，耗时 %.2f 秒", trace, time.perf_counter() - start)
        return {"ok": True, "stream": _stream_answer(context, history, question)}

    except AuthenticationError:
        logger.error("[%s] API Key 无效或已过期", trace)
        return {"ok": False, "error": "AI 服务认证失败，请检查 .env 中的 API Key 是否正确。"}
    except RateLimitError:
        logger.error("[%s] API 请求被限流", trace)
        return {"ok": False, "error": "请求过于频繁，请稍后再试。"}
    except (APITimeoutError, APIConnectionError):
        logger.error("[%s] AI 服务连接超时或失败", trace)
        return {"ok": False, "error": "AI 服务暂时连接不上，请稍后重试。"}
    except FileNotFoundError as exc:
        logger.error("[%s] 向量库不存在: %s", trace, exc)
        return {"ok": False, "error": "系统尚未构建知识库，请先运行向量库构建脚本。"}
    except RAGError as exc:
        logger.error("[%s] RAG 处理错误: %s", trace, exc)
        return {"ok": False, "error": "系统处理出错，请稍后再试。"}
    except Exception as exc:
        logger.exception("[%s] 问答过程发生未预期错误", trace)
        return {"ok": False, "error": "系统暂时无法回答，请稍后再试。"}
    finally:
        _trace_id.reset(token)
