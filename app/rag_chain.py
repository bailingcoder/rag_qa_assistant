from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage,SystemMessage,BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Any,Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
import json
import re
from openai import APITimeoutError, APIConnectionError, RateLimitError, AuthenticationError
import uuid                              # P1-8：生成 trace_id
import time                              # P1-8：计时
from contextvars import ContextVar       # P1-8：trace_id 跨节点传递


from app.logger import setup_logger
from app.config import get_api_key,load_config,INDEX_DIR,MAX_CONTEXT_CHARS,MAX_PARENTS
from app.embeddings import get_embeddings
from app.vector_store import ensure_vector_store
from app.prompts import SCORE_PROMPT, ROUTE_PROMPT, SYSTEM_PROMPT_WITH_CONTEXT, SYSTEM_PROMPT_NO_CONTEXT, RETRY_HINT


logger = setup_logger()


class RAGError(Exception):
    """RAG 应用的自定义异常基类"""


class RAGState(BaseModel):
    """问答 Agent 的状态模型。Pydantic 会在运行时校验字段类型。"""

    context: str = Field(default="", description="检索到的资料片段（拼接后）")
    answer: str = Field(default="", description="最终答案")
    messages: Annotated[list[BaseMessage], add_messages] = Field(
        default_factory=list, description="完整对话历史，用 add_messages 追加"
    )
    score: float = Field(default=0.0, description="答案质量评分，评分范围0-10分")
    reason: str =Field(default="", description="评分理由")
    retry_count: int = Field(default=0, description="已重试次数")
    need_retrieve: bool = Field(default=True, description="是否需要检索知识库")


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

def _usage_tokens(msg) -> tuple[int, int]:
    """从 AIMessage 提取 (输入token, 输出token)，取不到返回 (0, 0)"""
    um = getattr(msg, "usage_metadata", None)              # langchain 新版
    if um:
        return um.get("input_tokens", 0), um.get("output_tokens", 0)
    tu = (getattr(msg, "response_metadata", {}) or {}).get("token_usage", {})   # 旧版兜底
    return tu.get("prompt_tokens", 0), tu.get("completion_tokens", 0)


def get_last_question(messages: list) -> str:
    """获取最后一条用户提问（跳过 AIMessage/ToolMessage）"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def retrieve_node(state: RAGState) -> dict:
    """检索节点：从向量库找与最新问题最相关的资料"""
    config = load_config()
    question = get_last_question(state.messages)
    vector_store = ensure_vector_store(
        index_dir=str(INDEX_DIR),
        embeddings=get_embeddings(),
    )
    docs = vector_store.similarity_search(question, k=config["retriever"]["top_k"])

    parents:dict[str,str] = {}
    total = 0
    for doc in docs:
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

    context = "\n\n".join(parents.values()) if parents else "\n\n".join(d.page_content for d in docs)
    logger.info(
        "检索 %d 块，去重后 %d 个父块，最终 %d 块 / %d token",
        len(docs), len(parents), len(parents), total,
    )
    return {"context": context}


def reset_context_node(state: RAGState) -> dict:
    """寒暄分支专用：清空 context，避免残留上一轮的资料"""
    return {"context": ""}


def generate_node(state: RAGState) -> dict:
    """生成节点：根据是否有资料，选择不同的 system prompt"""
    if state.context:
        system = SYSTEM_PROMPT_WITH_CONTEXT.format(context=state.context)
    else:
        system = SYSTEM_PROMPT_NO_CONTEXT.format(context=state.context)

    messages=[system]+state.messages
    if state.retry_count > 0:
        hint = RETRY_HINT.format(score=state.score, reason=state.reason)
        messages = messages + [HumanMessage(content=hint)]

    result = get_llm().invoke(messages)
    in_tok, out_tok = _usage_tokens(result)
    logger.info("生成答案（第 %d 次），token: 输入 %d / 输出 %d", state.retry_count + 1, in_tok, out_tok)
    return {"answer": result.content, "messages": [result]}

def after_generate(state: RAGState) -> str:
    """寒暄（无资料）直接结束不评分；有资料才进评分循环"""
    return "judge" if state.context else "end"

def judge_answer_node(state: RAGState) -> dict:
    """评审节点：用 LLM 给生成的答案打分"""
    history="\n".join(
        f"{'用户' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in state.messages
    )
    question=get_last_question(state.messages)
    answer = state.answer
    context = state.context
    data = _llm_json(SCORE_PROMPT.format(history=history,question=question, context=context, answer=answer))
    score = float(data["score"])
    reason = str(data["reason"])

    logger.info("第 %d 次评分: %.1f 分，原因: %s", state.retry_count + 1, score,  reason)
    return {
        "score": score,
        "reason": reason,
        "retry_count": state.retry_count + 1,
    }



def route_node(state: RAGState) -> dict:
    """路由节点：让 LLM 判断是否需要检索"""
    question = get_last_question(state.messages)

    # 规则快速挡：极短 或 明显寒暄，直接不检索（省一次 LLM 调用）
    if _is_greeting(question) or len(re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", question)) <= 2:
        logger.info("路由决策: %s\n原因: %s", "直接回答", "命中寒暄/短文本规则，跳过知识库检索")
        return {"need_retrieve": False}

    try:
        data = _llm_json(ROUTE_PROMPT.format(question=question))
        need_retrieve=bool(data["need_retrieve"])
        reason=str(data["reason"])
    except (json.JSONDecodeError, KeyError, TypeError):
        # 解析失败：默认去检索（宁可多检索，不错过资料）
        logger.warning("路由解析失败，默认检索")
        reason = "路由解析失败，默认检索"
        need_retrieve = True

    logger.info("路由决策: %s\n原因: %s","检索" if need_retrieve else "直接回答", reason)
    return {"need_retrieve": need_retrieve}


def route_condition(state: RAGState) -> str:
    return "retrieve" if state.need_retrieve else "chat"

def should_retry(state: RAGState) -> str:
    """评分 >= 8 → 通过；不达标且未到上限 → 回 generate 重试；到上限 → 放弃重试"""
    config = load_config()
    if state.score >= config["retriever"]["score_threshold"]:
        return "end"
    if state.retry_count >= config["retriever"]["max_retry"]:
        logger.warning("已达最大重试次数(%d)，采用当前答案", config["retriever"]["max_retry"])
        return "end"
    return "generate"

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reset_context", reset_context_node)
    graph.add_node("generate", generate_node)
    graph.add_node("judge", judge_answer_node)
    graph.add_node("route", route_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        route_condition,
        {"retrieve": "retrieve", "chat": "reset_context"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("reset_context", "generate")
    graph.add_conditional_edges(
        "generate",
        after_generate,
        {"judge": "judge", "end": END},
    )
    graph.add_conditional_edges(
        "judge",
        should_retry,
        {"generate": "generate", "end": END},
    )

    return graph.compile(checkpointer=MemorySaver())


def ask(question: str, thread_id: str = "default") -> dict:
    """问答入口。统一处理异常，返回 {"ok": bool, "answer"/"error": str}。"""
    global _graph
    if _graph is None:
        _graph = build_graph()

    trace = uuid.uuid4().hex[:8]              # P1-8
    token = _trace_id.set(trace)              # P1-8
    start = time.perf_counter()               # P1-8
    logger.info("[%s] 开始问答，thread=%s，问题: %s", trace, thread_id, question)

    try:
        result = _graph.invoke({"messages": [HumanMessage(content=question)], "retry_count": 0},
            config={"configurable": {"thread_id": thread_id}},
        )
        logger.info("[%s] 问答完成，耗时 %.2f 秒", trace, time.perf_counter() - start)
        return {"ok": True, "answer": result["answer"]}          # P1-7

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


