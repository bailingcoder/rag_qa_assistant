import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag_chain import ask
from app.logger import setup_logger
import time

def typewriter(text: str):
    """把完整答案逐块 yield，模拟流式打字机效果"""
    for i in range(0, len(text), 3):      # 每次 3 个字
        yield text[i:i + 3]
        time.sleep(0.02)                   # 控制打字速度，可调

logger=setup_logger()

st.set_page_config(page_title="智能知识库问答助手", page_icon="📚")
st.title("📚 智能知识库问答助手")



#初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

#加载会话信息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

with st.sidebar:
    st.header("关于")
    st.write("基于你的知识库文档回答问题。\n\n首次提问会自动构建向量库（较慢），之后秒回。")
    if st.button("清空历史对话"):
        st.session_state.messages = []
        st.rerun()

question=st.chat_input("请输入您的问题")
if question:
    st.session_state.messages.append({"role":"user","content":question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            answer=ask(question,thread_id="thread_001")
        st.write_stream(typewriter(answer))
    st.session_state.messages.append({"role":"assistant","content":answer})