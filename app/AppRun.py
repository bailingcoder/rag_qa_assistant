import streamlit as st
import sys
from pathlib import Path
import json
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag_chain import ask
from app.logger import setup_logger
from app.config import SESSION_DIR



logger=setup_logger()

st.set_page_config(
    page_title="智能知识库问答助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)
st.title("📚 智能知识库问答助手")


def typewriter(text: str):
    """把完整答案逐块 yield，模拟流式打字机效果"""
    for i in range(0, len(text), 3):      # 每次 3 个字
        yield text[i:i + 3]
        time.sleep(0.02)                   # 控制打字速度，可调


def generate_session_name():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_session():
    if st.session_state.session_name:
        session_data={
            "messages":st.session_state.messages,
            "session_name":st.session_state.session_name
        }
        if not Path(SESSION_DIR).exists():
            Path(SESSION_DIR).mkdir()

        with open(SESSION_DIR / f"{st.session_state.session_name}.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=4)


def load_sessions():
    session_list = []
    if Path(SESSION_DIR).exists():
        for file in Path(SESSION_DIR).glob("*.json"):
            session_list.append(file.name[:-5])
    session_list.reverse()

    logger.info(f"加载会话列表：{session_list}")
    return session_list

def load_session(session_name):
    try:
        if Path(SESSION_DIR / f"{session_name}.json").exists():
            with open(SESSION_DIR / f"{session_name}.json", "r", encoding="utf-8") as f:
                session_data = json.load(f)
                st.session_state.messages = session_data["messages"]
                st.session_state.session_name = session_data["session_name"]
        logger.info(f"加载会话：{session_name}")
    except Exception as e:
        logger.error(f"加载会话出错：{e}")

def delete_session(session_name):
    try:
        if Path(SESSION_DIR / f"{session_name}.json").exists():
            Path(SESSION_DIR / f"{session_name}.json").unlink()
            logger.info(f"删除会话：{session_name}")
            if st.session_state.session_name == session_name:
                st.session_state.messages = []
                st.session_state.session_name = generate_session_name()
                logger.info(f"新建会话：{st.session_state.session_name}")

    except Exception as e:
        logger.error(f"删除会话出错：{e}")



#初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_name" not in st.session_state:
    st.session_state.session_name = generate_session_name()

#加载会话信息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

with st.sidebar:
    st.header("智能知识库问答助手")

    if st.button("新建会话",width="stretch",icon="✒️"):
        #保存会话信息
        save_session()
        logger.info(f"保存会话：{st.session_state.session_name}")

        #创建一个新会话
        if st.session_state.messages:
            st.session_state.session_name = generate_session_name()
            st.session_state.messages = []
            save_session()
            logger.info(f"新建会话：{st.session_state.session_name}")
            st.rerun()

    st.text("会话历史")
    #加载会话列表信息
    session_list = load_sessions()

    for session in session_list:
        col1, col2 = st.columns([4, 1])
        with col1:
            # 加载会话信息
            if st.button(session, width="stretch", icon="📄", key=f"load_{session}",
                         type="primary" if session == st.session_state.session_name else "secondary"):
                load_session(session)
                st.rerun()
        with col2:
            # 删除会话信息
            if st.button("", width="stretch", icon="❌", key=f"delete_{session}"):
                delete_session(session)
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