import streamlit as st
from rag_utils import VectorDB, QueryEngine

with st.sidebar:
    "Versão Alpha 0.1"

st.title("LexAI 🏛️🔎")
st.caption("Inteligência Artificial para o Direito")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Como posso te ajudar?"}]

if "query_engine" not in st.session_state:
    index = VectorDB(index_name="ver-carf").initialize_index()
    st.session_state["query_engine"] = QueryEngine(index, similarity_top_k=5).initialize_query_engine()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = st.session_state.query_engine.invoke({"input": prompt})["answer"]
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)