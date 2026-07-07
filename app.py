__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
...
import streamlit as st
import os
from document_processor import process_uploaded_file
from vector_store import VectorStoreManager
from retriever import RerankingRetriever
from qa_chain import answer_question
from evaluator import run_evaluation

st.set_page_config(page_title="Medical QA System", page_icon="⚕️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    [data-testid="stSidebar"] { background: rgba(15, 23, 42, 0.6) !important; backdrop-filter: blur(12px) !important; border-right: 1px solid rgba(255, 255, 255, 0.1); }
    .stButton>button { background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%); color: white !important; border: none; border-radius: 8px; font-weight: 600; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    [data-testid="stChatMessage"] { background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 1rem; margin-bottom: 1rem; border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(8px); }
    [data-testid="stChatInput"], .stAlert { border-radius: 12px; border: none; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "vs_manager" not in st.session_state: st.session_state.vs_manager = VectorStoreManager()
if "retriever" not in st.session_state:
    st.session_state.retriever = RerankingRetriever(base_retriever=st.session_state.vs_manager.get_retriever(k=10))
if "eval_history" not in st.session_state: st.session_state.eval_history = []

with st.sidebar:
    st.title("⚙️ Settings")
    provider = st.selectbox("LLM Provider", ["Gemini", "OpenAI"])
    api_key = st.text_input(f"{provider} API Key", type="password")
    uploaded_files = st.file_uploader("Upload Medical PDFs", type=["pdf"], accept_multiple_files=True)
    use_ocr = st.checkbox("Enable OCR", value=False)
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Extracting text and generating embeddings..."):
            all_chunks = []
            for file in uploaded_files: all_chunks.extend(process_uploaded_file(file, use_ocr=use_ocr))
            if all_chunks:
                st.session_state.vs_manager.add_documents(all_chunks)
                st.success(f"Added {len(all_chunks)} chunks to vector store.")

st.title("⚕️ Medical QA System with RAG")
tab1, tab2 = st.tabs(["💬 Chat & QA", "📊 Admin Evaluation"])

with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "confidence" in msg: st.caption(f"Confidence Score: {msg['confidence']:.2f}")

    if prompt := st.chat_input("Ask a medical question..."):
        if not api_key:
            st.error("Please enter your API Key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    top_docs = st.session_state.retriever.get_relevant_documents(prompt, top_n=3)
                    resp = answer_question(prompt, top_docs, provider, api_key)
                    st.markdown(resp.get("answer"))
                    st.caption(f"Confidence: {resp.get('confidence_score', 0.0):.2f}")
                    st.session_state.messages.append({"role": "assistant", "content": resp.get("answer"), "confidence": resp.get("confidence_score", 0.0)})

with tab2:
    st.header("Ragas Evaluation")
    test_question = st.text_input("Test Question")
    ground_truth = st.text_area("Ground Truth Answer")
    if st.button("Run Evaluation"):
        if not api_key: st.error("API Key required.")
        else:
            with st.spinner("Evaluating..."):
                top_docs = st.session_state.retriever.get_relevant_documents(test_question, top_n=3)
                ans = answer_question(test_question, top_docs, provider, api_key)
                eval_res = run_evaluation(test_question, ans.get("answer"), [d.page_content for d in top_docs], ground_truth, provider, api_key)
                st.json(eval_res)
