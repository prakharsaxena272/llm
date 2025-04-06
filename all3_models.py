import os
import streamlit as st
import chromadb
import json
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Page config
st.set_page_config(page_title="Model Comparison - Constitution", layout="wide")

# Sticky header CSS + scroll + bottom scroll JS
st.markdown("""
    <style>
    .sticky-header {
        position: sticky;
        top: 0;
        background-color: white;
        padding: 1rem 0 0.5rem 0;
        z-index: 999;
        border-bottom: 1px solid #e0e0e0;
    }
    .scrollable-chat {
        height: 75vh;
        overflow-y: auto;
        padding-right: 1rem;
    }
    </style>
    <script>
    window.addEventListener("load", () => {
        var chatBox = window.parent.document.querySelectorAll('section.main > div')[0];
        if (chatBox) chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'smooth' });
    });
    </script>
""", unsafe_allow_html=True)

# Sticky header
st.markdown('<div class="sticky-header"><h1>🤖📘 Compare Constitution Chatbots (Mistral vs LLaMA2 vs Gemma)</h1></div>', unsafe_allow_html=True)

# User prompt
prompt = st.text_input("Ask a question about the Constitution of India")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Download chat
if st.session_state.chat_history:
    chat_json = json.dumps(st.session_state.chat_history, indent=2)
    st.download_button("📥 Download Chat History", chat_json, file_name="constitution_chat_history.json", mime="application/json")

# Load engines
@st.cache_resource
def load_query_engine(model_name, storage_dir, collection_name):
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = OllamaEmbedding(model_name=model_name)
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir, vector_store=vector_store)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    return index.as_query_engine(llm=Ollama(model=model_name))

try:
    mistral_engine = load_query_engine("mistral", "storage", "constitution_index")
    llama2_engine = load_query_engine("llama2", "storage2", "constitution_index2")
    gemma_engine = load_query_engine("gemma", "storage_gemma", "constitution_index_gemma")
except Exception as e:
    st.error(f"❌ Error loading models or storage: {e}")
    st.stop()

# Prompt submission
if prompt:
    pre_prompt = (
        "You are a legal assistant focused only on the Indian Constitution. "
        "Answer strictly based on the constitutional text. If the answer is unclear, say so.\n\n"
        "Question:\n"
    )
    full_prompt = pre_prompt + prompt

    with st.spinner("Models are thinking..."):
        mistral_response = mistral_engine.query(full_prompt)
        llama2_response = llama2_engine.query(full_prompt)
        gemma_response = gemma_engine.query(full_prompt)

    st.session_state.chat_history.append({
        "user": prompt,
        "mistral": str(mistral_response),
        "llama2": str(llama2_response),
        "gemma": str(gemma_response)
    })

# Show chat history
st.markdown('<div class="scrollable-chat">', unsafe_allow_html=True)

for chat in st.session_state.chat_history:
    st.markdown("### 🙋 You:")
    st.markdown(chat["user"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔮 Mistral**")
        st.markdown(chat["mistral"])
    with col2:
        st.markdown("**🦙 LLaMA2**")
        st.markdown(chat["llama2"])
    with col3:
        st.markdown("**🌸 Gemma**")
        st.markdown(chat["gemma"])

st.markdown('</div>', unsafe_allow_html=True)
