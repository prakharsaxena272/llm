import os
import chromadb
import streamlit as st
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Streamlit UI
st.set_page_config(page_title="Indian Constitution Chatbot - Gemma", layout="centered")
st.title("🇮🇳 Indian Constitution Chatbot (Gemma Model)")
st.markdown("Ask any question about the Constitution of India, like **What is Article 370?**")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Connect to ChromaDB
port_number = 8000
chroma_client = chromadb.HttpClient(host="localhost", port=port_number)

# Load collection for Gemma
collection_name = "constitution_index_gemma"
try:
    collection = chroma_client.get_collection(collection_name)
except:
    st.error(f"❌ Collection '{collection_name}' not found. Please run `ingest_gemma.py` first.")
    st.stop()

# Load vector store and embedding model
vector_store = ChromaVectorStore(chroma_collection=collection)
embed_model = OllamaEmbedding(model_name="gemma")

# Load index from separate Gemma storage
storage_dir = "storage_gemma"
if not os.path.exists(storage_dir):
    st.error(f"❌ Storage directory '{storage_dir}' not found. Please run `ingest_gemma.py` first.")
    st.stop()

storage_context = StorageContext.from_defaults(persist_dir=storage_dir, vector_store=vector_store)
index = load_index_from_storage(storage_context, embed_model=embed_model)

# Set up query engine with Gemma model
query_engine = index.as_query_engine(llm=Ollama(model="gemma"))

# Display welcome message
with st.chat_message("assistant"):
    st.markdown("Hello! Ask me anything about the Constitution of India.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    pre_prompt = (
        "You are a helpful assistant knowledgeable about the Constitution of India. "
        "Answer using only content from the Constitution. Be accurate and clear. "
        "If a section is not available, say it honestly.\n\n"
        "User's Question:\n"
    )
    full_prompt = pre_prompt + prompt

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(full_prompt)
            st.markdown(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
