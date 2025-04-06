import os
import chromadb
import streamlit as st
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

st.set_page_config(page_title="Alternate Constitution Chatbot", layout="centered")
st.title("📘 Indian Constitution Chatbot (Model Test)")
st.markdown("Ask about the Constitution of India. This instance uses a **different model**.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Connect to ChromaDB
port_number = 8000
chroma_client = chromadb.HttpClient(host="localhost", port=port_number)

# Load new collection
collection_name = "constitution_index2"
try:
    collection = chroma_client.get_collection(collection_name)
except:
    st.error(f"❌ Collection '{collection_name}' not found. Run `ingest2.py` first.")
    st.stop()

vector_store = ChromaVectorStore(chroma_collection=collection)
embed_model = OllamaEmbedding(model_name="llama2")  # 🔁 Change model here for testing
storage_dir = "storage2"
if not os.path.exists(storage_dir):
    st.error("❌ Storage folder 'storage2' not found. Run `ingest2.py` first.")
    st.stop()

storage_context = StorageContext.from_defaults(persist_dir=storage_dir, vector_store=vector_store)
index = load_index_from_storage(storage_context, embed_model=embed_model)
query_engine = index.as_query_engine(llm=Ollama(model="llama2"))  # 🔁 Change model here too

with st.chat_message("assistant"):
    st.markdown("Hi there! I'm your assistant for testing different LLMs on the Constitution.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    pre_prompt = (
        "You are a legal assistant focused only on the Indian Constitution. "
        "Answer strictly based on the constitutional text. If the answer is unclear, say so.\n\n"
        "Question:\n"
    )
    full_prompt = pre_prompt + prompt

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(full_prompt)
            st.markdown(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
