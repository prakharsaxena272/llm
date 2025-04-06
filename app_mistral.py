import os
import chromadb
import streamlit as st
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# ✅ Streamlit Config & Title
st.set_page_config(page_title="📘 Constitution Chat - Mistral", layout="centered")
st.title("📘 Constitution Chatbot (Mistral)")
st.markdown("Chat with the Indian Constitution using the **Mistral** model.")

# ✅ Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Connect to ChromaDB
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# ✅ Load Existing Collection
collection_name = "constitution_index"
try:
    collection = chroma_client.get_collection(collection_name)
except:
    st.error(f"❌ Collection '{collection_name}' not found. Run `ingest.py` first.")
    st.stop()

# ✅ Vector Store & Embeddings
vector_store = ChromaVectorStore(chroma_collection=collection)
embed_model = OllamaEmbedding(model_name="mistral")
storage_dir = "storage"

if not os.path.exists(storage_dir):
    st.error("❌ Storage folder not found. Run `ingest.py` first.")
    st.stop()

# ✅ Load Index
storage_context = StorageContext.from_defaults(persist_dir=storage_dir, vector_store=vector_store)
index = load_index_from_storage(storage_context, embed_model=embed_model)
query_engine = index.as_query_engine(llm=Ollama(model="mistral"))

# ✅ Show Initial Assistant Message
with st.chat_message("assistant"):
    st.markdown("Hi! Ask me anything about the Constitution of India 🇮🇳")

# ✅ Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ Chat Input
if prompt := st.chat_input("Your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            pre_prompt = (
                "You are a legal assistant focused only on the Indian Constitution. "
                "Answer strictly based on the constitutional text. If unclear, say so.\n\n"
                "Question:\n"
            )
            response = query_engine.query(pre_prompt + prompt)
            st.markdown(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
