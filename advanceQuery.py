import os
import chromadb
import streamlit as st
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# ✅ Streamlit UI setup
st.set_page_config(page_title="Indian Constitution Chatbot", layout="centered")
st.title("🇮🇳 Indian Constitution RAG Chatbot")
st.markdown("Ask any question about the Constitution of India, like **What is Article 370?**")

# ✅ Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Ensure ChromaDB is running
port_number = 8000
chroma_client = chromadb.HttpClient(host="localhost", port=port_number)

# ✅ Load Existing Collection
collection_name = "constitution_index"
try:
    collection = chroma_client.get_collection(collection_name)
except:
    st.error(f"❌ ERROR: Collection '{collection_name}' not found. Please run `ingest.py` first.")
    st.stop()

# ✅ Load Vector Store & Embeddings
vector_store = ChromaVectorStore(chroma_collection=collection)
embed_model = OllamaEmbedding(model_name="mistral")

# ✅ Load the index from local storage
storage_dir = "storage"
if not os.path.exists(storage_dir):
    st.error(f"❌ ERROR: Storage directory '{storage_dir}' not found. Ensure indexing was completed.")
    st.stop()

storage_context = StorageContext.from_defaults(persist_dir=storage_dir, vector_store=vector_store)
index = load_index_from_storage(storage_context, embed_model=embed_model)
query_engine = index.as_query_engine(llm=Ollama(model="mistral"))

# ✅ Chat interface
with st.chat_message("assistant"):
    st.markdown("Hello! Ask me anything about the Constitution of India.")

# ✅ Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ Input box
if prompt := st.chat_input("Your question..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Compose prompt with pre-instruction
    pre_prompt = (
        "You are a helpful assistant knowledgeable about the Constitution of India. "
        "Respond accurately using constitutional content. If the section is not present, say so honestly.\n\n"
        "User's Question:\n"
    )
    full_prompt = pre_prompt + prompt

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(full_prompt)
            st.markdown(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
