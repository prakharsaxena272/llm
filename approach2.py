import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Define constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "data", "constitution.pdf")
INDEX_DIR = os.path.join(SCRIPT_DIR, "index_storage")
PERSIST_DIR = "index_storage"
COLLECTION_NAME = "constitution_index"
OLLAMA_MODEL = "llama3"

# Step 1: Build index from PDF if not already stored
def build_index():
    # Load documents from data folder
    reader = SimpleDirectoryReader(input_dir=PDF_DIR, recursive=True)
    documents = reader.load_data()

    # Initialize Chroma client
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Define embedding and LLM
    embed_model = OllamaEmbedding(model_name=OLLAMA_MODEL)
    service_context = ServiceContext.from_defaults(
        llm=Ollama(model=OLLAMA_MODEL),
        embed_model=embed_model
    )

    # Setup storage and index
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR, vector_store=vector_store
    )
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=service_context
    )

    index.storage_context.persist()
    return index


# Step 2: Load index if already exists
def load_index():
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    embed_model = OllamaEmbedding(model_name=OLLAMA_MODEL)
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR, vector_store=vector_store
    )
    service_context = ServiceContext.from_defaults(
        llm=Ollama(model=OLLAMA_MODEL),
        embed_model=embed_model
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        service_context=service_context
    )

# Step 3: Streamlit UI
st.title("📜 Constitution of India Chatbot")

if "index" not in st.session_state:
    if not os.path.exists(PERSIST_DIR):
        st.info("Index not found. Building now...")
        st.session_state.index = build_index()
    else:
        st.session_state.index = load_index()

query = st.text_input("Ask a question:")
if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        response = st.session_state.index.as_query_engine().query(query)
        st.subheader("📖 Response:")
        st.write(response.response)
