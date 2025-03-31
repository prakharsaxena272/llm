import os
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    QueryBundle,
    SimpleDirectoryReader)
from llama_index.core.retrievers import VectorStoreRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Ingest.py (for indexing)
def ingest_data(data_dir="data", persist_dir="./chroma_db", collection_name="constitution_index"):
    """Ingests data from a PDF in data_dir, and saves it to a chroma database."""
    os.makedirs(data_dir, exist_ok=True)

    documents = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".pdf"]).load_data()
    print("✅ PDF Loaded Successfully!")

    port_number = 8000
    print(f"DEBUG: Trying to connect to ChromaDB on port {port_number}")

    chroma_client = chromadb.HttpClient(host="localhost", port=port_number)

    chroma_client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_collection(collection_name))

    embed_model = OllamaEmbedding(model_name="mistral")

    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    index.storage_context.persist()

    print("✅ Indexing Completed & Saved!")

# queryRag.py (for querying)
def query_index(query_text, persist_dir="./chroma_db", collection_name="constitution_index", ollama_model="mistral"):
    """Queries an index stored in ChromaDB using Ollama."""
    try:
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

        index = load_index_from_storage(storage_context)

        retriever = VectorStoreRetriever(vector_store=index.vector_store, top_k=5)

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=Ollama(model=ollama_model),
            embedding_model=OllamaEmbedding(model_name=ollama_model),
            text_qa_template="Give a detailed answer to the question, using the provided context."
        )

        response = query_engine.query(query_text)

        nodes = retriever.retrieve(query_text)
        print("Retrieved Nodes:", nodes)

        return str(response)

    except Exception as e:
        return