import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

# Load PDF document
documents = SimpleDirectoryReader(input_dir="data", required_exts=[".pdf"]).load_data()
print("✅ PDF Loaded Successfully!")

# Connect to ChromaDB
port_number = 8000
chroma_client = chromadb.HttpClient(host="localhost", port=port_number)

# New collection for model testing
collection_name = "constitution_index2"
chroma_client.get_or_create_collection(collection_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_collection(collection_name))

# Replace with model you want to test (e.g., llama2, gemma, etc.)
embed_model = OllamaEmbedding(model_name="llama2")

# Persist to a new folder
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
index.storage_context.persist(persist_dir="storage2")

print("✅ Indexing Completed & Saved to 'storage2'!")
