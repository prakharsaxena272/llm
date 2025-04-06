import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

# ✅ Ensure data directory exists
os.makedirs("data", exist_ok=True)

# ✅ Load PDF document
documents = SimpleDirectoryReader(input_dir="data", required_exts=[".pdf"]).load_data()
print("✅ PDF Loaded Successfully!")

# ✅ Connect to ChromaDB (Running on Docker)
port_number = 8000
chroma_client = chromadb.HttpClient(host="localhost", port=port_number)

# ✅ Ensure Collection Exists
collection_name = "constitution_index_gemma"
collection = chroma_client.get_or_create_collection(collection_name)

# ✅ Initialize Vector Store
vector_store = ChromaVectorStore(chroma_collection=collection)

# ✅ Use Ollama with Gemma embedding model
embed_model = OllamaEmbedding(model_name="gemma")  # You can change this if needed

# ✅ No `persist_dir` yet – we’ll persist after indexing
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ✅ Build and persist index
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

# ✅ Now persist to storage directory
index.storage_context.persist(persist_dir="storage_gemma")

print("✅ Indexing with Gemma Completed & Saved to 'storage_gemma'!")
