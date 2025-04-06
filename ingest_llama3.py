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

# ✅ Ensure Collection Exists (Reusing the same collection)
collection_name = "constitution_index"
chroma_client.get_or_create_collection(collection_name)

# ✅ Initialize Vector Store
vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_collection(collection_name))

# ✅ Use LLaMA 3 for local embeddings
embed_model = OllamaEmbedding(model_name="llama3")

# ✅ New storage directory for LLaMA 3
storage_dir = "storage_llama3"
os.makedirs(storage_dir, exist_ok=True)

# ✅ Setup StorageContext
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ✅ Pass the local embedding model
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
index.storage_context.persist(persist_dir=storage_dir)

print(f"✅ Indexing Completed & Saved in '{storage_dir}'!")
