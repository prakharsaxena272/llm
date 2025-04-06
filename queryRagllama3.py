import os
import chromadb
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
# from chromadb.errors import CollectionNotFoundError

# ✅ Ensure ChromaDB is running
port_number = 8000
chroma_client = chromadb.HttpClient(host="localhost", port=port_number)

# ✅ Load Existing Collection
collection_name = "constitution_index"
try:
    collection = chroma_client.get_collection(collection_name)
except :
    raise Exception(f"❌ ERROR: Collection '{collection_name}' not found. Ensure `ingest.py` was run correctly!")

# ✅ Load Vector Store
vector_store = ChromaVectorStore(chroma_collection=collection)

# ✅ Use the same embedding model (LLaMA 3 instead of Mistral)
embed_model = OllamaEmbedding(model_name="llama3")

# ✅ Load from Persisted Storage
storage_dir = "storage_llama3"  # Updated directory for LLaMA 3 indexing
if not os.path.isdir(storage_dir) or not os.listdir(storage_dir):
    raise Exception(f"❌ ERROR: Storage directory '{storage_dir}' is empty. Ensure indexing was completed.")

storage_context = StorageContext.from_defaults(persist_dir=storage_dir, vector_store=vector_store)

# ✅ Load the saved index
index = load_index_from_storage(storage_context, embed_model=embed_model)

# ✅ Query Engine using LLaMA 3 LLM
query_engine = index.as_query_engine(llm=Ollama(model="llama3"))

# ✅ Test Query
question = "what is article 370?"
response = query_engine.query(question)

print("🔍 Question:", question)
print("📜 Answer:", response.response)  # Extracts actual text response
