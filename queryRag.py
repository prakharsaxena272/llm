import os
import chromadb
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# ✅ Ensure ChromaDB is running
port_number = 8000
chroma_client = chromadb.HttpClient(host="localhost", port=port_number)

# ✅ Load Existing Collection
collection_name = "constitution_index"
try:
    collection = chroma_client.get_collection(collection_name)
except:
    raise Exception(f"❌ ERROR: Collection '{collection_name}' not found. Ensure `ingest.py` was run correctly!")

# ✅ Load Vector Store
vector_store = ChromaVectorStore(chroma_collection=collection)

# ✅ Use the same embedding model
embed_model = OllamaEmbedding(model_name="mistral")

# ✅ Load from Persisted Storage
storage_dir = "storage"  # The directory where JSON files are saved
if not os.path.exists(storage_dir):
    raise Exception(f"❌ ERROR: Storage directory '{storage_dir}' not found. Ensure indexing was completed.")

storage_context = StorageContext.from_defaults(persist_dir=storage_dir, vector_store=vector_store)

# ✅ Load the saved index
index = load_index_from_storage(storage_context, embed_model=embed_model)

# ✅ Query Engine using the same LLM
query_engine = index.as_query_engine(llm=Ollama(model="mistral"))

# ✅ Test Query
question = "What is Article 36 about?"
response = query_engine.query(question)

print("🔍 Question:", question)
print("📜 Answer:", response)
