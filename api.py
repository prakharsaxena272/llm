from fastapi import FastAPI
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.llms.ollama import Ollama

app = FastAPI()

# Load stored index
storage_context = StorageContext.from_defaults(persist_dir="./chroma_db")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(llm=Ollama(model="mistral"))

@app.get("/query/")
async def query_rag(question: str):
    response = query_engine.query(question)
    return {"response": str(response)}

