import asyncio
import os
from typing import Optional
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# === Configuration ===
VECTOR_DB_PATH = "vector_db"
EMBEDDING_MODEL = "mistral"
LLM_MODEL = "mistral"

# === FastAPI Setup ===
app = FastAPI(title="Peanut AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request Schema ===
class Query(BaseModel):
    question: str

class HealthResponse(BaseModel):
    status: str
    message: str

# === Persona Prompt Template ===
PERSONA_PROMPT = """
You are Peanut, Aquib Younis's AI assistant. Use only the info provided in your brain (no assumptions).

🧠 BEHAVIOR:
- If user asks about everything → summarize all stored info in categories
- If user asks something specific → answer ONLY that, concisely

📦 RULES:
- DO NOT guess or add made-up skills or tech
- DO NOT repeat generic sentences like "he is skilled in..."
- DO NOT mention skills or tools that are not in the vector DB
- DO NOT mention Aquib unless asked
- List data **only if user asks**

FORMAT IF FULL PROFILE IS ASKED:
Name, Roles, Skills (category-wise), Projects (name, role, link), Personality, Fun Fact

Question: {question}

Response:
"""


# === Globals ===
vector_db: Optional[Chroma] = None
qa_chain: Optional[RetrievalQA] = None
embeddings: Optional[OllamaEmbeddings] = None
llm: Optional[Ollama] = None

# === Cache Placeholder ===
@lru_cache(maxsize=128)
def get_cached_response(question: str) -> str:
    return ""  # Optional: Implement caching logic later

# === Load Vector DB ===
def load_vector_db():
    global embeddings
    return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

# === Initialize All Models ===
async def initialize_models():
    global vector_db, qa_chain, embeddings, llm

    print("🔧 Initializing Ollama embeddings & LLM...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    llm = Ollama(model=LLM_MODEL, temperature=0.7)

    print("📦 Loading vector DB...")
    vector_db = load_vector_db()

    print("🔗 Building RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=False,
        chain_type="stuff"
    )
    print("✅ Models initialized successfully.")

# === Startup Hook ===
@app.on_event("startup")
async def startup_event():
    await initialize_models()

# === Chat Endpoint ===
@app.post("/chat")
async def chat(query: Query):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="Model not initialized yet.")

    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        prompt = PERSONA_PROMPT.format(question=question)
        answer = await qa_chain.ainvoke(prompt)

        return {
            "result": answer,
            "status": "success"
        }

    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# === Dev Run ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
