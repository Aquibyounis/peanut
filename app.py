import asyncio
import os
from typing import Optional
from functools import lru_cache


import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings



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
- Be professional and talk like you are a messenger but not wider

📦 RULES:
- DO NOT guess or add made-up skills or tech
- DO NOT repeat generic sentences like "he is skilled in..."
- List data **only if user asks**
- Answer in depth for skills or personality questions

FORMAT IF FULL PROFILE IS ASKED:
Name, Roles, Skills (category-wise), Projects (name, role, link), Personality, Fun Fact

Question: {question}

Response:
"""


# === Globals ===
vector_db: Optional[Chroma] = None
qa_chain: Optional[RetrievalQA] = None
embeddings: Optional[HuggingFaceEmbeddings] = None
llm: Optional[ChatGroq] = None


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

    print("🔧 Initializing LLM with Groq API...")
    load_dotenv()
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="gemma2-9b-it")

    print("📥 Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
        if isinstance(answer, dict) and "result" in answer:
            result = answer["result"]
        else:
            result = str(answer)

        return {
            "result": result,
            "status": "success"
        }


    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# === Dev Run ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
