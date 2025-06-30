# main.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

print("🚀 Rebuilding Peanut's brain with RecursiveCharacterTextSplitter...")

# === Settings ===
DATA_FILE = "peanut_data.txt"
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 100
EMBED_MODEL = "mistral"
PERSIST_DIR = "vector_db"

# === Load and Chunk Text ===
with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)

docs = splitter.split_documents([Document(page_content=text)])

# === Generate Embeddings ===
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# === Create Vector DB and Save ===
db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=PERSIST_DIR)
db.persist()

print("✅ Peanut's smarter brain is saved to:", PERSIST_DIR)
