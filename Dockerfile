FROM python:3.10-slim

# --- Install dependencies ---
RUN apt-get update && apt-get install -y curl gnupg && apt-get clean

# --- Install Ollama ---
#RUN curl -fsSL https://ollama.com/install.sh | sh || true

# --- Set working directory ---
WORKDIR /app

# --- Copy files (including vector_db from Git LFS) ---
COPY . .

# --- Install Python requirements ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Optional: Rebuild vector DB only if it doesn't exist ---
RUN python main.py

# --- Expose port & run ---
EXPOSE 10000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
