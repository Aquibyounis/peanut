FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl gnupg
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
