FROM python:3.10-slim

# Optional: system packages needed for chromadb
RUN apt-get update && apt-get install -y build-essential curl

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose Railway's dynamic port
EXPOSE $PORT

# Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
