@echo off

echo 🚀 Starting Ollama model (mistral)...
start "" /B cmd /C "ollama run mistral"

REM Wait 10 seconds to allow Ollama server to be ready
timeout /t 10 /nobreak >nul

echo 🧠 Running main.py to prepare vector DB...
python main.py

echo 🚀 Starting FastAPI server...
uvicorn app:app --host 127.0.0.1 --port 8000 --reload

pause
