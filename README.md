# mcp-local-agents (local/open-source)

## 1) Prereqs
- Docker + Docker Compose
- Or local python 3.11
- Ollama installed (if running outside docker)

## 2) Run with Docker
1) Copy env:
   cp .env.example .env

2) Pull model inside the Ollama container (first time):
   docker compose up -d ollama
   docker exec -it $(docker ps -qf name=ollama) ollama pull llama3.1:8b

3) Start API:
   docker compose up --build

API: http://localhost:8000  
Docs: http://localhost:8000/docs  
Streamlit: run locally (see below)

## 3) Run locally (no docker)
pip install -r requirements.txt
cp .env.example .env
python -m app.main

## 4) Streamlit UI
streamlit run app/ui/streamlit_app.py

## 5) Quick test
curl -X POST http://localhost:8000/chat \
-H "Content-Type: application/json" \
-d '{"message":"Summarize what tools you have and then call ping."}'
