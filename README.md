# Competitive Intelligence LangGraph Multi-Agent System

An end-to-end autonomous competitive intelligence system built with LangGraph, Flask, Streamlit, and Qdrant Cloud. The agent ingests, filters, retrieves, and synthesizes unstructured market data to produce executive intelligence dossiers.

---

## Architecture Overview

The system follows a multi-node cyclical state graph architecture with explicit safety guardrails, dynamic vector retrieval, and synthesis capabilities.

[User Directive]
       │
       ▼
┌──────────────┐
│ Input Guard  │ ── (Unsafe / Off-topic) ──► [Terminal Report]
└──────────────┘
       │ (Safe)
       ▼
┌──────────────┐
│ Agent Brain  │ ◄───┐
└──────────────┘     │
       │             │
   (Tool Call)       │ (Observation)
       ▼             │
┌──────────────┐     │
│ Execute Tool │ ────┘
└──────────────┘
       │ (Synthesis Complete)
       ▼
┌──────────────┐
│  Synthesize  │
└──────────────┘
       │
       ▼
┌──────────────┐
│ Output Guard │ ──► [Validated Executive Dossier]
└──────────────┘

### Graph Nodes
1. **input_guardrail**: Enforces strict policy validation against prompt injection, malicious instructions, and out-of-scope directives.
2. **agent_reasoning**: Formulates strategic queries, decides which tools to invoke, and processes retrieved facts.
3. **execute_tools**: Executes deterministic tool calls against the vector database or external feeds.
4. **synthesizer**: Aggregates intermediate observations into a structured executive dossier.
5. **output_guardrail**: Verifies content hygiene, ensuring no prompt leaks, toxicity, or safety policy violations exist in the final dossier.

---

## Tech Stack

- **Framework**: LangChain & LangGraph
- **Vector DB**: Qdrant Cloud (Cosine Similarity, `text-embedding-3-small`)
- **Backend Service**: Flask REST API (Port 5000)
- **Frontend UI**: Streamlit (Port 8501)
- **Containerization**: Docker & Docker Compose

---

## Directory Structure

Final Project Ido Cohen/
├── data/
│   └── competitor_profiles.json
├── docs/
│   └── langgraph_visualization.png
├── src/
│   ├── app.py
│   ├── graph.py
│   ├── rag_ingestion.py
│   └── tools.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
└── streamlit_app.py

---

## Installation and Local Setup

### 1. Clone & Environment Setup
git clone <repository_url>
cd "Final Project Ido Cohen"
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt

### 2. Environment Variables
Create a `.env` file in the root directory:
OPENAI_API_KEY=your_openai_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

### 3. Data Ingestion
Populate the Qdrant Cloud vector collection:
python src/rag_ingestion.py

---

## Running with Docker Compose (Recommended)

To build and run the entire multi-container stack:
docker compose up --build

Access the services:
- **Streamlit Web UI**: http://localhost:8501
- **REST API Health Check**: http://localhost:5000/health
- **Research Endpoint**: POST http://localhost:5000/api/research

---

## API Documentation

### Health Check
- **Endpoint**: GET /health
- **Response**:
{
  "service": "Competitive Intelligence Agent API",
  "status": "healthy"
}

### Intelligence Research
- **Endpoint**: POST /api/research
- **Payload**:
{
  "query": "What are AlphaCorp's market share, key strengths, and primary product offerings?"
}
- **Response**:
{
  "is_safe": true,
  "query": "What are AlphaCorp's market share, key strengths, and primary product offerings?",
  "report": "### Strategic Dossier: AlphaCorp..."
}