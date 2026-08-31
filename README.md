# Multi-Modal Retrieval-Augmented Generation (RAG) Intelligence Platform

An enterprise-ready, production-grade Multi-Modal RAG Chatbot application engineered to seamlessly extract semantic context, parse complex tables/images, and perform conversational reasoning across diverse unstructured enterprise datasets.

## 🌟 Architectural Features
* **Asynchronous API Ingestion Layer:** Engineered an asynchronous FastAPI backend structure to handle continuous processing and streaming token generations securely.
* **Multi-Modal Document Intelligence:** Integrated parsing architectures designed to parse textual, tabular, and unstructured documentation data concurrently.
* **Persistent Session Isolation:** Custom logic built to manage state tracking, conversational historical schemas, and system state rules natively.
* **Streamlit Responsive UI:** Built a highly polished, responsive front-end dashboard facilitating file uploads, historical tracking, and low-latency interaction streams.

## 🛠️ Complete Technical Infrastructure
* **Backend Layer:** FastAPI, Python 3.10+
* **Frontend Dashboard:** Streamlit UI
* **Orchestration Framework:** LangChain Expression Language (LCEL)
* **Deployment/Containerization:** Docker / Multi-stage Dockerfile


## ⚙️ How to Run and Deploy Natively

### 1. Local Development Setup
To run the multi-modal system locally on your machine, clone your repository, navigate into the directory, and configure your keys:
```bash
# Clone the repository
git clone https://github.com
cd multi-modal-rag-intelligence-engine

# Create your localized environment configuration
cp .env.example .env
# Open .env and add your valid OPENAI_API_KEY
```

### 2. Launching via Docker Compose (Recommended)
This platform is completely containerized. You can spin up both the FastAPI backend and Streamlit user dashboard locally with a single terminal command:
```bash
docker-compose up --build
```
Once initialized, open your browser to access the environments:
* **Frontend User Interface:** `http://localhost:8501`
* **Backend API Docs (Swagger UI):** `http://localhost:8000/docs`

### 3. Cloud Deployment Blueprint (Render / Coolify)
This repository contains a pre-configured architecture blueprint (`render.yaml`) supporting **Infrastructure as Code (IaC)** pipelines:
1. Connect your GitHub account to your hosting provider dashboard (e.g., Render).
2. Create a new project via **Blueprint**, selecting this repository.
3. The platform will automatically read your `render.yaml` configuration and launch your decoupled backend and frontend web services simultaneously. 
4. Ensure you input your secure environment variables (`OPENAI_API_KEY`) within your cloud manager dashboard.

