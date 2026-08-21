"""
FastAPI REST Server for Voice-Enabled RAG System
Exposes API endpoints for audio queries, benchmark suites, and chunking strategy analytics.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from dataset_loader import MSMARCOXIBackendLoader
from chunking_engine import MultiStrategyChunkingEngine
from vector_store import VectorStore
from stt_engine import SpeechToTextEngine
from model_harness import ModelHarnessOrchestrator, VoiceRAGRequest, VoiceRAGResponse
from latency_analytics import LatencyAnalyticsEngine
from benchmark_runner import run_benchmark_suite

app = FastAPI(
    title="Voice-Enabled RAG System API",
    description="Sub-200ms Latency Voice RAG API trained on AI4Bharat MSMARCO-XI dataset.",
    version="1.0.0"
)

# Global State Engine
loader = MSMARCOXIBackendLoader(lang="hi", max_samples=200)
dataset = loader.load_dataset()

chunk_engine = MultiStrategyChunkingEngine(strategy_name="fixed_overlap")
chunks = chunk_engine.chunk_documents(dataset)

vector_store = VectorStore()
vector_store.build_index(chunks)

stt_engine = SpeechToTextEngine(provider="sarvam")
orchestrator = ModelHarnessOrchestrator(
    stt_engine=stt_engine,
    vector_store=vector_store,
    chunking_engine=chunk_engine
)

from fastapi.responses import HTMLResponse, RedirectResponse

@app.get("/", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
@app.get("/api.py", response_class=HTMLResponse)
def index_landing_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HACKER गोवा HOUSE // Voice RAG API</title>
        <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {
                background: #062b19;
                color: #F8F9FA;
                font-family: 'Space Grotesk', sans-serif;
                margin: 0; padding: 40px 20px;
                display: flex; justify-content: center; align-items: center; min-height: 80vh;
            }
            .card {
                background: #0b1f1a;
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 20px;
                padding: 40px;
                max-width: 650px;
                box-shadow: 0 16px 45px rgba(0,0,0,0.5);
                text-align: center;
            }
            .logo {
                font-family: 'Cinzel', serif;
                font-size: 2rem;
                font-weight: 900;
                color: #FDB827;
                letter-spacing: 2px;
                margin-bottom: 10px;
            }
            .badge {
                background: #E53E3E;
                color: #FFF;
                font-size: 0.6em;
                padding: 2px 6px;
                border-radius: 4px;
                vertical-align: middle;
            }
            .tag {
                display: inline-block;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                color: #00FF88;
                background: rgba(0,255,136,0.1);
                border: 1px solid rgba(0,255,136,0.3);
                padding: 4px 12px;
                border-radius: 12px;
                margin-bottom: 20px;
            }
            .btn {
                display: inline-block;
                background: #FDB827;
                color: #000;
                font-weight: 700;
                text-decoration: none;
                padding: 12px 24px;
                border-radius: 8px;
                margin: 8px;
                font-size: 0.95rem;
                transition: all 0.2s;
            }
            .btn:hover { background: #FFAA00; transform: translateY(-2px); }
            .btn-outline {
                background: transparent;
                color: #FDB827;
                border: 1px solid #FDB827;
            }
            .btn-outline:hover { background: rgba(253,184,39,0.1); }
            pre {
                background: #061713;
                border: 1px solid rgba(255,255,255,0.06);
                padding: 15px;
                border-radius: 8px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.8rem;
                text-align: left;
                overflow-x: auto;
                color: #A7F3D0;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="logo">HACKER <span class="badge">गोवा</span> HOUSE</div>
            <div class="tag">● API ACTIVE // SUB-200ms VOICE RAG // TASK #2</div>
            <p style="color: #9CA3AF; line-height: 1.6;">
                Production Voice-Enabled Retrieval-Augmented Generation REST API for Hacker House Goa 2026.
            </p>
            <div>
                <a href="/docs" class="btn">🚀 Interactive Swagger Docs</a>
                <a href="/health" class="btn btn-outline">⚡ Health Check</a>
            </div>
            <br/>
            <pre>POST /query
Content-Type: application/json

{
  "query_text": "What is a corporation?",
  "language_code": "en-IN",
  "chunking_strategy": "semantic_boundary"
}</pre>
        </div>
    </body>
    </html>
    """

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "dataset_samples": len(dataset),
        "indexed_chunks": len(chunks),
        "stt_provider": stt_engine.provider
    }

class TextQueryRequest(BaseModel):
    query_text: str
    language_code: Optional[str] = "hi-IN"
    chunking_strategy: Optional[str] = "fixed_overlap"
    stt_provider: Optional[str] = "sarvam"

@app.post("/query", response_model=Dict[str, Any])
@app.post("/api/query", response_model=Dict[str, Any])
def query_rag(req: TextQueryRequest):
    """
    Executes Voice RAG query (or text query) through STT, Guardrails, Retrieval, Harness, and Grounding.
    """
    chunk_engine.set_strategy(req.chunking_strategy)
    rag_req = VoiceRAGRequest(
        prompt_text=req.query_text,
        language_code=req.language_code,
        chunking_strategy=req.chunking_strategy,
        stt_provider=req.stt_provider
    )
    res: VoiceRAGResponse = orchestrator.run_pipeline(rag_req)
    return res.model_dump()

@app.post("/query/audio")
@app.post("/api/query/audio")
async def query_rag_audio(
    file: UploadFile = File(...),
    language_code: str = Form("hi-IN"),
    chunking_strategy: str = Form("fixed_overlap"),
    stt_provider: str = Form("sarvam")
):
    """
    Processes audio upload file through Speech-To-Text and RAG pipeline.
    """
    audio_bytes = await file.read()
    stt_engine.provider = stt_provider
    chunk_engine.set_strategy(chunking_strategy)

    rag_req = VoiceRAGRequest(
        audio_bytes=audio_bytes,
        audio_filename=file.filename or "audio.wav",
        language_code=language_code,
        chunking_strategy=chunking_strategy,
        stt_provider=stt_provider
    )
    res: VoiceRAGResponse = orchestrator.run_pipeline(rag_req)
    return res.model_dump()

@app.get("/chunking/compare")
@app.get("/api/chunking/compare")
def compare_chunking():
    """
    Returns comparative benchmarking analytics across all 4 chunking strategies.
    """
    return chunk_engine.compare_strategies(dataset[:20])

@app.get("/benchmark")
@app.get("/api/benchmark")
def execute_benchmark(num_samples: int = 50, strategy: str = "fixed_overlap"):
    """
    Runs benchmark suite and returns P50, P70, P100 latency percentiles.
    """
    return run_benchmark_suite(num_samples=num_samples, strategy=strategy)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
