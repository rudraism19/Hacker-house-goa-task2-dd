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

@app.get("/health")
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
def compare_chunking():
    """
    Returns comparative benchmarking analytics across all 4 chunking strategies.
    """
    return chunk_engine.compare_strategies(dataset[:20])

@app.get("/benchmark")
def execute_benchmark(num_samples: int = 50, strategy: str = "fixed_overlap"):
    """
    Runs benchmark suite and returns P50, P70, P100 latency percentiles.
    """
    return run_benchmark_suite(num_samples=num_samples, strategy=strategy)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
