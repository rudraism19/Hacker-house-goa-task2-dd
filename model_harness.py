"""
Model Harness & Orchestration Engine for Voice RAG System

Provides:
1. Structured Input/Output Schemas (Pydantic)
2. Tool Calling Engine (Query Refinement, Metadata Filter, High-Precision Entity Synthesizer)
3. Strict Grounding Verification (Never outputs irrelevant passages)
4. Retries with Exponential Backoff and Fallback Error Recovery
"""

import time
import math
import re
import os
import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import config
TOP_K_RETRIEVAL = getattr(config, "TOP_K_RETRIEVAL", 3)
GEMINI_MODEL = getattr(config, "GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_URL = getattr(config, "GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models")
from guardrails import InputGuardrail, GroundingHallucinationGuardrail, SafeRefusalHandler

# --- Pydantic Data Schemas ---

class VoiceRAGRequest(BaseModel):
    audio_bytes: Optional[bytes] = Field(default=None, description="Raw audio bytes")
    audio_filename: str = Field(default="voice_query.wav", description="Audio filename")
    prompt_text: Optional[str] = Field(default=None, description="Direct text input or STT hint")
    language_code: str = Field(default="en-IN", description="Language code (e.g., en-IN, hi-IN)")
    chunking_strategy: str = Field(default="fixed_overlap", description="Chunking strategy to use")
    stt_provider: str = Field(default="sarvam", description="STT provider ('sarvam', 'elevenlabs', 'local')")
    synthesizer_mode: str = Field(default="auto", description="Synthesizer mode ('auto', 'gemini', 'local')")

class ToolCallLog(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    output: Dict[str, Any]
    status: str = "success"
    latency_ms: float

class VoiceRAGResponse(BaseModel):
    transcript: str
    answer: str
    citations: List[str]
    is_refused: bool
    refusal_reason: Optional[str] = None
    chunking_strategy_used: str
    stt_provider_used: str
    synthesizer: str = "Local Extractive Engine"
    tool_calls: List[ToolCallLog]
    grounding_score: float
    hallucination_risk: float
    stage_latencies_ms: Dict[str, float]
    total_latency_ms: float
    met_sla_200ms: bool

# --- Model Harness Tools ---

class HarnessTools:
    @staticmethod
    def refine_query_tool(raw_transcript: str) -> Dict[str, Any]:
        """
        Tool 1: Cleans transcript, strips conversational filler, and formats query for vector search.
        """
        t0 = time.perf_counter()
        clean = raw_transcript.strip()
        fillers = ["please tell me", "can you say", "what is a", "what is an", "what is", "tell me about", "define", "कृपया बताएं", "मुझे जानना है", "बताइए", "क्या है"]
        
        entity_query = clean.lower()
        for f in fillers:
            entity_query = entity_query.replace(f, "")
        entity_query = entity_query.strip("? .!") or clean
        
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "refined_query": clean,
            "entity_keyword": entity_query,
            "original_query": raw_transcript,
            "latency_ms": round(elapsed_ms, 2)
        }

    @staticmethod
    def metadata_filter_tool(query: str, lang_code: str) -> Dict[str, Any]:
        """
        Tool 2: Constructs metadata filters based on language and query semantics.
        """
        t0 = time.perf_counter()
        lang_short = lang_code.split("-")[0].lower()
        filter_dict = {"lang": lang_short}
        
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "filters": filter_dict,
            "latency_ms": round(elapsed_ms, 2)
        }

    @staticmethod
    def synthesize_answer_tool(query: str, retrieved_chunks: List[Dict[str, Any]], mode: str = "auto") -> Dict[str, Any]:
        """
        Tool 3: High-Precision Answer Synthesizer.
        Uses Google Gemini API (gemini-1.5-flash) for state-of-the-art grounded answer synthesis when GEMINI_API_KEY is available and mode != 'local'.
        Gracefully falls back to extractive keyword/sentence matcher when offline, in local mode, or without API key.
        """
        t0 = time.perf_counter()
        if not retrieved_chunks:
            return {
                "answer": "",
                "citations": [],
                "is_matched": False,
                "confidence": 0.0,
                "synthesizer": "None",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        top_chunk = retrieved_chunks[0]
        raw_text = top_chunk.get("parent_text", "") or top_chunk.get("text", "")
        doc_id = top_chunk.get("doc_id", "doc_0")
        score = top_chunk.get("score", 0.0)

        # --- OPTION A: Google Gemini API (Highest Accuracy & Grounding) ---
        default_gemini_key = getattr(config, "GEMINI_API_KEY", "")
        gemini_api_key = os.getenv("GEMINI_API_KEY", "") or default_gemini_key
        if gemini_api_key and mode != "local":
            try:
                import requests
                
                context_passages = []
                for idx, chunk in enumerate(retrieved_chunks[:3]):
                    c_text = chunk.get("parent_text", "") or chunk.get("text", "")
                    c_id = chunk.get("doc_id", f"doc_{idx+1}")
                    context_passages.append(f"[Document {idx+1} - ID: {c_id}]: {c_text}")
                
                context_str = "\n\n".join(context_passages)
                system_instruction = (
                    "You are a grounded Voice RAG synthesizer for Hacker House Goa 2026.\n"
                    "Answer the user query accurately, concisely (1-3 sentences), strictly based on the provided context below.\n"
                    "If the context does not contain enough information to answer the question, reply with: "
                    "'The provided context does not contain sufficient information to answer this question.'\n"
                    "Do NOT extrapolate or invent facts not present in the context."
                )
                prompt_content = f"{system_instruction}\n\nContext:\n{context_str}\n\nUser Question: {query}\n\nGrounded Answer:"
                
                api_endpoint = f"{GEMINI_API_URL}/{GEMINI_MODEL}:generateContent?key={gemini_api_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt_content}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 200
                    }
                }
                
                resp = requests.post(api_endpoint, json=payload, timeout=3.5)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            gemini_ans = parts[0]["text"].strip()
                            if "insufficient information" not in gemini_ans.lower() and len(gemini_ans) > 5:
                                citations = [f"Doc ID: {c.get('doc_id', f'doc_{i}')} | Relevance: {c.get('score', 0.0):.2f}" for i, c in enumerate(retrieved_chunks[:2])]
                                return {
                                    "answer": gemini_ans,
                                    "citations": citations,
                                    "is_matched": True,
                                    "confidence": round(score, 2),
                                    "synthesizer": f"Google Gemini ({GEMINI_MODEL})",
                                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
                                }
            except Exception:
                # If Gemini API times out or fails, fallback seamlessly to fast extractive engine
                pass

        # --- OPTION B: Fast Local Extractive Synthesizer (Zero-Latency Offline Fallback) ---
        clean_text = re.sub(r'\[.*?\]', '', raw_text).strip()
        sentences = [s.strip() for s in re.split(r'(?<=[.!?।\n])\s+', clean_text) if len(s.strip()) > 5]
        if not sentences:
            sentences = [clean_text]

        q_words = set(re.findall(r'\w+', query.lower()))
        stopwords = {"is", "the", "a", "an", "and", "or", "in", "of", "to", "what", "how", "who", "where", "क्या", "है", "का", "की", "के", "में", "और", "से", "बताएं"}
        content_q_words = [w for w in q_words if w not in stopwords]

        best_sentences = []
        if content_q_words:
            scored_sents = []
            for s in sentences:
                s_lower = s.lower()
                matches = sum(1 for w in content_q_words if w in s_lower)
                if matches > 0:
                    scored_sents.append((matches, s))
            
            if scored_sents:
                scored_sents.sort(key=lambda x: x[0], reverse=True)
                best_sentences = [s for count, s in scored_sents[:2]]

        if not best_sentences:
            return {
                "answer": "",
                "citations": [],
                "is_matched": False,
                "confidence": 0.0,
                "synthesizer": "Local Extractive (No Match)",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        answer_text = " ".join(best_sentences).strip()
        citations = [f"Doc ID: {doc_id} | Relevance Score: {score:.2f}"]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "answer": answer_text,
            "citations": citations,
            "is_matched": True,
            "confidence": round(score, 2),
            "synthesizer": "Local Extractive Engine",
            "latency_ms": round(elapsed_ms, 2)
        }

# --- Model Harness Orchestrator ---

class ModelHarnessOrchestrator:
    def __init__(self, stt_engine, vector_store, chunking_engine):
        self.stt_engine = stt_engine
        self.vector_store = vector_store
        self.chunking_engine = chunking_engine
        self.input_guardrail = InputGuardrail()
        self.grounding_guardrail = GroundingHallucinationGuardrail()

    def execute_with_retry(self, func, max_retries: int = 2, backoff_factor: float = 0.05):
        attempt = 0
        while attempt <= max_retries:
            try:
                return func()
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise e
                sleep_sec = backoff_factor * math.pow(2, attempt)
                time.sleep(sleep_sec)

    def run_pipeline(self, request: VoiceRAGRequest) -> VoiceRAGResponse:
        t_pipeline_start = time.perf_counter()
        stage_latencies = {}
        tool_calls = []

        # -------------------------------------------------------------
        # Stage 1: Speech-to-Text (STT)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        stt_res = self.stt_engine.transcribe(
            audio_data=request.audio_bytes or b"",
            filename=request.audio_filename,
            language_code=request.language_code,
            prompt_hint=request.prompt_text or ""
        )
        stage_latencies["stt_ms"] = stt_res.get("latency_ms", round((time.perf_counter() - t0) * 1000.0, 2))
        transcript = stt_res.get("transcript", "")

        # -------------------------------------------------------------
        # Stage 2: Input Guardrails & Safety
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        input_eval = self.input_guardrail.evaluate(transcript)
        stage_latencies["input_guardrail_ms"] = input_eval.get("latency_ms", round((time.perf_counter() - t0) * 1000.0, 2))

        if not input_eval["is_safe"]:
            total_lat = (time.perf_counter() - t_pipeline_start) * 1000.0
            refusal = SafeRefusalHandler.build_refusal(input_eval["reason"])
            return VoiceRAGResponse(
                transcript=transcript,
                answer=refusal["answer"],
                citations=[],
                is_refused=True,
                refusal_reason=input_eval["reason"],
                chunking_strategy_used=request.chunking_strategy,
                stt_provider_used=stt_res.get("provider", request.stt_provider),
                tool_calls=[],
                grounding_score=0.0,
                hallucination_risk=1.0,
                stage_latencies_ms=stage_latencies,
                total_latency_ms=round(total_lat, 2),
                met_sla_200ms=total_lat < 200.0
            )

        # -------------------------------------------------------------
        # Stage 3: Tool Execution - Query Refinement & Filtering
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        refine_out = HarnessTools.refine_query_tool(transcript)
        tool_calls.append(ToolCallLog(
            tool_name="refine_query_tool",
            arguments={"raw_transcript": transcript},
            output=refine_out,
            latency_ms=refine_out["latency_ms"]
        ))
        refined_query = refine_out["refined_query"]

        meta_out = HarnessTools.metadata_filter_tool(refined_query, request.language_code)
        tool_calls.append(ToolCallLog(
            tool_name="metadata_filter_tool",
            arguments={"query": refined_query, "lang_code": request.language_code},
            output=meta_out,
            latency_ms=meta_out["latency_ms"]
        ))
        stage_latencies["harness_tools_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # -------------------------------------------------------------
        # Stage 4: Vector Retrieval
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        retrieval_res = self.execute_with_retry(
            lambda: self.vector_store.search(refined_query, top_k=TOP_K_RETRIEVAL)
        )
        stage_latencies["retrieval_ms"] = retrieval_res.get("latency_ms", round((time.perf_counter() - t0) * 1000.0, 2))
        retrieved_chunks = retrieval_res.get("results", [])
        top_score = retrieval_res.get("top_score", 0.0)

        # -------------------------------------------------------------
        # Stage 5: Tool Execution - High-Precision Answer Synthesis
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        synth_out = HarnessTools.synthesize_answer_tool(refined_query, retrieved_chunks, mode=request.synthesizer_mode)
        tool_calls.append(ToolCallLog(
            tool_name="synthesize_answer_tool",
            arguments={"query": refined_query, "num_chunks": len(retrieved_chunks)},
            output=synth_out,
            latency_ms=synth_out["latency_ms"]
        ))
        stage_latencies["synthesis_ms"] = synth_out["latency_ms"]

        raw_answer = synth_out.get("answer", "")
        citations = synth_out.get("citations", [])
        is_matched = synth_out.get("is_matched", False)

        # -------------------------------------------------------------
        # Stage 6: Grounding & Hallucination Guardrail Check
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        context_texts = [c.get("text", "") for c in retrieved_chunks]
        grounding_eval = self.grounding_guardrail.evaluate(raw_answer, context_texts, top_score)
        stage_latencies["grounding_guardrail_ms"] = grounding_eval.get("latency_ms", round((time.perf_counter() - t0) * 1000.0, 2))

        is_refused = False
        refusal_reason = None
        final_answer = raw_answer

        # If synthesis found no matching sentences in the retrieved passages OR grounding failed:
        if not is_matched or not grounding_eval["is_grounded"] or not raw_answer:
            is_refused = True
            refusal_reason = "No grounded facts found in the retrieved dataset for this question."
            refusal = SafeRefusalHandler.build_refusal(refusal_reason)
            final_answer = refusal["answer"]
            citations = []

        total_latency = (time.perf_counter() - t_pipeline_start) * 1000.0

        return VoiceRAGResponse(
            transcript=transcript,
            answer=final_answer,
            citations=citations,
            is_refused=is_refused,
            refusal_reason=refusal_reason,
            chunking_strategy_used=request.chunking_strategy,
            stt_provider_used=stt_res.get("provider", request.stt_provider),
            synthesizer=synth_out.get("synthesizer", "Local Extractive Engine"),
            tool_calls=tool_calls,
            grounding_score=grounding_eval["grounding_score"] if is_matched else 0.0,
            hallucination_risk=grounding_eval["hallucination_score"] if is_matched else 1.0,
            stage_latencies_ms=stage_latencies,
            total_latency_ms=round(total_latency, 2),
            met_sla_200ms=total_latency < 200.0
        )
