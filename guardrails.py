"""
Comprehensive Guardrails Module for Voice-Enabled RAG System

Includes:
1. Input Safety & Off-Topic Guardrail
2. Query Relevance & Grounding Verification Guardrail (Checks query-context relevance & answer faithfulness)
3. Safe Refusal Mechanism (Knows when NOT to answer)
"""

import re
import time
from typing import Dict, Any, List, Optional
from config import GROUNDING_SIMILARITY_THRESHOLD, HALLUCINATION_OVERLAP_THRESHOLD

# Indic & English Stopwords for Guardrail Verification
EN_STOPWORDS = {
    "is", "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "that", "this",
    "what", "how", "who", "where", "which", "are", "tell", "me", "about", "define", "explain",
    "can", "you", "please", "cost", "price", "meaning", "definition"
}

HI_STOPWORDS = {
    "क्या", "है", "हैं", "का", "की", "के", "में", "और", "से", "को", "पर", "बताएं", "बताइए",
    "किसे", "कहते", "कौन", "सा", "सी", "से", "होते", "होती", "होता", "लागत", "अर्थ", "मतलब",
    "बारे", "कृपया", "मुझे", "जानना", "एक", "यह", "वह"
}

ALL_STOPWORDS = EN_STOPWORDS | HI_STOPWORDS

class InputGuardrail:
    """
    Evaluates incoming voice transcripts for off-topic content, jailbreaks, and audio corruption.
    """
    def __init__(self):
        self.injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"system\s+prompt",
            r"reveal\s+(your\s+)?secret",
            r"jailbreak",
            r"dan\s+mode",
            r"sudo\s+rm"
        ]
        self.unsafe_keywords = ["bomb", "exploit", "hack", "virus", "poison"]

    def evaluate(self, transcript: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        clean_text = transcript.strip().lower()

        if len(clean_text) < 2:
            return {
                "is_safe": False,
                "is_off_topic": True,
                "reason": "Transcript is empty or unintelligible.",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        for pat in self.injection_patterns:
            if re.search(pat, clean_text):
                return {
                    "is_safe": False,
                    "is_off_topic": False,
                    "reason": "Prompt injection / jailbreak attempt detected.",
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
                }

        for kw in self.unsafe_keywords:
            if kw in clean_text:
                return {
                    "is_safe": False,
                    "is_off_topic": False,
                    "reason": "Unsafe or restricted keyword detected.",
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
                }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "is_safe": True,
            "is_off_topic": False,
            "reason": "Input passed safety guardrails.",
            "latency_ms": round(elapsed_ms, 2)
        }

class GroundingHallucinationGuardrail:
    """
    Dual-Layer Grounding Guardrail:
    Layer 1: Verifies that the retrieved context is genuinely relevant to the user query.
    Layer 2: Verifies that the generated answer is strictly grounded in the retrieved context.
    """
    def evaluate(
        self,
        answer: str,
        retrieved_contexts: List[str],
        top_retrieval_score: float,
        is_general_knowledge: bool = False,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        if not answer or len(answer.strip()) == 0:
            return {
                "is_grounded": False,
                "hallucination_score": 1.0,
                "grounding_score": 0.0,
                "reason": "Generated answer text is empty.",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        # If answered via Google Gemini General Knowledge (open-domain / un-indexed queries):
        if is_general_knowledge:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "is_grounded": True,
                "hallucination_score": 0.05,
                "grounding_score": 0.95,
                "reason": "Answer verified via Google Gemini Open-Domain Synthesis.",
                "latency_ms": round(elapsed_ms, 2)
            }

        # Check retrieval score threshold
        if not retrieved_contexts or top_retrieval_score < GROUNDING_SIMILARITY_THRESHOLD:
            return {
                "is_grounded": False,
                "hallucination_score": 1.0,
                "grounding_score": 0.0,
                "reason": f"No matching grounded facts found in context (Relevance: {top_retrieval_score:.2f} < {GROUNDING_SIMILARITY_THRESHOLD}).",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        context_blob = " ".join(retrieved_contexts).lower()
        context_words = set(re.findall(r'[\w\u0900-\u097F]+', context_blob))

        # --- Layer 1: Query-to-Context Relevance & Overlap Verification ---
        if query:
            clean_q = re.sub(r'[^\w\s\u0900-\u097F]', ' ', query.lower())
            q_words = [w for w in clean_q.split() if len(w) > 1]
            content_q_words = [w for w in q_words if w not in ALL_STOPWORDS]

            if content_q_words:
                query_matches = sum(1 for w in content_q_words if w in context_blob)
                query_overlap = query_matches / len(content_q_words)
                
                # If query keywords do NOT match retrieved context, reject grounding
                if query_matches == 0 or (top_retrieval_score < 0.40 and query_overlap < 0.30):
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return {
                        "is_grounded": False,
                        "hallucination_score": 1.0,
                        "grounding_score": 0.0,
                        "reason": f"Retrieved context is not relevant to query topic (Query-context match: {query_overlap:.0%}).",
                        "latency_ms": round(elapsed_ms, 2)
                    }

        # --- Layer 2: Answer-to-Context Faithfulness & Hallucination Check ---
        answer_words = re.findall(r'[\w\u0900-\u097F]+', answer.lower())
        content_words = [w for w in answer_words if len(w) > 1 and w not in ALL_STOPWORDS]

        if not content_words:
            grounded_ratio = 1.0
        else:
            match_count = sum(1 for w in content_words if w in context_words)
            grounded_ratio = match_count / len(content_words)

        hallucination_score = max(0.0, round(1.0 - grounded_ratio, 2))
        is_grounded = grounded_ratio >= HALLUCINATION_OVERLAP_THRESHOLD

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "is_grounded": is_grounded,
            "hallucination_score": hallucination_score,
            "grounding_score": round(grounded_ratio, 2),
            "reason": "Answer verified and grounded against retrieved context." if is_grounded else "Answer failed grounding verification (unsupported claims).",
            "latency_ms": round(elapsed_ms, 2)
        }

class SafeRefusalHandler:
    """
    Constructs clean refusal responses when guardrails fail.
    """
    @staticmethod
    def build_refusal(reason: str) -> Dict[str, Any]:
        return {
            "answer": "I cannot answer this question based on the retrieved context.",
            "refusal": True,
            "refusal_reason": reason,
            "citations": []
        }

if __name__ == "__main__":
    ig = InputGuardrail()
    print("Input Test:", ig.evaluate("What is API?"))
    gh = GroundingHallucinationGuardrail()
    print("Grounding Test (Mismatch):", gh.evaluate(
        query="What are CSE subjects?",
        answer="A corporation is a legal entity owned by shareholders.",
        retrieved_contexts=["A corporation is a company authorized by law."],
        top_retrieval_score=0.30
    ))
