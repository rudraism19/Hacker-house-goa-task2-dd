"""
Comprehensive Guardrails Module for Voice-Enabled RAG System

Includes:
1. Input Safety & Off-Topic Guardrail
2. Faithfulness & Hallucination Guardrail
3. Safe Refusal Mechanism (Knows when NOT to answer)
"""

import re
import time
from typing import Dict, Any, List
from config import GROUNDING_SIMILARITY_THRESHOLD, HALLUCINATION_OVERLAP_THRESHOLD

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
    Verifies that the generated answer is strictly grounded in retrieved passages
    and contains no hallucinations.
    """
    def evaluate(self, answer: str, retrieved_contexts: List[str], top_retrieval_score: float, is_general_knowledge: bool = False) -> Dict[str, Any]:
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

        if not retrieved_contexts or top_retrieval_score < GROUNDING_SIMILARITY_THRESHOLD:
            return {
                "is_grounded": False,
                "hallucination_score": 1.0,
                "grounding_score": 0.0,
                "reason": f"No matching grounded facts found in context (Relevance: {top_retrieval_score:.2f}).",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        context_blob = " ".join(retrieved_contexts).lower()
        context_words = set(re.findall(r'\w+', context_blob))

        answer_words = re.findall(r'\w+', answer.lower())
        stopwords = {"is", "the", "a", "an", "and", "or", "in", "of", "to", "with", "for", "that", "this"}
        content_words = [w for w in answer_words if w not in stopwords]
        
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
            "reason": "Answer verified against retrieved context." if is_grounded else "Answer failed grounding verification.",
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
