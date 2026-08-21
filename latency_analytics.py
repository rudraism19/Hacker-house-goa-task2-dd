"""
Latency Analytics & Micro-Benchmarking Engine
Calculates P50, P70, P100 percentiles across pipeline stages and test suites.
"""

import numpy as np
from typing import List, Dict, Any
from model_harness import VoiceRAGResponse

class LatencyAnalyticsEngine:
    """
    Computes statistical percentiles (P50, P70, P100) and SLA metrics across benchmark runs.
    """
    @staticmethod
    def calculate_percentiles(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"p50": 0.0, "p70": 0.0, "p100": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        arr = np.array(values, dtype=np.float32)
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p70": round(float(np.percentile(arr, 70)), 2),
            "p100": round(float(np.percentile(arr, 100)), 2),
            "mean": round(float(np.mean(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "std": round(float(np.std(arr)), 2)
        }

    def compute_suite_analytics(self, responses: List[VoiceRAGResponse]) -> Dict[str, Any]:
        if not responses:
            return {}

        total_latencies = [r.total_latency_ms for r in responses]
        total_stats = self.calculate_percentiles(total_latencies)

        # Stage breakdowns
        stage_names = ["stt_ms", "input_guardrail_ms", "harness_tools_ms", "retrieval_ms", "synthesis_ms", "grounding_guardrail_ms"]
        stage_breakdown = {}

        for stage in stage_names:
            stage_vals = [r.stage_latencies_ms.get(stage, 0.0) for r in responses]
            stage_breakdown[stage] = self.calculate_percentiles(stage_vals)

        # SLA Compliance (<200ms)
        sla_passed_count = sum(1 for r in responses if r.met_sla_200ms)
        sla_pass_rate = round((sla_passed_count / len(responses)) * 100.0, 1)

        return {
            "total_samples": len(responses),
            "overall_latency": total_stats,
            "stage_breakdown": stage_breakdown,
            "sla_target_ms": 200.0,
            "sla_pass_rate_percent": sla_pass_rate
        }

    @staticmethod
    def print_markdown_report(analytics: Dict[str, Any]) -> str:
        """
        Formats latency analytics into a readable Markdown report.
        """
        overall = analytics.get("overall_latency", {})
        breakdown = analytics.get("stage_breakdown", {})
        sla_pass = analytics.get("sla_pass_rate_percent", 0.0)
        samples = analytics.get("total_samples", 0)

        report = [
            f"# Voice RAG Latency Analytics Report",
            f"**Total Benchmark Runs**: {samples} | **Sub-200ms SLA Pass Rate**: `{sla_pass}%`\n",
            f"### Overall End-to-End Pipeline Latency",
            f"| Percentile | Latency (ms) | Description |",
            f"| :--- | :--- | :--- |",
            f"| **P50 (Median)** | **`{overall.get('p50', 0.0)} ms`** | 50% of requests complete within this duration |",
            f"| **P70** | **`{overall.get('p70', 0.0)} ms`** | 70% of requests complete within this duration |",
            f"| **P100 (Max)** | **`{overall.get('p100', 0.0)} ms`** | Worst-case maximum response latency |",
            f"| **Mean +- Std** | `{overall.get('mean', 0.0)} +- {overall.get('std', 0.0)} ms` | Average execution time |",
            f"\n### Stage-by-Stage Latency Breakdown (Percentiles)",
            f"| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |",
            f"| :--- | :---: | :---: | :---: | :---: |"
        ]

        stage_labels = {
            "stt_ms": "1. Speech-to-Text (STT)",
            "input_guardrail_ms": "2. Input Guardrails",
            "harness_tools_ms": "3. Harness Tools & Query Refinement",
            "retrieval_ms": "4. Vector Retrieval",
            "synthesis_ms": "5. Answer Synthesis",
            "grounding_guardrail_ms": "6. Grounding Guardrails"
        }

        for stg, label in stage_labels.items():
            stg_stat = breakdown.get(stg, {})
            report.append(f"| **{label}** | {stg_stat.get('p50', 0.0)} | {stg_stat.get('p70', 0.0)} | {stg_stat.get('p100', 0.0)} | {stg_stat.get('mean', 0.0)} |")

        return "\n".join(report)

if __name__ == "__main__":
    anal = LatencyAnalyticsEngine()
    dummy_stats = anal.calculate_percentiles([120.5, 145.0, 180.2, 195.1, 110.0])
    print("Dummy Stats:", dummy_stats)
