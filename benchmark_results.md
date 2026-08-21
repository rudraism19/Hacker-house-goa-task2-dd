# Voice RAG Latency Analytics Report
**Total Benchmark Runs**: 50 | **Sub-200ms SLA Pass Rate**: `100.0%`

### Overall End-to-End Pipeline Latency
| Percentile | Latency (ms) | Description |
| :--- | :--- | :--- |
| **P50 (Median)** | **`2.43 ms`** | 50% of requests complete within this duration |
| **P70** | **`2.64 ms`** | 70% of requests complete within this duration |
| **P100 (Max)** | **`13.68 ms`** | Worst-case maximum response latency |
| **Mean +- Std** | `2.59 +- 2.0 ms` | Average execution time |

### Stage-by-Stage Latency Breakdown (Percentiles)
| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Speech-to-Text (STT)** | 0.0 | 0.0 | 0.0 | 0.0 |
| **2. Input Guardrails** | 0.01 | 0.01 | 9.29 | 0.2 |
| **3. Harness Tools & Query Refinement** | 0.02 | 0.03 | 0.34 | 0.03 |
| **4. Vector Retrieval** | 2.18 | 2.32 | 5.6 | 2.15 |
| **5. Answer Synthesis** | 0.05 | 0.06 | 0.57 | 0.06 |
| **6. Grounding Guardrails** | 0.08 | 0.1 | 0.24 | 0.08 |