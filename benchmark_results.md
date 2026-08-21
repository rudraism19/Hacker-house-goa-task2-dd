# Voice RAG Latency Analytics Report
**Total Benchmark Runs**: 20 | **Sub-200ms SLA Pass Rate**: `100.0%`

### Overall End-to-End Pipeline Latency
| Percentile | Latency (ms) | Description |
| :--- | :--- | :--- |
| **P50 (Median)** | **`1.95 ms`** | 50% of requests complete within this duration |
| **P70** | **`2.22 ms`** | 70% of requests complete within this duration |
| **P100 (Max)** | **`3.17 ms`** | Worst-case maximum response latency |
| **Mean +- Std** | `1.9 +- 0.9 ms` | Average execution time |

### Stage-by-Stage Latency Breakdown (Percentiles)
| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Speech-to-Text (STT)** | 0.0 | 0.0 | 0.0 | 0.0 |
| **2. Input Guardrails** | 0.01 | 0.01 | 0.03 | 0.01 |
| **3. Harness Tools & Query Refinement** | 0.02 | 0.02 | 0.04 | 0.02 |
| **4. Vector Retrieval** | 1.74 | 2.04 | 2.95 | 1.71 |
| **5. Answer Synthesis** | 0.04 | 0.04 | 0.07 | 0.04 |
| **6. Grounding Guardrails** | 0.06 | 0.08 | 0.2 | 0.07 |