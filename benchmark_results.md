# Voice RAG Latency Analytics Report
**Total Benchmark Runs**: 20 | **Sub-200ms SLA Pass Rate**: `100.0%`

### Overall End-to-End Pipeline Latency
| Percentile | Latency (ms) | Description |
| :--- | :--- | :--- |
| **P50 (Median)** | **`1.78 ms`** | 50% of requests complete within this duration |
| **P70** | **`1.92 ms`** | 70% of requests complete within this duration |
| **P100 (Max)** | **`4.88 ms`** | Worst-case maximum response latency |
| **Mean +- Std** | `1.89 +- 1.27 ms` | Average execution time |

### Stage-by-Stage Latency Breakdown (Percentiles)
| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Speech-to-Text (STT)** | 0.0 | 0.0 | 0.0 | 0.0 |
| **2. Input Guardrails** | 0.01 | 0.01 | 0.01 | 0.01 |
| **3. Harness Tools & Query Refinement** | 0.19 | 0.21 | 0.53 | 0.18 |
| **4. Vector Retrieval** | 1.31 | 1.45 | 4.18 | 1.44 |
| **5. Answer Synthesis** | 0.13 | 0.14 | 0.19 | 0.1 |
| **6. Grounding Guardrails** | 0.12 | 0.13 | 0.18 | 0.11 |