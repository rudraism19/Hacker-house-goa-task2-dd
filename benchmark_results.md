# Voice RAG Latency Analytics Report
**Total Benchmark Runs**: 50 | **Sub-200ms SLA Pass Rate**: `100.0%`

### Overall End-to-End Pipeline Latency
| Percentile | Latency (ms) | Description |
| :--- | :--- | :--- |
| **P50 (Median)** | **`2.47 ms`** | 50% of requests complete within this duration |
| **P70** | **`3.18 ms`** | 70% of requests complete within this duration |
| **P100 (Max)** | **`5.89 ms`** | Worst-case maximum response latency |
| **Mean +- Std** | `2.65 +- 1.58 ms` | Average execution time |

### Stage-by-Stage Latency Breakdown (Percentiles)
| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Speech-to-Text (STT)** | 0.0 | 0.0 | 0.0 | 0.0 |
| **2. Input Guardrails** | 0.01 | 0.01 | 0.16 | 0.01 |
| **3. Harness Tools & Query Refinement** | 0.2 | 0.24 | 3.06 | 0.33 |
| **4. Vector Retrieval** | 1.81 | 2.38 | 5.18 | 1.93 |
| **5. Answer Synthesis** | 0.16 | 0.19 | 0.9 | 0.18 |
| **6. Grounding Guardrails** | 0.12 | 0.15 | 0.41 | 0.13 |