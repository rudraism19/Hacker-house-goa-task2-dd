# Voice RAG Latency Analytics Report
**Total Benchmark Runs**: 50 | **Sub-200ms SLA Pass Rate**: `100.0%`

### Overall End-to-End Pipeline Latency
| Percentile | Latency (ms) | Description |
| :--- | :--- | :--- |
| **P50 (Median)** | **`1.16 ms`** | 50% of requests complete within this duration |
| **P70** | **`1.31 ms`** | 70% of requests complete within this duration |
| **P100 (Max)** | **`5.36 ms`** | Worst-case maximum response latency |
| **Mean +- Std** | `1.24 +- 0.84 ms` | Average execution time |

### Stage-by-Stage Latency Breakdown (Percentiles)
| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Speech-to-Text (STT)** | 0.0 | 0.0 | 0.0 | 0.0 |
| **2. Input Guardrails** | 0.01 | 0.01 | 0.02 | 0.01 |
| **3. Harness Tools & Query Refinement** | 0.18 | 0.21 | 3.81 | 0.25 |
| **4. Vector Retrieval** | 0.73 | 0.84 | 2.18 | 0.76 |
| **5. Answer Synthesis** | 0.06 | 0.08 | 0.4 | 0.07 |
| **6. Grounding Guardrails** | 0.11 | 0.12 | 0.43 | 0.09 |