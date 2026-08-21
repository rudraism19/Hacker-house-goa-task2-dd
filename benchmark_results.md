# Voice RAG Latency Analytics Report
**Total Benchmark Runs**: 20 | **Sub-200ms SLA Pass Rate**: `100.0%`

### Overall End-to-End Pipeline Latency
| Percentile | Latency (ms) | Description |
| :--- | :--- | :--- |
| **P50 (Median)** | **`2.37 ms`** | 50% of requests complete within this duration |
| **P70** | **`2.57 ms`** | 70% of requests complete within this duration |
| **P100 (Max)** | **`4.47 ms`** | Worst-case maximum response latency |
| **Mean +- Std** | `2.3 +- 1.17 ms` | Average execution time |

### Stage-by-Stage Latency Breakdown (Percentiles)
| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Speech-to-Text (STT)** | 0.0 | 0.0 | 0.0 | 0.0 |
| **2. Input Guardrails** | 0.01 | 0.01 | 0.03 | 0.01 |
| **3. Harness Tools & Query Refinement** | 0.02 | 0.02 | 0.05 | 0.02 |
| **4. Vector Retrieval** | 2.03 | 2.29 | 4.12 | 2.03 |
| **5. Answer Synthesis** | 0.05 | 0.06 | 0.35 | 0.06 |
| **6. Grounding Guardrails** | 0.09 | 0.12 | 0.51 | 0.1 |