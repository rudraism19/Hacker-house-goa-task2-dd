# Voice RAG Latency Analytics Report
**Total Benchmark Runs**: 50 | **Sub-200ms SLA Pass Rate**: `100.0%`

### Overall End-to-End Pipeline Latency
| Percentile | Latency (ms) | Description |
| :--- | :--- | :--- |
| **P50 (Median)** | **`2.06 ms`** | 50% of requests complete within this duration |
| **P70** | **`2.65 ms`** | 70% of requests complete within this duration |
| **P100 (Max)** | **`4.95 ms`** | Worst-case maximum response latency |
| **Mean +- Std** | `2.2 +- 1.16 ms` | Average execution time |

### Stage-by-Stage Latency Breakdown (Percentiles)
| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Speech-to-Text (STT)** | 0.0 | 0.0 | 0.0 | 0.0 |
| **2. Input Guardrails** | 0.01 | 0.01 | 0.02 | 0.01 |
| **3. Harness Tools & Query Refinement** | 0.21 | 0.23 | 2.3 | 0.24 |
| **4. Vector Retrieval** | 1.51 | 2.07 | 4.21 | 1.65 |
| **5. Answer Synthesis** | 0.15 | 0.18 | 0.48 | 0.14 |
| **6. Grounding Guardrails** | 0.12 | 0.13 | 0.23 | 0.11 |