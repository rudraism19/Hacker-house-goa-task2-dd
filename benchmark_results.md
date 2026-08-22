# Voice RAG Latency Analytics Report
**Total Benchmark Runs**: 20 | **Sub-200ms SLA Pass Rate**: `100.0%`

### Overall End-to-End Pipeline Latency
| Percentile | Latency (ms) | Description |
| :--- | :--- | :--- |
| **P50 (Median)** | **`0.89 ms`** | 50% of requests complete within this duration |
| **P70** | **`0.95 ms`** | 70% of requests complete within this duration |
| **P100 (Max)** | **`2.73 ms`** | Worst-case maximum response latency |
| **Mean +- Std** | `0.89 +- 0.57 ms` | Average execution time |

### Stage-by-Stage Latency Breakdown (Percentiles)
| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Speech-to-Text (STT)** | 0.0 | 0.0 | 0.0 | 0.0 |
| **2. Input Guardrails** | 0.01 | 0.01 | 0.03 | 0.01 |
| **3. Harness Tools & Query Refinement** | 0.03 | 0.03 | 0.16 | 0.03 |
| **4. Vector Retrieval** | 0.69 | 0.73 | 2.38 | 0.67 |
| **5. Answer Synthesis** | 0.06 | 0.06 | 0.09 | 0.05 |
| **6. Grounding Guardrails** | 0.0 | 0.11 | 0.32 | 0.06 |