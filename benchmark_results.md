# Voice RAG Latency Analytics Report
**Total Benchmark Runs**: 20 | **Sub-200ms SLA Pass Rate**: `100.0%`

### Overall End-to-End Pipeline Latency
| Percentile | Latency (ms) | Description |
| :--- | :--- | :--- |
| **P50 (Median)** | **`0.73 ms`** | 50% of requests complete within this duration |
| **P70** | **`0.9 ms`** | 70% of requests complete within this duration |
| **P100 (Max)** | **`17.53 ms`** | Worst-case maximum response latency |
| **Mean +- Std** | `1.61 +- 3.7 ms` | Average execution time |

### Stage-by-Stage Latency Breakdown (Percentiles)
| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Speech-to-Text (STT)** | 0.0 | 0.0 | 0.0 | 0.0 |
| **2. Input Guardrails** | 0.01 | 0.01 | 16.61 | 0.84 |
| **3. Harness Tools & Query Refinement** | 0.02 | 0.02 | 0.07 | 0.02 |
| **4. Vector Retrieval** | 0.57 | 0.64 | 2.65 | 0.61 |
| **5. Answer Synthesis** | 0.05 | 0.05 | 0.12 | 0.05 |
| **6. Grounding Guardrails** | 0.0 | 0.09 | 0.15 | 0.04 |