import sys
from fastapi.testclient import TestClient
from api import app

def test_api():
    client = TestClient(app)

    # 1. Test GET /
    res_root = client.get("/")
    assert res_root.status_code == 200, f"GET / failed: {res_root.status_code}"
    assert "HACKER" in res_root.text
    assert "VOICE-ENABLED RAG" in res_root.text
    assert "CHUNKING STRATEGIES LAB" in res_root.text
    assert "P50 / P70 / P100" in res_root.text
    print("[PASS] GET / HTML UI: 200 OK & Content Verified")

    # 2. Test POST /query
    res_query = client.post("/query", json={"query_text": "What is a corporation?", "language_code": "en-IN"})
    assert res_query.status_code == 200
    data = res_query.json()
    assert "answer" in data
    assert "total_latency_ms" in data
    assert "grounding_score" in data
    print(f"[PASS] POST /query: 200 OK (Latency: {data['total_latency_ms']}ms, Grounding: {data['grounding_score']})")

    # 3. Test GET /chunking/compare
    res_chunk = client.get("/chunking/compare")
    assert res_chunk.status_code == 200
    chunk_data = res_chunk.json()
    assert "semantic_boundary" in chunk_data
    assert "fixed_overlap" in chunk_data
    print(f"[PASS] GET /chunking/compare: 200 OK ({len(chunk_data)} strategies compared)")

    # 4. Test GET /benchmark
    res_bench = client.get("/benchmark?num_samples=20&strategy=semantic_boundary")
    assert res_bench.status_code == 200
    bench_data = res_bench.json()
    assert "overall_latency" in bench_data
    assert "stage_breakdown" in bench_data
    print(f"[PASS] GET /benchmark: 200 OK (P50: {bench_data['overall_latency']['p50']}ms, SLA: {bench_data['sla_pass_rate_percent']}%)")

    # 5. Test POST /set-gemini-key
    res_key = client.post("/set-gemini-key", json={"key": "test-key-123"})
    assert res_key.status_code == 200
    assert res_key.json()["status"] == "success"
    print("[PASS] POST /set-gemini-key: 200 OK")

if __name__ == "__main__":
    test_api()
