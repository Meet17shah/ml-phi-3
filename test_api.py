# ─────────────────────────────────────────────────────────────────────────────
# test_api.py — Fire test questions at the running Phi-3 FastAPI server
#
# Usage:
#   1. Start the server first:  uvicorn app:app --host 0.0.0.0 --port 8000
#   2. In a second terminal:    python test_api.py
# ─────────────────────────────────────────────────────────────────────────────

import requests
import time
import json

BASE_URL = "http://localhost:8000"

# ── 1. Health check ──────────────────────────────────────────────────────────
print("=" * 65)
print("Phi-3 ML Q&A API — Test Suite")
print("=" * 65)

print("\n[1] Health check...")
try:
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    resp.raise_for_status()
    health = resp.json()
    print(f"  Status       : {health['status']}")
    print(f"  Model loaded : {health['model_loaded']}")
    print(f"  Device       : {health['device'].upper()}")
    print(f"  GPU          : {health.get('gpu_name', 'N/A')}")
    print(f"  VRAM free    : {health.get('vram_free_gb', 'N/A')} GB")
except requests.exceptions.ConnectionError:
    print("\n❌ Could not connect to server at http://localhost:8000")
    print("   Make sure the server is running:")
    print("   uvicorn app:app --host 0.0.0.0 --port 8000")
    exit(1)

# ── 2. Test questions ─────────────────────────────────────────────────────────
test_questions = [
    "What is overfitting and how do you prevent it?",
    "Explain the difference between L1 and L2 regularization.",
    "What is the Adam optimizer and why is it popular?",
    "How does a Convolutional Neural Network work?",
    "What is LoRA and how does it help fine-tune large language models?",
    "What is gradient descent?",
    "Explain what a transformer is in machine learning.",
]

print(f"\n[2] Running {len(test_questions)} test questions...\n")
print("=" * 65)

total_start = time.perf_counter()
times = []

for i, q in enumerate(test_questions, 1):
    try:
        resp = requests.post(
            f"{BASE_URL}/ask",
            json={"question": q, "max_tokens": 80},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        t    = data["time_seconds"]
        toks = data["tokens_generated"]
        times.append(t)

        print(f" Q{i}: {data['question']}")
        print(f" A{i}: {data['answer']}")
        print(f" ⏱  {t:.2f}s  |  {toks} tokens  |  {toks/t:.0f} tok/s")
        print("─" * 65)

    except requests.exceptions.Timeout:
        print(f" Q{i}: TIMEOUT (> 30s) — model may be overloaded")
        print("─" * 65)
    except Exception as e:
        print(f" Q{i}: ERROR — {e}")
        print("─" * 65)

# ── 3. Summary ────────────────────────────────────────────────────────────────
total_elapsed = time.perf_counter() - total_start
if times:
    print(f"\n{'=' * 65}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Questions answered : {len(times)} / {len(test_questions)}")
    print(f"  Avg response time  : {sum(times)/len(times):.2f}s")
    print(f"  Min response time  : {min(times):.2f}s")
    print(f"  Max response time  : {max(times):.2f}s")
    print(f"  Total wall time    : {total_elapsed:.1f}s")
    avg = sum(times) / len(times)
    if avg <= 3.0:
        print(f"\n✅ All responses within target (<= 3s average). Model is working correctly!")
    else:
        print(f"\n⚠️  Average {avg:.1f}s — above 3s target. Try reducing max_tokens in app.py.")
