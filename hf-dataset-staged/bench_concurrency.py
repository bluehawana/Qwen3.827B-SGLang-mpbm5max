#!/usr/bin/env python3
"""Concurrency benchmark for a local OpenAI-compatible server (oMLX or the gateway).

Fires N concurrent chat requests with DISTINCT prompts (so the prefix cache
can't hide the real work), each generating MAX_OUT tokens, and reports
per-request latency + aggregate throughput.

    python3 bench_concurrency.py <model> <N> [prompt_tokens=3000] [max_out=256] [base_url]

Examples:
    python3 bench_concurrency.py mlx-community--Qwen3.8-27B-8bit 4          # direct to oMLX :8000
    python3 bench_concurrency.py mlx-community--Qwen3.8-27B-8bit 6 3000 256 http://127.0.0.1:8080  # via gateway
"""
import json, os, sys, time, threading, urllib.request, random

def _key():
    try:
        return json.load(open(os.path.expanduser("~/.omlx/settings.json")))["auth"]["api_key"]
    except Exception:
        return "omlx"

KEY = os.environ.get("QCLAUDE_API_KEY", _key())
model = sys.argv[1]
N = int(sys.argv[2])
PROMPT_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
MAX_OUT = int(sys.argv[4]) if len(sys.argv) > 4 else 256
BASE = (sys.argv[5] if len(sys.argv) > 5 else os.environ.get("BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
URL = BASE + "/v1/chat/completions"

WORDS = ("chassis axle transfer lock coil spring radiator piston valve torque desert dune "
         "rock crawl winch snorkel diff gear low range hub knuckle prop shaft").split()

def prompt(i):
    random.seed(1000 + i)
    body = " ".join(random.choice(WORDS) for _ in range(int(PROMPT_TOK * 0.75)))
    return (f"Session {i}. Read this parts log then answer.\n{body}\n\n"
            f"Task: write a detailed workshop guide on rebuilding a Land Cruiser front axle, "
            f"mentioning at least five parts from the log.")

results = [None] * N
def worker(i):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt(i)}],
        "max_tokens": MAX_OUT, "temperature": 0.7, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=payload,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        u = r.get("usage", {})
        results[i] = (time.time() - t0, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), None)
    except Exception as e:
        results[i] = (time.time() - t0, 0, 0, str(e)[:100])

T0 = time.time()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
for t in threads: t.start()
for t in threads: t.join()
wall = time.time() - T0

ok = [r for r in results if r and not r[3]]
errs = [r for r in results if r and r[3]]
tot_in = sum(r[1] for r in ok); tot_out = sum(r[2] for r in ok)
lat = sorted(r[0] for r in ok)
p50 = lat[len(lat)//2] if lat else 0
print(f"model={model.split('--')[-1]:6s} N={N:2d} in≈{PROMPT_TOK} out={MAX_OUT} via={BASE.split('//')[-1]:15s} "
      f"wall={wall:6.1f}s lat_p50={p50:6.1f}s lat_max={lat[-1] if lat else 0:6.1f}s "
      f"agg={tot_out/wall:5.1f} out-tok/s ok={len(ok)} err={len(errs)}")
for e in errs[:3]:
    print("   err:", e[3])
