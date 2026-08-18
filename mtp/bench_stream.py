#!/usr/bin/env python3
"""Single-stream decode benchmark against an OpenAI-compatible server (oMLX).
usage: bench_stream.py <model_id> [max_tokens=400] [reasoning_effort]
env: OMLX_BASE (default http://127.0.0.1:8000), OMLX_API_KEY (default: ~/.omlx/settings.json)
Reports TTFT and decode tok/s from the streamed usage block."""
import json, sys, time, urllib.request
model = sys.argv[1]; max_tokens = int(sys.argv[2]) if len(sys.argv)>2 else 400
effort = sys.argv[3] if len(sys.argv)>3 else None
import os
KEY=os.environ.get("OMLX_API_KEY") or (lambda: (json.load(open(os.path.expanduser("~/.omlx/settings.json"))).get("auth",{}).get("api_key","omlx")))()
BASE=os.environ.get("OMLX_BASE","http://127.0.0.1:8000")
prompt = ("Write a Python function `lru_cache_with_ttl(maxsize, ttl_seconds)` decorator that caches "
          "results with both an LRU eviction policy and per-entry expiry. Include a short docstring, "
          "thread-safety with a lock, and a small usage example. Then explain the design in 3 bullet points.")
body = {"model": model, "messages":[{"role":"user","content":prompt}], "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True}, "temperature": 0.7}
if effort: body["chat_template_kwargs"] = {"reasoning_effort": effort}
req = urllib.request.Request(BASE+"/v1/chat/completions", data=json.dumps(body).encode(),
      headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
t0=time.time(); first=None; n=0; usage=None; text=""
with urllib.request.urlopen(req, timeout=3000) as r:
    for line in r:
        line=line.decode().strip()
        if not line.startswith("data:") or line=="data: [DONE]": continue
        d=json.loads(line[5:])
        if d.get("usage"): usage=d["usage"]
        for ch in d.get("choices",[]):
            delta=ch.get("delta",{})
            piece=(delta.get("content") or "")+(delta.get("reasoning_content") or delta.get("reasoning") or "")
            if piece:
                if first is None: first=time.time()
                text+=piece
t1=time.time()
comp = usage.get("completion_tokens") if usage else None
ttft = (first-t0) if first else None
dec = (t1-first) if first else None
print(f"model={model} effort={effort} completion_tokens={comp} prompt_tokens={usage.get('prompt_tokens') if usage else None} "
      f"TTFT={ttft:.2f}s decode={dec:.2f}s -> {comp/dec if comp and dec else 0:.1f} tok/s (wall {t1-t0:.1f}s)")
print("---", text[:300].replace("\n"," "), "...")
