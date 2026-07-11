#!/usr/bin/env python3
"""
Round-robin HTTP router for 4 vLLM worker instances.

Listens on port 8090 (what UC4 config's vllm_url points to).
Distributes requests across workers on ports 8091-8094.
"""

import itertools
import uvicorn
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

BACKENDS = [
    "http://localhost:8091",
    "http://localhost:8092",
    "http://localhost:8093",
    "http://localhost:8094",
]
PORT = 8090
TIMEOUT = 120.0

_cycle = itertools.cycle(BACKENDS)
app = FastAPI(title="vLLM Round-Robin Router")


@app.get("/health")
async def health():
    """Return OK if any backend is healthy."""
    async with httpx.AsyncClient(timeout=5) as client:
        for backend in BACKENDS:
            try:
                r = await client.get(f"{backend}/health")
                if r.status_code == 200:
                    return {"status": "ok", "healthy_backends": backend}
            except Exception:
                continue
    return Response(content="no healthy backends", status_code=503)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str):
    backend = next(_cycle)
    url = f"{backend}/{path}"
    body = await request.body()

    # Strip headers that cause issues when proxying
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            params=dict(request.query_params),
            content=body,
        )

    # Pass through response headers (drop hop-by-hop)
    skip = {"transfer-encoding", "connection", "keep-alive"}
    resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in skip}

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
    )


if __name__ == "__main__":
    print(f"vLLM router starting on port {PORT}")
    print(f"Backends: {BACKENDS}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
