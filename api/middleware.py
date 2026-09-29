"""
middleware.py — FastAPI middleware for production observability.
Request ID injection, logging, and exception handling.
"""
from __future__ import annotations
import time, uuid, traceback
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from src import get_logger
logger = get_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()
        logger.info(f"[{request_id}] → {request.method} {request.url.path} | client={request.client.host if request.client else 'unknown'}")
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(f"[{request_id}] Unhandled: {exc}\n{traceback.format_exc()}")
            return JSONResponse(status_code=500, content={"error":"Internal server error","request_id":request_id,"detail":str(exc)})
        latency_ms = (time.perf_counter() - start) * 1000
        log_fn = logger.info if response.status_code < 400 else logger.warning
        log_fn(f"[{request_id}] ← {response.status_code} | {latency_ms:.1f}ms | {request.method} {request.url.path}")
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.1f}"
        return response
