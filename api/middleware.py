"""
middleware.py
=============
FastAPI middleware for production observability.

Provides:
  - Request ID injection (UUID per request for distributed tracing)
  - Request/response logging (method, path, status, latency)
  - Exception handling with structured error responses
"""

from __future__ import annotations

import time
import uuid
import traceback
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with:
      - Unique request_id for distributed tracing
      - HTTP method and path
      - Response status code
      - Total latency in milliseconds
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start_time = time.perf_counter()

        # Log incoming request
        logger.info(
            f"[{request_id}] → {request.method} {request.url.path} "
            f"| client={request.client.host if request.client else 'unknown'}"
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            # Catch unhandled exceptions — return 500 with request_id
            logger.error(
                f"[{request_id}] Unhandled exception: {exc}\n"
                f"{traceback.format_exc()}"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "detail": str(exc),
                },
            )

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Log response
        log_fn = logger.info if response.status_code < 400 else logger.warning
        log_fn(
            f"[{request_id}] ← {response.status_code} | {latency_ms:.1f}ms | "
            f"{request.method} {request.url.path}"
        )

        # Inject request metadata headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.1f}"

        return response
