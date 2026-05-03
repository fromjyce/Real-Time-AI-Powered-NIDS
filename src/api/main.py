"""
FastAPI application entry point.
Exposes REST endpoints and a WebSocket channel that pushes real-time
alerts to connected dashboard clients via a Kafka consumer.
"""

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from typing import Set

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.api.routes.alerts import router as alerts_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.models import router as models_router
from src.config import settings
from src.ingestion.kafka_consumer import NIDSConsumer

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
ALERTS_TOTAL = Counter("nids_alerts_total", "Total alerts generated", ["severity", "attack_class"])
ACTIVE_CONNECTIONS = Gauge("nids_ws_connections", "Active WebSocket connections")
THROUGHPUT = Gauge("nids_throughput_pps", "Packets per second")

# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        ACTIVE_CONNECTIONS.set(len(self._connections))
        logger.info("ws_connected", total=len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)
        ACTIVE_CONNECTIONS.set(len(self._connections))

    async def broadcast(self, message: dict) -> None:
        dead: Set[WebSocket] = set()
        for ws in list(self._connections):
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ── Kafka → WebSocket bridge ──────────────────────────────────────────────────
async def kafka_alert_consumer(app: FastAPI) -> None:
    """
    Runs the blocking Kafka consumer in a daemon thread and feeds records
    into an asyncio.Queue so the event loop is never blocked.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=2000)

    consumer = NIDSConsumer(
        topics=[settings.KAFKA_TOPIC_ALERTS],
        group_id="nids-api-ws-bridge",
        auto_offset_reset="latest",
    )
    consumer.start()

    def _poll_thread() -> None:
        try:
            for record in consumer.stream(timeout=1.0):
                asyncio.run_coroutine_threadsafe(queue.put(record), loop)
        finally:
            consumer.close()

    thread = threading.Thread(target=_poll_thread, daemon=True, name="kafka-ws-bridge")
    thread.start()

    try:
        while True:
            record = await queue.get()
            ALERTS_TOTAL.labels(
                severity=record.get("severity", "unknown"),
                attack_class=record.get("attack_class", "unknown"),
            ).inc()
            await manager.broadcast({"type": "alert", "payload": record})
    except asyncio.CancelledError:
        consumer._running.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(kafka_alert_consumer(app))
    logger.info("nids_api_started", host=settings.API_HOST, port=settings.API_PORT)
    yield
    task.cancel()
    logger.info("nids_api_stopped")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="NIDS API",
    description="Real-Time AI-Powered Network Intrusion Detection System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(alerts_router, prefix="/api/v1")
app.include_router(metrics_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"name": "NIDS API", "version": "1.0.0", "status": "running"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # Heartbeat every 30 seconds
            await asyncio.sleep(30)
            await ws.send_json({"type": "heartbeat", "payload": {"ts": time.time()}})
    except WebSocketDisconnect:
        manager.disconnect(ws)
        logger.info("ws_disconnected", total=len(manager._connections))
    except Exception:
        manager.disconnect(ws)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("unhandled_exception", path=str(request.url))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
    )
