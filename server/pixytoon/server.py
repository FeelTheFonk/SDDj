"""FastAPI WebSocket server — entry point for the PixyToon server."""

from __future__ import annotations

import asyncio
import logging
import threading
import time as _time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .engine import DiffusionEngine, GenerationCancelled
from .protocol import (
    Action,
    AnimationCompleteResponse,
    AnimationFrameResponse,
    ErrorResponse,
    ListResponse,
    PongResponse,
    ProgressResponse,
    RealtimeReadyResponse,
    RealtimeResultResponse,
    RealtimeStoppedResponse,
    Request,
)
from . import __version__
from . import lora_manager, palette_manager, ti_manager
from .postprocess import warmup_numba

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pixytoon.server")

# ─────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────

engine = DiffusionEngine()
_generate_lock: asyncio.Lock | None = None
_generating: dict[int, threading.Event] = {}  # connection id -> cancel event
_active_connections: set[WebSocket] = set()
_MAX_CONNECTIONS = 5
_ws_counter = 0  # monotonic connection ID (avoids id() reuse)
_realtime_owner: int | None = None  # ws_id that owns realtime mode (None = free)
_realtime_timeout_task: asyncio.Task | None = None  # auto-stop timer


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _generate_lock
    _generate_lock = asyncio.Lock()

    log.info("PixyToon server starting — loading diffusion engine...")
    loop = asyncio.get_running_loop()

    # Run engine load and Numba JIT warmup in parallel (independent tasks)
    async def _load_engine():
        await loop.run_in_executor(None, engine.load)

    async def _warmup_numba():
        log.info("Pre-compiling Numba JIT kernels...")
        await loop.run_in_executor(None, warmup_numba)
        log.info("Numba JIT warmup complete")

    await asyncio.gather(_load_engine(), _warmup_numba())

    log.info("Engine loaded. WebSocket ready on ws://%s:%d/ws", settings.host, settings.port)
    yield

    # Graceful shutdown: close active WebSocket connections
    for ws in list(_active_connections):
        try:
            await ws.close(code=1001, reason="Server shutting down")
        except Exception:
            pass
    _active_connections.clear()

    engine.unload()
    log.info("Engine unloaded.")


app = FastAPI(title="PixyToon Server", version=__version__, lifespan=_lifespan)


# ─────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    if len(_active_connections) >= _MAX_CONNECTIONS:
        await websocket.send_text(ErrorResponse(
            code="MAX_CONNECTIONS", message="Too many connections"
        ).model_dump_json())
        await websocket.close()
        return
    _active_connections.add(websocket)

    global _ws_counter
    _ws_counter += 1
    ws_id = _ws_counter
    _generating[ws_id] = threading.Event()
    log.info("Client connected: %s", websocket.client)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                req = Request.model_validate_json(raw)
            except Exception as e:
                await _send(websocket, ErrorResponse(
                    code="INVALID_REQUEST",
                    message=f"Malformed request: {e}",
                ))
                continue

            await _handle(websocket, req, ws_id)

    except WebSocketDisconnect:
        log.info("Client disconnected: %s", websocket.client)
        if _generating.get(ws_id, threading.Event()).is_set():
            engine.cancel()
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        if _generating.get(ws_id, threading.Event()).is_set():
            engine.cancel()
    finally:
        _active_connections.discard(websocket)
        _generating.pop(ws_id, None)
        # Clean up realtime session if this client owned it
        await _cleanup_realtime(ws_id)


def _make_thread_callback(websocket: WebSocket, loop: asyncio.AbstractEventLoop, timeout: float = 1.0):
    """Create a thread-safe callback that sends responses via the event loop.

    Used by both generate and animate handlers to send progress/frame updates
    from the engine thread back through the WebSocket.
    """
    def callback(response) -> None:
        try:
            try:
                if websocket.application_state.name != "CONNECTED":
                    return
            except AttributeError:
                return  # Starlette internal changed — skip safely
            fut = asyncio.run_coroutine_threadsafe(
                _send(websocket, response), loop
            )
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass  # Send failed or timed out — skip
        except Exception:
            pass
    return callback


async def _handle(websocket: WebSocket, req: Request, ws_id: int) -> None:
    """Dispatch request by action type."""
    try:
        if req.action == Action.PING:
            await _send(websocket, PongResponse())

        elif req.action == Action.CANCEL:
            if _generating.get(ws_id, threading.Event()).is_set():
                engine.cancel()
                # Don't send response here — the GenerationCancelled exception
                # handler will send CANCELLED when the generation actually stops.
            else:
                await _send(websocket, PongResponse())  # No-op — nothing to cancel

        elif req.action == Action.LIST_LORAS:
            items = lora_manager.list_loras()
            await _send(websocket, ListResponse(list_type="loras", items=items))

        elif req.action == Action.LIST_PALETTES:
            items = palette_manager.list_palettes()
            await _send(websocket, ListResponse(list_type="palettes", items=items))

        elif req.action == Action.LIST_CONTROLNETS:
            from . import pipeline_factory
            await _send(websocket, ListResponse(
                list_type="controlnets",
                items=[m.value for m in pipeline_factory.CONTROLNET_IDS],
            ))

        elif req.action == Action.LIST_EMBEDDINGS:
            items = ti_manager.list_embeddings()
            await _send(websocket, ListResponse(list_type="embeddings", items=items))

        elif req.action == Action.REALTIME_START:
            await _handle_realtime_start(websocket, req, ws_id)

        elif req.action == Action.REALTIME_FRAME:
            await _handle_realtime_frame(websocket, req, ws_id)

        elif req.action == Action.REALTIME_UPDATE:
            await _handle_realtime_update(websocket, req, ws_id)

        elif req.action == Action.REALTIME_STOP:
            await _handle_realtime_stop(websocket, ws_id)

        elif req.action == Action.GENERATE:
            # Block if realtime mode is active
            if _realtime_owner is not None:
                await _send(websocket, ErrorResponse(
                    code="REALTIME_BUSY",
                    message="Cannot generate while real-time mode is active",
                ))
                return
            gen_req = req.to_generate_request()
            loop = asyncio.get_running_loop()

            on_progress = _make_thread_callback(websocket, loop, timeout=1.0)

            # Serialize GPU access — pipeline is NOT thread-safe
            if _generate_lock is None:
                raise RuntimeError("Server not fully initialized")
            async with _generate_lock:
                _generating[ws_id].set()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: engine.generate(gen_req, on_progress=on_progress),
                        ),
                        timeout=settings.generation_timeout,
                    )
                except asyncio.TimeoutError:
                    engine.cancel()
                    raise RuntimeError(
                        f"Generation timed out after {settings.generation_timeout:.0f}s"
                    )
                finally:
                    _generating[ws_id].clear()
            await _send(websocket, result)

        elif req.action == Action.GENERATE_ANIMATION:
            # Block if realtime mode is active
            if _realtime_owner is not None:
                await _send(websocket, ErrorResponse(
                    code="REALTIME_BUSY",
                    message="Cannot animate while real-time mode is active",
                ))
                return
            anim_req = req.to_animation_request()

            # Server-side frame count validation (protocol allows 120, config may differ)
            if anim_req.frame_count > settings.max_animation_frames:
                await _send(websocket, ErrorResponse(
                    code="INVALID_REQUEST",
                    message=f"frame_count {anim_req.frame_count} exceeds max {settings.max_animation_frames}",
                ))
                return

            loop = asyncio.get_running_loop()

            on_anim_progress = _make_thread_callback(websocket, loop, timeout=1.0)
            on_anim_frame = _make_thread_callback(websocket, loop, timeout=2.0)

            if _generate_lock is None:
                raise RuntimeError("Server not fully initialized")
            # Auto-scale timeout for animation (30s per frame minimum)
            anim_timeout = max(
                settings.generation_timeout,
                anim_req.frame_count * 30,
            )
            async with _generate_lock:
                _generating[ws_id].set()
                try:
                    t0 = _time.perf_counter()
                    frames = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: engine.generate_animation(
                                anim_req,
                                on_frame=on_anim_frame,
                                on_progress=on_anim_progress,
                            ),
                        ),
                        timeout=anim_timeout,
                    )
                    total_ms = int((_time.perf_counter() - t0) * 1000)
                except asyncio.TimeoutError:
                    engine.cancel()
                    raise RuntimeError(
                        f"Animation timed out after {anim_timeout:.0f}s"
                    )
                finally:
                    _generating[ws_id].clear()

            await _send(websocket, AnimationCompleteResponse(
                total_frames=len(frames),
                total_time_ms=total_ms,
                tag_name=anim_req.tag_name,
            ))

        else:
            await _send(websocket, ErrorResponse(
                code="UNKNOWN_ACTION",
                message=f"Unknown action: {req.action}",
            ))

    except GenerationCancelled:
        log.info("Generation cancelled by client")
        try:
            await _send(websocket, ErrorResponse(code="CANCELLED", message="Generation cancelled"))
        except Exception:
            pass
    except torch.cuda.OutOfMemoryError as e:
        log.exception("CUDA OOM: %s", e)
        try:
            await _send(websocket, ErrorResponse(code="OOM", message=str(e)))
        except Exception:
            pass
    except Exception as e:
        log.exception("Handler error: %s", e)
        if "timed out" in str(e).lower():
            code = "TIMEOUT"
        else:
            code = "ENGINE_ERROR"
        try:
            await _send(websocket, ErrorResponse(code=code, message=str(e)))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# REAL-TIME PAINT HANDLERS
# ─────────────────────────────────────────────────────────────

async def _handle_realtime_start(websocket: WebSocket, req: Request, ws_id: int) -> None:
    global _realtime_owner, _realtime_timeout_task

    if _realtime_owner is not None and _realtime_owner != ws_id:
        await _send(websocket, ErrorResponse(
            code="REALTIME_BUSY",
            message="Another client is using real-time mode",
        ))
        return

    if _generate_lock is None:
        raise RuntimeError("Server not fully initialized")

    # Check if generate_lock is currently held (generation in progress)
    if _generate_lock.locked():
        await _send(websocket, ErrorResponse(
            code="GPU_BUSY",
            message="Cannot start real-time mode while a generation is in progress",
        ))
        return

    rt_req = req.to_realtime_start()
    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(
            None, lambda: engine.start_realtime(rt_req),
        )
        _realtime_owner = ws_id
        _reset_realtime_timeout()
        await _send(websocket, result)
    except Exception as e:
        log.exception("Failed to start realtime: %s", e)
        await _send(websocket, ErrorResponse(code="ENGINE_ERROR", message=str(e)))


async def _handle_realtime_frame(websocket: WebSocket, req: Request, ws_id: int) -> None:
    if _realtime_owner != ws_id:
        await _send(websocket, ErrorResponse(
            code="REALTIME_NOT_ACTIVE",
            message="Real-time mode not active for this connection",
        ))
        return

    frame_req = req.to_realtime_frame()
    if not frame_req.image:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST",
            message="realtime_frame requires image",
        ))
        return

    _reset_realtime_timeout()

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: engine.process_realtime_frame(
                frame_req.image,
                frame_req.frame_id,
                prompt_override=frame_req.prompt,
            ),
        )
        await _send(websocket, result)
    except torch.cuda.OutOfMemoryError as e:
        log.error("CUDA OOM during realtime frame: %s", e)
        await _send(websocket, ErrorResponse(code="OOM", message=str(e)))
    except Exception as e:
        log.warning("Realtime frame error: %s", e)
        await _send(websocket, ErrorResponse(code="ENGINE_ERROR", message=str(e)))


async def _handle_realtime_update(websocket: WebSocket, req: Request, ws_id: int) -> None:
    if _realtime_owner != ws_id:
        return  # Silently ignore updates from non-owner

    update_req = req.to_realtime_update()
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None, lambda: engine.update_realtime_params(update_req),
        )
    except Exception as e:
        log.warning("Realtime update error: %s", e)


async def _handle_realtime_stop(websocket: WebSocket, ws_id: int) -> None:
    if _realtime_owner != ws_id:
        await _send(websocket, ErrorResponse(
            code="REALTIME_NOT_ACTIVE",
            message="Real-time mode not active for this connection",
        ))
        return

    await _cleanup_realtime(ws_id)
    await _send(websocket, RealtimeStoppedResponse())


async def _cleanup_realtime(ws_id: int) -> None:
    """Stop realtime mode if owned by ws_id. Safe to call multiple times."""
    global _realtime_owner, _realtime_timeout_task

    if _realtime_owner != ws_id:
        return

    if _realtime_timeout_task is not None:
        _realtime_timeout_task.cancel()
        _realtime_timeout_task = None

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, engine.stop_realtime)
    except Exception as e:
        log.warning("Realtime cleanup error: %s", e)

    _realtime_owner = None
    log.info("Realtime session cleaned up for ws_id=%d", ws_id)


def _reset_realtime_timeout() -> None:
    """Reset the auto-stop timer for realtime mode."""
    global _realtime_timeout_task

    if _realtime_timeout_task is not None:
        _realtime_timeout_task.cancel()

    async def _auto_stop():
        await asyncio.sleep(settings.realtime_timeout)
        global _realtime_owner
        if _realtime_owner is not None:
            log.info("Realtime auto-stop: no frame for %.0fs", settings.realtime_timeout)
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, engine.stop_realtime)
            except Exception:
                pass
            _realtime_owner = None

    try:
        loop = asyncio.get_running_loop()
        _realtime_timeout_task = loop.create_task(_auto_stop())
    except RuntimeError:
        pass  # No running loop (shouldn't happen in normal operation)


async def _send(websocket: WebSocket, response) -> None:
    try:
        text = response.model_dump_json()
        await websocket.send_text(text)
    except (WebSocketDisconnect, RuntimeError):
        pass  # Client already disconnected or connection closing


# ─────────────────────────────────────────────────────────────
# HTTP HEALTH CHECK
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "version": __version__,
        "loaded": engine.is_loaded,
    })


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────────

def main() -> None:
    uvicorn.run(
        "pixytoon.server:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        ws_max_size=50 * 1024 * 1024,  # 50MB max WebSocket message
    )


if __name__ == "__main__":
    main()
