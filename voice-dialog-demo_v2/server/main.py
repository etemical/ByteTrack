"""FastAPI 入口：静态页面 + /ws 语音 WebSocket。"""
import asyncio
import contextlib
import json

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import asr, config
from .pipeline import Session

if config.TTS_ENGINE == "cosyvoice3":
    from . import tts_v3 as tts
else:
    from . import tts

_ready = False


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, asr.load_model)
    print("[startup] ASR ready", flush=True)
    await loop.run_in_executor(None, tts.load_model)
    print(f"[startup] TTS ready, sample_rate={tts.sample_rate()}", flush=True)
    _ready = True
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"ready": _ready}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if not _ready:
        await ws.send_json({"type": "error", "where": "boot", "msg": "models not ready"})
        await ws.close()
        return
    sess = Session(ws)
    nbytes = 0
    print("[ws] client connected", flush=True)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                nbytes += len(msg["bytes"])
                pcm = np.frombuffer(msg["bytes"], dtype=np.int16)
                if len(pcm):
                    sess.feed_audio(pcm)
            elif msg.get("text"):
                try:
                    obj = json.loads(msg["text"])
                except (TypeError, ValueError):
                    continue
                if obj.get("type") == "play_end":
                    sess.on_play_end()
                elif obj.get("type") == "barge":
                    await sess.on_barge()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] error: {e}", flush=True)
    finally:
        await sess.close()
        print(f"[ws] client disconnected, rx_bytes={nbytes}", flush=True)


@app.get("/")
async def index():
    return FileResponse(
        "static/index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/static/app.js")
async def app_js():
    return FileResponse(
        "static/app.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


app.mount("/static", StaticFiles(directory="static"), name="static")


def main():
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
