"""FunASR ASR：流式 paraformer-zh-streaming 出字/打断，句末离线模型整句覆盖。"""
import os
import queue
import re
import threading
import time

import numpy as np

from . import config

_model = None
_offline = None
_offline_failed = False
_model_lock = threading.Lock()
_hotword_ok = True
_offline_hotword_ok = True
_TAG_RE = re.compile(r"<\|[^|>]*\|>")

_CHUNK_SAMPLES = int(config.ASR_SAMPLE_RATE * config.ASR_CHUNK_MS / 1000)  # 9600


def _is_sensevoice(path: str) -> bool:
    return "sensevoice" in (path or "").lower()


def _clean_offline_text(text: str) -> str:
    text = _TAG_RE.sub("", text or "")
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        text = rich_transcription_postprocess(text)
    except Exception:
        pass
    return (text or "").strip()


def load_model():
    global _model, _offline, _offline_failed
    from funasr import AutoModel

    if _model is None:
        _model = AutoModel(
            model=config.ASR_MODEL,
            model_revision=config.ASR_MODEL_REVISION,
            device=config.ASR_DEVICE,
            disable_update=True,
        )
        warm = np.zeros(config.ASR_SAMPLE_RATE, dtype=np.float32)
        c = {}
        _model.generate(input=warm, cache=c, is_final=True,
                        chunk_size=[0, 10, 5], encoder_chunk_look_back=4,
                        decoder_chunk_look_back=1)
        print("[ASR] streaming paraformer-online ready", flush=True)

    if _offline is None and not _offline_failed and config.ASR_OFFLINE_MODEL:
        path = config.ASR_OFFLINE_MODEL
        if os.path.isdir(path) or not os.path.isabs(path):
            try:
                kw = dict(model=path, device=config.ASR_DEVICE, disable_update=True)
                _offline = AutoModel(**kw)
                warm = np.zeros(config.ASR_SAMPLE_RATE, dtype=np.float32)
                if _is_sensevoice(path):
                    _offline.generate(input=warm, language="zh", use_itn=False)
                else:
                    _offline.generate(input=warm)
                name = "SenseVoice" if _is_sensevoice(path) else "paraformer-zh"
                print(f"[ASR] offline {name} ready", flush=True)
            except Exception as e:
                _offline_failed = True
                _offline = None
                print(f"[ASR] offline model load failed, using streaming refine: {e}", flush=True)
        else:
            _offline_failed = True
            print(f"[ASR] offline model missing: {path}", flush=True)
    return _model


class AsrSession:
    """一条用户语句的流式识别。回调在 worker 线程触发，调用方负责 marshal 回事件循环。"""

    def __init__(self, on_partial, on_final):
        self.on_partial = on_partial  # (accumulated_text)
        self.on_final = on_final      # (final_text)
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._acc_text = ""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, pcm_f32: np.ndarray):
        self._q.put(pcm_f32)

    def finish(self):
        self._q.put(None)  # 结束标记：flush + is_final

    def close(self):
        self._stop.set()

    def _generate(self, model, cache, chunk: np.ndarray, is_final: bool):
        global _hotword_ok
        kw = dict(
            input=chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=[0, 10, 5],
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1,
        )
        if _hotword_ok and config.ASR_HOTWORD:
            kw["hotword"] = config.ASR_HOTWORD
        try:
            with _model_lock:
                return model.generate(**kw)
        except TypeError:
            _hotword_ok = False
            kw.pop("hotword", None)
            with _model_lock:
                return model.generate(**kw)

    def _generate_offline(self, audio: np.ndarray):
        global _offline_hotword_ok
        kw = dict(input=audio)
        if _is_sensevoice(config.ASR_OFFLINE_MODEL):
            kw.update(language="zh", use_itn=True)
        if _offline_hotword_ok and config.ASR_HOTWORD:
            kw["hotword"] = config.ASR_HOTWORD
        try:
            with _model_lock:
                return _offline.generate(**kw)
        except TypeError:
            _offline_hotword_ok = False
            kw.pop("hotword", None)
            with _model_lock:
                return _offline.generate(**kw)

    def _decode(self, model, cache, chunk: np.ndarray, is_final: bool):
        if len(chunk) == 0:
            chunk = np.zeros(960, dtype=np.float32)  # 60ms 静音触发 flush
        res = self._generate(model, cache, chunk, is_final)
        if res and res[0].get("text"):
            self._acc_text += res[0]["text"]
            if not is_final and self.on_partial:
                self.on_partial(self._acc_text)

    def _refine_final(self, stream_model, audio: np.ndarray) -> str:
        """句末整句识别：优先离线模型，否则用流式模型再解。"""
        streamed = self._acc_text.strip()
        if len(audio) < config.ASR_SAMPLE_RATE // 4:
            return streamed
        t0 = time.perf_counter()
        try:
            if _offline is not None:
                res = self._generate_offline(audio)
            else:
                res = self._generate(stream_model, {}, audio, True)
        except Exception as e:
            print(f"[ASR] refine failed: {e}", flush=True)
            return streamed
        refined = (res[0].get("text") or "").strip() if res else ""
        if _offline is not None:
            refined = _clean_offline_text(refined)
        ms = (time.perf_counter() - t0) * 1000
        src = "sensevoice" if _is_sensevoice(config.ASR_OFFLINE_MODEL) else (
            "offline" if _offline is not None else "stream")
        if refined and refined != streamed:
            print(f"[ASR] refine/{src} '{streamed}' -> '{refined}' ({ms:.0f}ms)", flush=True)
            return refined
        if refined:
            print(f"[ASR] refine/{src} unchanged '{streamed}' ({ms:.0f}ms)", flush=True)
        return streamed or refined

    def _run(self):
        model = load_model()
        cache = {}
        buf = np.zeros(0, dtype=np.float32)
        uttered: list[np.ndarray] = []
        final = False
        flushed = False
        while not self._stop.is_set() and not flushed:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                final = True
            else:
                buf = np.concatenate([buf, item])
                uttered.append(item)

            while len(buf) >= _CHUNK_SAMPLES or (final and not flushed):
                take = min(len(buf), _CHUNK_SAMPLES)
                chunk, buf = buf[:take], buf[take:]
                is_final = final and len(buf) == 0
                self._decode(model, cache, chunk, is_final)
                if is_final:
                    flushed = True
                    audio = np.concatenate(uttered) if uttered else np.zeros(0, dtype=np.float32)
                    text = self._refine_final(model, audio)
                    if self.on_final:
                        self.on_final(text)
                    break
