"""CosyVoice3 流式 TTS。与 tts.py（CosyVoice2）接口一致，由调用方按 TTS_ENGINE 选用。

与 v2 的差异：
- 无 load_jit（v3 未提供 flow encoder 的 jit 导出），加速靠 vLLM + TRT
- prompt_text 必须含 <|endofprompt|>（CosyVoice3LM 硬性断言），指令文本放在其前面
- v3 自带 FSQ silent/breath token 过滤（instruct/语气词不会被念出来）
- token_hop_len 需匹配训练 static_chunk_size（25），不随意调小
"""
import queue
import sys
import threading

import numpy as np
import torch

from . import config

_tts = None
_job_q: queue.Queue = queue.Queue()
_worker_started = False

SPK_ID = "xiaoyun"


def _to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(x, -0.98, 0.98)
    return (x * 32767.0).astype(np.int16).tobytes()


class _ChunkSmoother:
    """句首/停顿后再起：去静音 + 淡入。不做长交叉淡化（CosyVoice3 重算块相位对不齐，叠了会更爆）。"""

    def __init__(self, sr: int = 24000):
        self.sr = sr
        self.startup = True
        self.fade_n = int(sr * 0.050)
        self.pause_fade = int(sr * 0.036)
        self.held: np.ndarray | None = None
        self.step_n = max(8, int(sr * 0.003))

    def reset(self):
        self.startup = True
        self.held = None

    def begin_stream(self):
        self.startup = True

    def _fade_win(self, n: int) -> np.ndarray:
        # 余弦淡入：前几毫秒接近 0，压住咔嗒，不剪后面的字
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        return 0.5 - 0.5 * np.cos(np.pi * t)

    def _fade_in(self, x: np.ndarray, n: int) -> np.ndarray:
        n = min(n, x.size)
        if n < 8:
            return x
        y = x.copy()
        y[:n] *= self._fade_win(n)
        return y

    def _skip_leading_click(self, x: np.ndarray, fade_n: int) -> np.ndarray:
        """只处理开头：先去掉前置静音，再在随后 80ms 内丢掉脉冲+静音。不往后搜，避免吃字。"""
        frame = max(8, int(self.sr * 0.005))
        if x.size < frame * 8:
            return self._fade_in(x, fade_n)
        # 最多跳过 200ms 前置静音
        n0 = min(x.size // frame, 40)
        rms0 = np.sqrt(np.mean(x[: n0 * frame].reshape(n0, frame) ** 2, axis=1))
        start = 0
        for i, e in enumerate(rms0):
            if e >= 0.012:
                start = i * frame
                break
        else:
            start = min(n0 * frame, max(0, x.size - 32))
        x = x[start:]
        n = min(x.size // frame, 16)
        if n < 6:
            return self._fade_in(x, fade_n)
        rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
        first = next((i for i, e in enumerate(rms) if e >= 0.016), 0)
        quiet_run = 0
        for i in range(first, n):
            if rms[i] < 0.010:
                quiet_run += 1
                if quiet_run >= 3:
                    j = i + 1
                    while j < n and rms[j] < 0.015:
                        j += 1
                    cut = j * frame if j < n else (i - quiet_run + 1) * frame
                    if 0 < cut < x.size - 32:
                        x = x[cut:]
                    return self._fade_in(x, fade_n)
            else:
                quiet_run = 0
        cut = first * frame + int(self.sr * 0.016)
        if 0 < cut < x.size - 32:
            x = x[cut:]
        return self._fade_in(x, fade_n)

    def _prep_start(self, x: np.ndarray) -> np.ndarray:
        return self._skip_leading_click(x, self.fade_n)

    def _fade_after_pauses(self, x: np.ndarray) -> np.ndarray:
        frame = max(8, int(self.sr * 0.01))
        n = x.size // frame
        if n < 10:
            return x
        rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
        onsets = []
        quiet = 0
        for i, e in enumerate(rms):
            if e < 0.008:
                quiet += 1
                continue
            if quiet >= 4 and (i - quiet) * frame > int(self.sr * 0.05):
                onsets.append(i * frame)
            quiet = 0
        y = x
        for onset in reversed(onsets):
            y = np.concatenate([y[:onset], self._skip_leading_click(y[onset:], self.pause_fade)])
        return y

    def feed(self, x: np.ndarray):
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).reshape(-1)
        if x.size < 16:
            return
        if self.startup:
            x = self._prep_start(x)
            self.startup = False
            if x.size < 16:
                return
        else:
            x = self._fade_after_pauses(x)
        if self.held is not None:
            n = min(self.step_n, self.held.size, x.size)
            if abs(float(self.held[-1] - x[0])) > 0.06:
                ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                mixed = self.held[:n] * np.sqrt(1.0 - ramp) + x[:n] * np.sqrt(ramp)
                x = np.concatenate([mixed, x[n:]])
            else:
                x = np.concatenate([self.held, x])
            self.held = None
        if x.size <= self.step_n:
            self.held = x.copy()
            return
        yield x[:-self.step_n]
        self.held = x[-self.step_n:].copy()

    def flush(self):
        if self.held is None or self.held.size == 0:
            self.held = None
            return
        y = self.held
        self.held = None
        yield y


_smoother = _ChunkSmoother()


def _prompt_text() -> str:
    # v3 prompt 格式：[可选指令]<|endofprompt|>参考音频文字稿
    base = config.PROMPT_TEXT
    ins = config.TTS_INSTRUCT_V3
    return f"{ins}<|endofprompt|>{base}" if ins else f"<|endofprompt|>{base}"


def load_model():
    global _tts
    if _tts is not None:
        return _tts
    sys.path.append(config.COSYVOICE_REPO)
    sys.path.append(config.COSYVOICE_REPO + "/third_party/Matcha-TTS")
    from cosyvoice.cli.cosyvoice import CosyVoice3

    if config.TTS_LOAD_VLLM:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
    _tts = CosyVoice3(config.COSYVOICE3_MODEL, load_trt=config.TTS_LOAD_TRT,
                      load_vllm=config.TTS_LOAD_VLLM, fp16=config.TTS_V3_FP16)
    _tts.add_zero_shot_spk(_prompt_text(), config.PROMPT_WAV, SPK_ID)
    torch.cuda.empty_cache = lambda: None
    _start_worker()
    submit("你好，我是小云。", lambda: True, lambda b: None, lambda: None)
    return _tts


def sample_rate() -> int:
    return _tts.sample_rate if _tts else config.TTS_SAMPLE_RATE


def submit(text: str, is_alive, on_chunk, on_done):
    """is_alive(): False 表示该任务已过期（被打断），chunk 丢弃。
    on_chunk(pcm16_bytes)；on_done()。均在 worker 线程触发。"""
    _job_q.put((text, is_alive, on_chunk, on_done))


def _start_worker():
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    threading.Thread(target=_worker, daemon=True).start()


def cancel_pending():
    """丢掉队列里还没开始的任务，并补调 on_done，避免打断后 worker 仍在合成旧句。"""
    while True:
        try:
            _text, _alive, _on_chunk, on_done = _job_q.get_nowait()
        except queue.Empty:
            break
        try:
            on_done()
        except Exception:
            pass


def _infer(text: str):
    # token_hop_len 在 tts() 内会自增且不重置，每次任务前复位
    if hasattr(_tts.model, "token_hop_len"):
        _tts.model.token_hop_len = config.TTS_FIRST_HOP_V3
        _tts.model.token_max_hop_len = 4 * config.TTS_FIRST_HOP_V3
    return _tts.inference_zero_shot(text, _prompt_text(), config.PROMPT_WAV,
                                    zero_shot_spk_id=SPK_ID, stream=True)


def _worker():
    while True:
        text, is_alive, on_chunk, on_done = _job_q.get()
        try:
            if not is_alive():
                _smoother.reset()
                continue
            _smoother.begin_stream()
            interrupted = False
            for out in _infer(text):
                if not is_alive():
                    interrupted = True
                    _smoother.reset()
                    break
                speech = out["tts_speech"]
                pcm_f32 = speech.detach().float().cpu().numpy().reshape(-1)
                for piece in _smoother.feed(pcm_f32):
                    on_chunk(_to_pcm16(piece))
            if not interrupted and is_alive():
                for piece in _smoother.flush():
                    on_chunk(_to_pcm16(piece))
        except Exception as e:  # 单个任务失败不拖垮 worker
            import traceback
            traceback.print_exc()
            print(f"[TTSv3] synth error: {e}", flush=True)
            _smoother.reset()
        finally:
            try:
                on_done()
            except Exception:
                pass
