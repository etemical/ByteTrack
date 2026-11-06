"""会话编排：VAD → 流式 ASR → DeepSeek 流式 → CosyVoice 流式，支持打断。

状态机：IDLE → LISTENING → THINKING → SPEAKING → IDLE

采集/打断：麦克风一直上送。播报中前端一听到人声就停播；服务端立刻开 ASR，
用最近约 260ms 环形缓冲保住句首，之后只收人声。VAD 说完再 finish，不重开会话。

回声：浏览器 AEC + 300ms 起话确认 + confirm。SenseVoice 句末校正比 Paraformer
更能听出回声，因此只对「短句尾 / 整句回声」做轻量文本过滤，避免自循环。
"""
import asyncio
import os
import re
import time
from collections import deque
from difflib import SequenceMatcher

import numpy as np

from . import asr, config, llm, vad
from .gate import IGNORE_MARK, is_ignore_reply, should_ignore_utt
from .tts_text import for_tts

if config.TTS_ENGINE == "cosyvoice3":
    from . import tts_v3 as tts
else:
    from . import tts

_SENT_RE = re.compile(r"[。！？!?；;\n]")
_CLAUSE_RE = re.compile(r"[，,：:、~—]")
_TTS_STRIP_RE = re.compile(r"[*#`>\[\]()（）]")
_ECHO_STRIP_RE = re.compile(r"[^\w一-鿿]+")
_NUM_FOLD = str.maketrans(
    {"零": "0", "〇": "0", "一": "1", "二": "2", "两": "2", "三": "3",
     "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}
)


def _norm_echo(s: str) -> str:
    return _ECHO_STRIP_RE.sub("", s or "").replace("儿", "").translate(_NUM_FOLD)


class Session:
    def __init__(self, ws):
        self.ws = ws
        self.loop = asyncio.get_running_loop()
        self.vad = vad.VadDetector(self._on_vad_start, self._on_vad_end)
        self.asr_sess: asr.AsrSession | None = None
        self.state = "IDLE"
        self.gen = 0
        self.reply_task: asyncio.Task | None = None
        self.history: list[dict] = []
        self._send_lock = asyncio.Lock()
        self._interrupt_deadline: float | None = None
        self._tts_pending = 0
        self._llm_done = False
        self._ts: dict[str, float] = {}
        self._dbg: list[np.ndarray] | None = [] if config.DEBUG_AUDIO else None
        self._bot_texts: deque[str] = deque(maxlen=3)
        self._echo_tail_until = 0.0
        self._audio_secs = 0.0
        self._first_audio_at: float | None = None
        self._idle_task: asyncio.Task | None = None
        self._keep_utt = False  # 本句已确认打断，final 不要当回声丢掉
        self._barge_at: float | None = None  # 扬声器静音时刻
        self._ring: deque[np.ndarray] = deque()
        self._ring_n = 0
        self._bot_ui_gen = 0

    async def send_json(self, obj: dict):
        async with self._send_lock:
            await self.ws.send_json(obj)

    async def send_bytes(self, b: bytes, gen: int | None = None):
        async with self._send_lock:
            if gen is not None and not self._alive(gen):
                return
            await self.ws.send_bytes(b)

    async def _set_state(self, s: str):
        self.state = s
        await self.send_json({"type": "state", "value": s})

    def feed_audio(self, pcm: np.ndarray):
        f32 = pcm.astype(np.float32) / 32768.0
        self._push_ring(f32)
        self.vad.feed(f32)
        # 打断后 ASR 可能在 VAD 确认前就开了，不能再等 speaking 才送。
        if self.asr_sess is not None:
            self.asr_sess.feed(f32)
            if self._dbg is not None:
                self._dbg.append(f32)

    def _push_ring(self, f32: np.ndarray):
        self._ring.append(f32)
        self._ring_n += len(f32)
        max_n = int(config.ASR_SAMPLE_RATE * 0.4)
        while self._ring_n > max_n and self._ring:
            old = self._ring.popleft()
            self._ring_n -= len(old)

    def _ring_tail(self, ms: float) -> np.ndarray:
        n = int(config.ASR_SAMPLE_RATE * ms / 1000)
        if not self._ring or n <= 0:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(list(self._ring))
        return audio[-n:] if len(audio) > n else audio

    def _listen_from_ring(self):
        """停播后立刻开识别，用环形缓冲保住句首，避免再等 VAD 确认。"""
        if self.asr_sess is not None:
            return
        self.asr_sess = asr.AsrSession(self._on_asr_partial, self._on_asr_final)
        if self._dbg is not None:
            self._dbg = []
        tail = self._ring_tail(260)
        if len(tail):
            self.asr_sess.feed(tail)
            if self._dbg is not None:
                self._dbg.append(tail.copy())
        print(f"[barge] asr start, ring {len(tail) / config.ASR_SAMPLE_RATE * 1000:.0f}ms", flush=True)

    def _dbg_dump(self, text: str):
        if self._dbg is None:
            return
        audio, self._dbg = np.concatenate(self._dbg), None
        if len(audio) < config.ASR_SAMPLE_RATE // 2:
            return
        import soundfile as sf
        os.makedirs(config.DEBUG_AUDIO_DIR, exist_ok=True)
        path = os.path.join(config.DEBUG_AUDIO_DIR, f"utt_{int(time.time())}.wav")
        sf.write(path, audio, config.ASR_SAMPLE_RATE)
        print(f"[dbg] ASR 输入音频 -> {path}  识别: {text}", flush=True)

    def _is_echo_text(self, text: str) -> bool:
        """轻量过滤：整句回声，或 2–5 字落在上一句句尾（咋事呀 / 主意呢）。
        不用宽松子串，避免把「双床房」「入住」当回声杀掉。"""
        t = _norm_echo(text)
        if not t:
            return False
        for raw in self._bot_texts:
            bot = _norm_echo(raw)
            if not bot:
                continue
            if len(t) >= 8 and (t in bot or bot in t):
                return True
            tail = bot[-10:]
            if 2 <= len(t) <= 5 and t in tail:
                return True
            if 3 <= len(t) <= 5 and SequenceMatcher(None, t, tail).ratio() >= 0.6:
                return True
        return False

    def _trim_barge_preroll(self, preroll: np.ndarray) -> np.ndarray:
        """打断起话：600ms 预滚前半段多半是 TTS，只留停播附近的人声。"""
        barging = self.state in ("SPEAKING", "THINKING") or self._barge_at is not None
        if not barging or preroll is None or len(preroll) == 0:
            return preroll
        sr = config.ASR_SAMPLE_RATE
        # 确认窗 300ms + 约一个字（220ms）。再短会偶发吞句首，再长就把 TTS 带回去。
        keep_s = (config.VAD_MIN_SPEECH_MS + 220) / 1000.0
        keep_n = int(keep_s * sr)
        if len(preroll) <= keep_n:
            return preroll
        print(f"[barge] trim preroll {len(preroll) / sr * 1000:.0f}ms -> {keep_s * 1000:.0f}ms", flush=True)
        return preroll[-keep_n:]

    def _on_vad_start(self, preroll: np.ndarray):
        if self.state in ("SPEAKING", "THINKING"):
            if config.INTERRUPT_MODE == "immediate":
                self.loop.create_task(self._interrupt())
            else:
                self._interrupt_deadline = time.perf_counter() + config.INTERRUPT_CONFIRM_MS / 1000
        if self.asr_sess is not None:
            # 打断时 ASR 已在听，不要再灌 600ms 带 TTS 的预滚。
            if self.state == "IDLE":
                self.loop.create_task(self._set_state("LISTENING"))
            return
        self.asr_sess = asr.AsrSession(self._on_asr_partial, self._on_asr_final)
        if self._dbg is not None:
            self._dbg = []
        preroll = self._trim_barge_preroll(preroll)
        if len(preroll):
            self.asr_sess.feed(preroll)
            if self._dbg is not None:
                self._dbg.append(preroll.copy())
        if self.state == "IDLE":
            self.loop.create_task(self._set_state("LISTENING"))

    def _on_vad_end(self):
        self._ts["vad_end"] = time.perf_counter()
        if self.asr_sess is not None:
            self.asr_sess.finish()

    def _on_asr_partial(self, text: str):
        self.loop.call_soon_threadsafe(self._handle_partial, text)

    def _on_asr_final(self, text: str):
        self.loop.call_soon_threadsafe(lambda: self.loop.create_task(self._handle_final(text)))

    def _handle_partial(self, text: str):
        if self._is_echo_text(text):
            return
        self.loop.create_task(self.send_json({"type": "asr_partial", "text": text}))
        if self.state in ("SPEAKING", "THINKING") and self._interrupt_deadline is not None:
            if time.perf_counter() > self._interrupt_deadline:
                self._interrupt_deadline = None
            elif text.strip():
                self._interrupt_deadline = None
                self.loop.create_task(self._interrupt())

    async def _handle_final(self, text: str):
        self._ts["asr_final"] = time.perf_counter()
        self._dbg_dump(text)
        sess, self.asr_sess = self.asr_sess, None
        if sess:
            sess.close()
        self._interrupt_deadline = None
        self._barge_at = None
        in_tail = (not self._keep_utt) and time.perf_counter() < self._echo_tail_until
        if (not self._keep_utt) and text.strip() and len(_norm_echo(text)) <= 2 and (
                in_tail or self.state in ("SPEAKING", "THINKING")):
            print(f"[echo] short drop {text!r}", flush=True)
            if self.state == "LISTENING":
                await self._set_state("IDLE")
            return
        if (not self._keep_utt) and self._is_echo_text(text):
            print(f"[echo] drop {text!r}", flush=True)
            if self.state == "LISTENING":
                await self._set_state("IDLE")
            return
        self._keep_utt = False
        await self.send_json({"type": "asr_final", "text": text})
        if not text:
            if self.state == "LISTENING":
                await self._set_state("IDLE")
            return
        if self.state in ("SPEAKING", "THINKING"):
            await self._interrupt()
        if should_ignore_utt(text):
            print(f"[gate] ignore {text!r}", flush=True)
            await self.send_json({"type": "ignored", "text": text})
            await self._set_state("IDLE")
            return
        await self._start_reply(text)

    async def _interrupt(self):
        if self.state not in ("SPEAKING", "THINKING"):
            return
        self.state = "LISTENING"  # 先占住，避免 barge 与 VAD 各发一次 interrupted
        self._barge_at = time.perf_counter()
        self.gen += 1
        self._tts_pending = 0
        self._keep_utt = True
        await self.send_json({"type": "interrupted"})
        tts.cancel_pending()
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
            self.reply_task = None
        # 只有 LLM 还没写完的半句才从历史里拿掉。已经说完的回复必须留着，
        # 否则下一轮模型会以为没介绍过、没答过 WiFi，又从头讲一遍。
        if (not self._llm_done) and self.history and self.history[-1]["role"] == "assistant":
            self.history.pop()
        await self._set_state("LISTENING")

    async def on_barge(self):
        """前端疑似人声：先静音听，等 VAD 确认再正式打断；敲键盘会在 watch 里恢复播报。"""
        if self.state not in ("SPEAKING", "THINKING"):
            return
        self._barge_at = time.perf_counter()
        self._listen_from_ring()
        self.loop.create_task(self._barge_asr_watch())

    async def _barge_asr_watch(self):
        """约 550ms 内 VAD 没确认，当作误触发（键盘/噪声），恢复播报。"""
        await asyncio.sleep(0.55)
        if self._barge_at is None:
            return
        if self.state == "LISTENING":
            return
        if self.vad.speaking:
            return
        print("[barge] cancel, no vad", flush=True)
        sess, self.asr_sess = self.asr_sess, None
        if sess:
            sess.close()
        self._barge_at = None
        self._keep_utt = False
        if self._dbg is not None:
            self._dbg = []
        await self.send_json({"type": "barge_cancel"})

    def on_play_end(self):
        """扬声器队列排空：合成早已结束时，这时才离开 SPEAKING。"""
        if self.state != "SPEAKING":
            return
        if self._llm_done and self._tts_pending <= 0:
            if self._idle_task and not self._idle_task.done():
                self._idle_task.cancel()
            self.loop.create_task(self._finish_speaking())

    async def _start_reply(self, user_text: str):
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
            self.reply_task = None
        self.gen += 1
        gen = self.gen
        self._llm_done = False
        self._tts_pending = 0
        # 被打断且模型还没写完时，历史里会连续两条 user。丢掉未回答的旧问，
        # 否则模型会把 WiFi、自我介绍和新问题拼成一段废话。
        while self.history and self.history[-1]["role"] == "user":
            dropped = self.history.pop()
            print(f"[hist] drop unanswered {dropped['content']!r}", flush=True)
        self.history.append({"role": "user", "content": user_text})
        await self.send_json({"type": "user_final", "text": user_text})
        await self._set_state("THINKING")
        self._bot_texts.append("")
        self._audio_secs = 0.0
        self._first_audio_at = None
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        self.reply_task = self.loop.create_task(self._reply(user_text, gen))

    def _alive(self, gen: int) -> bool:
        return gen == self.gen

    async def _reply(self, user_text: str, gen: int):
        buf = ""
        full = ""
        first_sent = False
        bot_started = False
        pending = ""

        def on_first_token(ts):
            self._ts["llm_first"] = ts

        async def ensure_bot_start():
            nonlocal bot_started
            if bot_started or not self._alive(gen):
                return
            bot_started = True
            self._bot_ui_gen = gen
            await self.send_json({"type": "bot_start", "gen": gen, "rate": tts.sample_rate()})

        async def on_delta(delta: str):
            nonlocal buf, full, first_sent, pending
            # 先攒几个字，避免 [IGNORE] 流到界面和 TTS
            if pending is not None:
                pending += delta
                if IGNORE_MARK.startswith(pending) and pending != IGNORE_MARK:
                    return
                if pending.startswith(IGNORE_MARK):
                    full = pending
                    pending = None
                    return
                delta, pending = pending, None
            full += delta
            if self._bot_texts:
                self._bot_texts[-1] = full
            await ensure_bot_start()
            await self.send_json({"type": "llm_delta", "text": delta})
            buf += delta
            buf = await self._flush_sentences(buf, gen, aggressive=not first_sent)
            if not first_sent and self._tts_pending > 0:
                first_sent = True

        async def drop_ignored():
            print(f"[gate] llm ignore {user_text!r}", flush=True)
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            if self._bot_texts:
                self._bot_texts.pop()
            self._llm_done = True
            await self.send_json({"type": "ignored", "text": user_text})
            await self._set_state("IDLE")

        try:
            full_text = await llm.stream_chat(
                [{"role": "system", "content": config.SYSTEM_PROMPT}] + self.history[-20:],
                on_delta, on_first_token,
            )
            if pending:
                full_text = (full_text or "") + pending
            if is_ignore_reply(full_text):
                await drop_ignored()
                return
            if not (full_text or "").strip():
                print("[LLM] empty, use fallback", flush=True)
                full_text = "嗯，您刚说的我没听太清，能再说一遍吗？"
                await on_delta(full_text)
            if buf.strip():
                await self._submit_tts(buf, gen)
                buf = ""
            self._llm_done = True
            self.history.append({"role": "assistant", "content": full_text})
            if self._bot_texts:
                self._bot_texts[-1] = full_text or ""
            await self.send_json({"type": "llm_done", "text": full_text})
            await self._maybe_finish(gen)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.send_json({"type": "error", "where": "llm", "msg": str(e)})
            self._llm_done = True
            await self._maybe_finish(gen)

    async def _flush_sentences(self, buf: str, gen: int, aggressive: bool) -> str:
        # 只把首句尽快送去合成。后面攒到 LLM 结束再一次合成，
        # 避免每个句号/逗号都新开 CosyVoice 流（句中爆破音的主因）。
        if not aggressive:
            return buf
        while True:
            m = _SENT_RE.search(buf)
            if m:
                await self._submit_tts(buf[: m.end()], gen)
                buf = buf[m.end():]
                continue
            c = _CLAUSE_RE.search(buf)
            if c and c.end() >= config.FIRST_SENT_MIN_CHARS:
                await self._submit_tts(buf[: c.end()], gen)
                buf = buf[c.end():]
                continue
            if len(buf) >= config.FIRST_SENT_MIN_CHARS * 3:
                await self._submit_tts(buf, gen)
                return ""
            return buf

    async def _submit_tts(self, text: str, gen: int):
        text = _TTS_STRIP_RE.sub("", text).strip()
        text = for_tts(text)
        if not text:
            return
        if "tts_submit" not in self._ts:
            self._ts["tts_submit"] = time.perf_counter()
        self._tts_pending += 1

        def is_alive():
            return self._alive(gen)

        def on_chunk(pcm: bytes):
            self.loop.call_soon_threadsafe(self._handle_tts_chunk, pcm, gen)

        def on_done():
            self.loop.call_soon_threadsafe(
                lambda: self.loop.create_task(self._tts_done(gen)))

        if self._bot_ui_gen != gen:
            self._bot_ui_gen = gen
            await self.send_json({"type": "bot_start", "gen": gen, "rate": tts.sample_rate()})
        tts.submit(text, is_alive, on_chunk, on_done)
        await self._set_state("SPEAKING")

    def _handle_tts_chunk(self, pcm: bytes, gen: int):
        if not self._alive(gen):
            return
        if self._first_audio_at is None:
            self._first_audio_at = time.perf_counter()
        self._audio_secs += len(pcm) / 2 / tts.sample_rate()
        if "first_audio" not in self._ts:
            self._ts["first_audio"] = time.perf_counter()
            m = {"type": "metrics"}
            if "vad_end" in self._ts:
                m["vad_end_to_asr_ms"] = round((self._ts["asr_final"] - self._ts["vad_end"]) * 1000)
                m["asr_to_llm_first_ms"] = round((self._ts["llm_first"] - self._ts["asr_final"]) * 1000)
                m["llm_first_to_audio_ms"] = round((self._ts["first_audio"] - self._ts["llm_first"]) * 1000)
                m["tts_first_chunk_ms"] = round((self._ts["first_audio"] - self._ts["tts_submit"]) * 1000)
                m["total_e2e_ms"] = round((self._ts["first_audio"] - self._ts["vad_end"]) * 1000)
            self.loop.create_task(self.send_json(m))
            print("[metrics]", m, flush=True)
        self.loop.create_task(self.send_bytes(pcm, gen))

    async def _tts_done(self, gen: int):
        if not self._alive(gen):
            return
        self._tts_pending -= 1
        await self._maybe_finish(gen)

    async def _maybe_finish(self, gen: int):
        if self._alive(gen) and self._llm_done and self._tts_pending <= 0:
            await self.send_json({"type": "bot_end", "gen": gen})
            self._ts = {}
            remaining = 0.0
            if self._first_audio_at is not None:
                played = time.perf_counter() - self._first_audio_at
                remaining = max(0.0, self._audio_secs - played)
            self._idle_task = self.loop.create_task(
                self._idle_after(gen, remaining + 0.3))

    async def _idle_after(self, gen: int, delay: float):
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if self._alive(gen) and self.state == "SPEAKING":
            await self._finish_speaking()

    async def _finish_speaking(self):
        if self.state != "SPEAKING":
            return
        self._echo_tail_until = time.perf_counter() + config.ECHO_TAIL_MS / 1000
        await self._set_state("IDLE")

    async def close(self):
        self.gen += 1
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        if self.asr_sess:
            self.asr_sess.close()
            self.asr_sess = None
