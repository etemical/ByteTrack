"""Silero VAD：32ms 帧状态机，回调 on_speech_start / on_speech_end。"""
import numpy as np
import torch
from silero_vad import load_silero_vad

from . import config

_FRAME = 512  # 32ms @16k


class VadDetector:
    def __init__(self, on_speech_start, on_speech_end):
        self._model = load_silero_vad(onnx=True)
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self._buf = np.zeros(0, dtype=np.float32)
        self._pre_roll = np.zeros(0, dtype=np.float32)
        self.speaking = False
        self._speech_run = 0
        self._silence_run = 0
        self._start_frames = max(1, int(config.VAD_MIN_SPEECH_MS / 32))
        self._end_frames = max(1, int(config.VAD_MIN_SILENCE_MS / 32))
        self._pad_len = int(config.ASR_SAMPLE_RATE * config.VAD_SPEECH_PAD_MS / 1000)

    def feed(self, pcm_f32: np.ndarray):
        """喂任意长度 float32[-1,1]；返回 [(is_speech_active, speech_audio_chunk)] 事件由回调抛出。"""
        self._buf = np.concatenate([self._buf, pcm_f32])
        while len(self._buf) >= _FRAME:
            frame = self._buf[:_FRAME]
            self._buf = self._buf[_FRAME:]
            self._process(frame)

    def _prob(self, frame: np.ndarray) -> float:
        with torch.no_grad():
            return float(self._model(torch.from_numpy(frame), config.ASR_SAMPLE_RATE).item())

    def _process(self, frame: np.ndarray):
        p = self._prob(frame)
        if not self.speaking:
            # 维护预滚动缓冲
            self._pre_roll = np.concatenate([self._pre_roll, frame])[-self._pad_len:]
            if p >= config.VAD_THRESHOLD:
                self._speech_run += 1
                if self._speech_run >= self._start_frames:
                    self.speaking = True
                    self._silence_run = 0
                    self._speech_run = 0
                    if self.on_speech_start:
                        self.on_speech_start(self._pre_roll.copy())
                    self._pre_roll = np.zeros(0, dtype=np.float32)
            else:
                self._speech_run = 0
        else:
            if p < config.VAD_NEG_THRESHOLD:
                self._silence_run += 1
                if self._silence_run >= self._end_frames:
                    self.speaking = False
                    self._silence_run = 0
                    if self.on_speech_end:
                        self.on_speech_end()
            else:
                self._silence_run = 0

    def reset(self):
        """丢掉当前说话状态，不触发 on_speech_end。"""
        self.speaking = False
        self._speech_run = 0
        self._silence_run = 0
        self._buf = np.zeros(0, dtype=np.float32)
        self._pre_roll = np.zeros(0, dtype=np.float32)
