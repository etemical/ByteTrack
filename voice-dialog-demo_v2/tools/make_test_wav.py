"""生成测试用 16k 语音 wav（用已部署的 CosyVoice 合成，闭环自测）。

用法（远程）：python tools/make_test_wav.py
输出：/root/autodl-tmp/voice-dialog-demo_v2/testdata/q1.wav, q2.wav
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from server import config, tts

TEXTS = {
    "q1": "你好，请问早餐是几点到几点？",
    "q2": "帮我查一下明天还有大床房吗？",
}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testdata")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tts.load_model()
    sr = tts.sample_rate()
    for name, text in TEXTS.items():
        chunks = []
        done = []
        tts.submit(text, lambda: True, chunks.append, lambda: done.append(1))
        while not done:
            import time
            time.sleep(0.05)
        audio = np.concatenate(
            [np.frombuffer(c, dtype=np.int16).astype(np.float32) / 32768 for c in chunks])
        audio16k = resample_poly(audio, 16000, sr).astype(np.float32)
        path = os.path.join(OUT_DIR, f"{name}.wav")
        sf.write(path, audio16k, 16000, subtype="PCM_16")
        print("wrote", path, f"{len(audio16k)/16000:.1f}s")


if __name__ == "__main__":
    main()
