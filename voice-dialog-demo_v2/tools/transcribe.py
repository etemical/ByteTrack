"""离线校验：用 ASR 回听机器人音频，确认播报内容。远程用法：python tools/transcribe.py bot.wav"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def main(path):
    from server import asr

    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    asr.load_model()
    finals = []
    sess = asr.AsrSession(lambda t: None, lambda t: finals.append(t))
    for i in range(0, len(audio), 9600):
        sess.feed(audio[i:i + 9600])
    sess.finish()
    while not finals:
        time.sleep(0.1)
    print("[transcribe]", finals[0])


if __name__ == "__main__":
    main(sys.argv[1])
