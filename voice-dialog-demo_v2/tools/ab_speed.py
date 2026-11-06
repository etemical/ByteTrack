"""A/B 实验：同一文本在不同 instruct 下的合成时长对比（需先停服务，独占 GPU）。"""
import sys
import time

sys.path.insert(0, "/root/autodl-tmp/voice-dialog-demo_v2")

import torch

from server import tts_v3, config

TEXT = "早餐是早上六点半到九点半哦，您直接去二楼餐厅就行，赶不及的话跟我们说一声，帮您打包。"

tts_v3.load_model()

for name, instruct in [("empty", ""), ("fast", "用稍快的语速说，干脆利落"),
                       ("faster", "语速快，短促有力"), ("normal", "用正常自然的语速说")]:
    config.TTS_INSTRUCT_V3 = instruct
    spk = "ab_" + name
    tts_v3._tts.add_zero_shot_spk(tts_v3._prompt_text(), config.PROMPT_WAV, spk)
    t0 = time.perf_counter()
    n = 0
    for out in tts_v3._tts.inference_zero_shot(
            TEXT, tts_v3._prompt_text(), config.PROMPT_WAV,
            zero_shot_spk_id=spk, stream=True):
        n += out["tts_speech"].shape[1]
    dur = n / tts_v3._tts.sample_rate
    print(f"[{name:7s}] instruct={instruct!r:30s} 音频时长 {dur:.2f}s  合成耗时 {time.perf_counter()-t0:.1f}s", flush=True)
