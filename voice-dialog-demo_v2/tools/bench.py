"""分阶段基准：TTS 首块/分块时间线、LLM TTFT、测试 wav 能量分布。

远程用法：
  python tools/bench.py tts   # TTS 合成时间线
  python tools/bench.py llm   # DeepSeek TTFT
  python tools/bench.py wav testdata/q1.wav   # wav 每秒 RMS
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bench_tts(text="你好，请问早餐是几点到几点？", rounds="3"):
    import numpy as np
    from server import config
    if config.TTS_ENGINE == "cosyvoice3":
        from server import tts_v3 as tts
    else:
        from server import tts

    tts.load_model()
    print(f"[bench] model ready, sample_rate={tts.sample_rate()}")
    for r in range(int(rounds)):
        chunks = []
        done = []
        t0 = time.perf_counter()
        marks = []

        def on_chunk(b):
            chunks.append(b)
            marks.append((time.perf_counter() - t0, len(b)))

        tts.submit(text, lambda: True, on_chunk, lambda: done.append(1))
        while not done:
            time.sleep(0.02)
        total_audio = sum(len(b) for b in chunks) / 2 / tts.sample_rate()
        print(f"[bench] round{r} text={text!r} chunks={len(chunks)} audio={total_audio:.2f}s")
        for t, n in marks[:6]:
            print(f"  chunk @{t*1000:7.0f}ms  {n/2/tts.sample_rate()*1000:.0f}ms audio")
        print(f"[bench] round{r} first_chunk={marks[0][0]*1000:.0f}ms  total={marks[-1][0]*1000:.0f}ms")
        if r == int(rounds) - 1:
            import soundfile as sf
            audio = np.concatenate([np.frombuffer(b, dtype=np.int16) for b in chunks])
            sf.write("/tmp/bench_tts.wav", audio, tts.sample_rate())
            print("[bench] saved /tmp/bench_tts.wav")


def bench_llm():
    import asyncio
    from server import llm

    async def go():
        t0 = time.perf_counter()
        first = []

        async def noop(d):
            pass

        full = await llm.stream_chat(
            [{"role": "user", "content": "你好，一句话介绍你自己"}],
            noop, lambda ts: first.append(ts - t0),
        )
        print(f"[bench] llm ttft={first[0]*1000:.0f}ms  full={full!r}")

    asyncio.run(go())


def bench_wav(path):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    print(f"[bench] {path} {len(audio)/sr:.1f}s")
    for i in range(0, len(audio), sr):
        seg = audio[i : i + sr]
        print(f"  {i//sr:3d}s rms={np.sqrt((seg**2).mean()):.4f}")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "tts":
        bench_tts(*sys.argv[2:])
    elif what == "llm":
        bench_llm()
    elif what == "wav":
        bench_wav(sys.argv[2])
