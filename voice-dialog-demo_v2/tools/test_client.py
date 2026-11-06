"""无头测试客户端：把 wav 当麦克风实时流送进 /ws，打印事件与延迟指标。

用法（远程）：
  python tools/test_client.py testdata/q1.wav                 # 单轮问答
  python tools/test_client.py testdata/q1.wav --barge q2.wav  # 播报中途插入第二句测试打断
"""
import argparse
import asyncio
import json
import sys
import time

import numpy as np
import soundfile as sf
import websockets

FRAME = 1024  # 64ms @16k


async def stream_wav(ws, path, realtime=True, max_ms=None):
    audio, sr = sf.read(path, dtype="int16")
    assert sr == 16000, f"{path} must be 16kHz"
    sent_ms = 0
    for i in range(0, len(audio), FRAME):
        chunk = audio[i : i + FRAME]
        if len(chunk) < FRAME:
            chunk = np.pad(chunk, (0, FRAME - len(chunk)))
        await ws.send(chunk.tobytes())
        sent_ms += 64
        if max_ms and sent_ms >= max_ms:
            return False
        if realtime:
            await asyncio.sleep(0.064)
    return True


async def run(path, barge=None, server="ws://127.0.0.1:8000/ws", save=None):
    async with websockets.connect(server, max_size=None) as ws:
        events = asyncio.Queue()
        bot_pcm = bytearray()

        async def recv():
            async for msg in ws:
                if isinstance(msg, bytes):
                    await events.put(("audio", msg))
                else:
                    await events.put(("json", json.loads(msg)))

        rt = asyncio.create_task(recv())
        t0 = time.perf_counter()
        audio_bytes = 0
        bot_speaking = False
        barge_req = asyncio.Event()   # 主循环请求插话
        barge_done = False
        interrupted = False
        stop_tx = asyncio.Event()
        bot_end_count = 0

        async def send_task():
            # 单一发送协程：插话时暂停静音流，避免语音/静音帧交替导致 VAD 无法起判
            await stream_wav(ws, path)
            silence = bytes(FRAME * 2)
            while not stop_tx.is_set():
                if barge and barge_req.is_set():
                    barge_req.clear()
                    await stream_wav(ws, barge)
                    continue
                await ws.send(silence)
                await asyncio.sleep(0.064)

        st = asyncio.create_task(send_task())
        st.add_done_callback(
            lambda t: None if t.cancelled() else t.exception() and print(f"[send_task 异常] {t.exception()}"))
        last_audio_log = 0
        while True:
            try:
                kind, payload = await asyncio.wait_for(events.get(), timeout=30)
            except asyncio.TimeoutError:
                print("[timeout] 30s 无事件，退出")
                break
            if kind == "audio":
                audio_bytes += len(payload)
                bot_pcm += payload
                bot_speaking = True
                if barge and not barge_done and audio_bytes > 24000 * 2 * 1:  # 播报 ~1s 后插话
                    barge_done = True
                    print(f"[barge] t={time.perf_counter()-t0:.2f}s 插入第二句: {barge}")
                    barge_req.set()
                continue
            m = payload
            t = time.perf_counter() - t0
            if m["type"] == "asr_partial":
                print(f"[{t:6.2f}] ASR~ {m['text']}")
            elif m["type"] == "asr_final":
                print(f"[{t:6.2f}] ASR! {m['text']}")
            elif m["type"] == "llm_delta":
                print(m["text"], end="", flush=True)
            elif m["type"] == "llm_done":
                print(f"\n[{t:6.2f}] LLM done")
            elif m["type"] == "interrupted":
                interrupted = True
                bot_speaking = False
                print(f"\n[{t:6.2f}] *** interrupted ***")
            elif m["type"] == "metrics":
                print(f"[{t:6.2f}] METRICS {json.dumps({k:v for k,v in m.items() if k!='type'})}")
            elif m["type"] == "bot_end":
                bot_end_count += 1
                print(f"[{t:6.2f}] bot_end, 音频 {audio_bytes/48000:.1f}s")
                if not barge:
                    break
                if interrupted:  # 打断后的新回复播完即结束
                    break
            elif m["type"] == "state":
                print(f"[{t:6.2f}] state={m['value']}")
                if m["value"] == "IDLE":
                    bot_speaking = False
            elif m["type"] == "error":
                print(f"[{t:6.2f}] ERROR@{m.get('where')}: {m.get('msg')}")

        st.cancel()
        rt.cancel()
        if save and bot_pcm:
            audio = np.frombuffer(bytes(bot_pcm), dtype=np.int16)
            sf.write(save, audio, 24000)
            print(f"[saved] bot audio -> {save} ({len(audio)/24000:.1f}s)")
        print(f"[done] total audio {audio_bytes/48000:.1f}s, interrupted={interrupted}")
        return interrupted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--barge", default=None)
    ap.add_argument("--server", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--save", default=None, help="保存机器人音频到 wav（24kHz）")
    args = ap.parse_args()
    interrupted = asyncio.run(run(args.wav, args.barge, args.server, args.save))
    if args.barge:
        print("BARGE-IN TEST:", "PASS" if interrupted else "FAIL(未触发打断)")
        sys.exit(0 if interrupted else 1)


if __name__ == "__main__":
    main()
