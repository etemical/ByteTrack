"""CosyVoice 流式 TTS。全局单 worker 线程串行消费合成任务（GPU 单卡，避免并发争抢）。"""
import queue
import sys
import threading
import numpy as np
import torch

from . import config

_tts = None
_job_q: queue.Queue = queue.Queue()
_worker_started = False


SPK_ID = "xiaojin"


def load_model():
    global _tts
    if _tts is not None:
        return _tts
    sys.path.append(config.COSYVOICE_REPO)
    sys.path.append(config.COSYVOICE_REPO + "/third_party/Matcha-TTS")
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

    if config.TTS_MODE == "zero_shot":
        if config.TTS_LOAD_VLLM:
            # vllm 不识别导出配置里的 CosyVoice2ForCausalLM，需先注册
            from vllm import ModelRegistry
            from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
            ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
        _tts = CosyVoice2(config.COSYVOICE_MODEL, load_jit=config.TTS_LOAD_JIT,
                          load_trt=config.TTS_LOAD_TRT, load_vllm=config.TTS_LOAD_VLLM,
                          fp16=config.TTS_FP16)
        # 预注册提示音：免除每次调用的提示音频特征提取
        _tts.add_zero_shot_spk(config.PROMPT_TEXT, config.PROMPT_WAV, SPK_ID)
        if config.TTS_INSTRUCT:
            # instruct2 复用 zero_shot 缓存时，prompt_text 槽位必须是指令文本，
            # 否则 LLM 会把 PROMPT_TEXT 当待合成文本念出来
            _tts.add_zero_shot_spk(config.TTS_INSTRUCT, config.PROMPT_WAV, SPK_ID + "_instruct")
    else:
        _tts = CosyVoice(config.COSYVOICE_MODEL, load_jit=config.TTS_LOAD_JIT,
                         load_trt=False, fp16=config.TTS_FP16)
    # CosyVoice2 每次推理结束 torch.cuda.empty_cache()，同步+释放导致下次分配变慢，屏蔽之
    torch.cuda.empty_cache = lambda: None
    _start_worker()
    # 预热（CUDA kernel / 图编译）
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
    # token_hop_len 在 tts() 内会自增且不重置，每次任务前复位小首块
    if hasattr(_tts.model, "token_hop_len"):
        _tts.model.token_hop_len = config.TTS_FIRST_HOP
        _tts.model.token_max_hop_len = 50
    if config.TTS_MODE == "zero_shot":
        if config.TTS_INSTRUCT:
            return _tts.inference_instruct2(text, config.TTS_INSTRUCT, config.PROMPT_WAV,
                                            zero_shot_spk_id=SPK_ID + "_instruct", stream=True)
        return _tts.inference_zero_shot(text, config.PROMPT_TEXT, config.PROMPT_WAV,
                                        zero_shot_spk_id=SPK_ID, stream=True)
    return _tts.inference_sft(text, config.TTS_SPK, stream=True)


def _worker():
    while True:
        text, is_alive, on_chunk, on_done = _job_q.get()
        try:
            if not is_alive():
                continue
            for out in _infer(text):
                if not is_alive():
                    break
                speech = out["tts_speech"]
                pcm = (speech * (2 ** 15)).clamp(-(2 ** 15), 2 ** 15 - 1)
                pcm = pcm.to(torch.int16).cpu().numpy().tobytes()
                on_chunk(pcm)
        except Exception as e:  # 单个任务失败不拖垮 worker
            import traceback
            traceback.print_exc()
            print(f"[TTS] synth error: {e}", flush=True)
        finally:
            try:
                on_done()
            except Exception:
                pass
