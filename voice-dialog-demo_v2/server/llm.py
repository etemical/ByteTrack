"""DeepSeek 流式 LLM。"""
import asyncio
import time
from openai import AsyncOpenAI

from . import config

_client: AsyncOpenAI | None = None
_LLM_TIMEOUT_S = 15.0


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            timeout=_LLM_TIMEOUT_S,
            max_retries=0,
        )
    return _client


async def _reset_client():
    global _client
    old, _client = _client, None
    if old is None:
        return
    try:
        await old.close()
    except Exception:
        pass


async def stream_chat(messages, on_delta, on_first_token=None):
    """流式对话。on_delta(delta_text) 逐个增量回调；返回完整文本。可被 asyncio.CancelledError 中断。"""
    client = get_client()
    kw = dict(
        model=config.DEEPSEEK_MODEL,
        messages=messages,
        stream=True,
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=config.LLM_TEMPERATURE,
        extra_body={"thinking": {"type": "disabled"}},
    )
    stream = None
    full = ""
    finish = None
    last = (messages[-1].get("content") or "")[:40] if messages else ""
    print(f"[LLM] start n={len(messages)} last={last!r}", flush=True)

    async def _run():
        nonlocal stream, full, finish
        try:
            stream = await client.chat.completions.create(**kw)
        except Exception as e:
            if "thinking" in str(e).lower() or "extra_body" in str(e).lower():
                kw.pop("extra_body", None)
                stream = await client.chat.completions.create(**kw)
            else:
                raise
        first = True
        async for chunk in stream:
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            if ch.finish_reason:
                finish = ch.finish_reason
            delta = ch.delta.content if ch.delta else None
            if not delta:
                continue
            if first:
                first = False
                if on_first_token:
                    on_first_token(time.perf_counter())
            full += delta
            await on_delta(delta)

    try:
        await asyncio.wait_for(_run(), timeout=_LLM_TIMEOUT_S)
    except asyncio.TimeoutError:
        print(f"[LLM] timeout {_LLM_TIMEOUT_S:.0f}s chars={len(full)} finish={finish}", flush=True)
        await _reset_client()
    except asyncio.CancelledError:
        await _reset_client()
        raise
    except Exception as e:
        print(f"[LLM] error {type(e).__name__}: {e}", flush=True)
        await _reset_client()
        raise
    finally:
        if stream is not None:
            try:
                await stream.aclose()
            except Exception:
                pass
    if not (full or "").strip():
        print(f"[LLM] empty content finish={finish}", flush=True)
    return full
