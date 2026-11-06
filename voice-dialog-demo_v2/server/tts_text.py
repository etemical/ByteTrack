"""送 TTS 前把号码类数字改成逐位读法，避免 CosyVoice 念成十一万八千九百。"""
import re

# 电话里 1 读「幺」，其余按位
_SPELL = str.maketrans("0123456789", "零幺二三四五六七八九")
_FW = str.maketrans("０１２３４５６７８９－—", "0123456789--")

# 这些是数量/时间，仍交给前端读成四百八十八、十一点、一百元
# 「488一晚」中间有「一」，所以晚/天/元/块 前面允许可选的「一」
_KEEP_UNIT = (
    r"(?:点|分|秒|年|月|日|岁|毛|角|折|层|"
    r"(?:一)?(?:晚|天|元|块)|"
    r"%|％|人|次|倍|公斤|千克|斤|米|公里|度|℃|起)"
)

_CODE_SEP = re.compile(r"(?:\+?86[\s\-]*)?(?:\d[\s\-]*){4,}\d")
# 数字和单位之间可能有空格：388 一晚、100 元
_DIGIT_RUN = re.compile(rf"(?<!\d)(\d{{3,}})(?!\s*{_KEEP_UNIT})")


def _spell_digits(s: str) -> str:
    return s.translate(_SPELL)


def for_tts(text: str) -> str:
    """阿拉伯数字号码 → 逐位汉字；屏幕上的 LLM 原文不要走这里。"""
    if not text:
        return text
    text = text.translate(_FW)

    def sep_repl(m: re.Match) -> str:
        raw = re.sub(r"[+\s\-]", "", m.group(0))
        if raw.startswith("86") and len(raw) >= 13:
            raw = raw[2:]
        if raw.isdigit() and len(raw) >= 5:
            return _spell_digits(raw)
        return m.group(0)

    text = _CODE_SEP.sub(sep_repl, text)
    text = _DIGIT_RUN.sub(lambda m: _spell_digits(m.group(1)), text)
    return text
