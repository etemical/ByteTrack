# 非酒店业务不插话：不依赖某个具体话题词
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

from server.gate import is_ignore_reply, should_ignore_utt

LONG_OFFTOPIC = (
    "你等一下我先跟他说完，作业写完了没有，赶紧吃饭，今天先别玩手机了。"
)

CASES = [
    ("喂。", True),
    ("喂", True),
    (LONG_OFFTOPIC, True),
    ("这边跟你确认一下你近期有没有遇到类似情况呢", True),
    ("住3晚", False),
    ("住两晚", False),
    ("好的", False),
    ("对", False),
    ("大床房", False),
    ("帮我查一下明天还有大床房吗", False),
    ("你好，请问早餐是几点", False),
    ("在吗", False),
    ("我叫张伟", False),
    ("身份证号是110101", False),
]

fails = 0
for text, want in CASES:
    got = should_ignore_utt(text)
    ok = got == want
    if not ok:
        fails += 1
    tag = "OK " if ok else "FAIL"
    print(f"{tag} ignore={got} want={want}  {text[:40]!r}")

assert is_ignore_reply("[IGNORE]")
assert is_ignore_reply("[IGNORE]\n别插话")
assert not is_ignore_reply("好的，帮您办理入住")
print("FAILS:", fails)
sys.exit(1 if fails else 0)
