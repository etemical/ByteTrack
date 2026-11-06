# 离线验证回声过滤规则：用用户 17:29 真实自循环语料 + 回归语料
import re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

src = open("server/pipeline.py", encoding="utf-8").read()
block = src[src.index("_SENT_RE"):src.index("class Session")]
ns = {"re": re}
from difflib import SequenceMatcher
ns["SequenceMatcher"] = SequenceMatcher
exec(block, ns)
_norm_echo, _fuzzy_in, _confuse_match = ns["_norm_echo"], ns["_fuzzy_in"], ns["_confuse_match"]

REPLY2 = '哈哈，小明您真逗！咱酒店有标准间、大床房和豪华套房，都挺宽敞舒服的。您说的"二套的房"是套房吧？'
REPLY3 = '哈哈，您这语音转得有点乱哦。我猜您是问房型吧？我们有大床房、双床房和套房，您想住哪种？我给您介绍下哈！'
REPLY_Q1 = '早餐是早上6点半到10点哦，在一楼餐厅。您明儿个想吃啥，直接去就行啦。'
REPLY_Q2 = '我帮您看看哈，明天大床房还有呢，您要住几晚呀？'


def is_echo(text, bot_texts, loose):
    """复刻 _is_echo_text 主循环（手工同步，防本地类定义依赖 Session）"""
    from difflib import SequenceMatcher
    t = _norm_echo(text)
    if not t:
        return False
    for raw in bot_texts:
        bot = _norm_echo(raw)
        if not bot:
            continue
        if loose and len(t) >= 2 and t in bot:
            return True
        if loose and _fuzzy_in(t, bot):
            return True
        if _confuse_match(t, bot):
            return True
        if len(t) >= 4 and (t in bot or bot in t):
            return True
        tail = bot[-12:]
        if len(t) <= 6:
            if t in tail:
                return True
            if len(t) >= 3 and SequenceMatcher(None, t, tail).ratio() >= 0.5:
                return True
            if len(t) >= 4 and sum(1 for c in t if c in tail) >= len(t) - 1:
                return True
    return False


# (文本, 近期播报, loose下应判回声?, 非loose下应判回声?)
CASES = [
    # —— 用户 17:29 真实自循环语料（必须拦）——
    ("小明，你。", [REPLY2], True, True),           # 您→你 漂移 3 字（本次漏网#1）
    ("小明你", [REPLY2], True, True),               # 同上 partial 形态
    ("转的有点乱哦，我猜你是问。", [REPLY3], True, False),  # 的→得+您→你 漂移（漏网#2）：loose 拦（实际流程走打断宽限窗）；strict 不 fuzzy 是设计
    ("转的", [REPLY3], True, True),                # 2 字碎片：易混字规则双模式拦
    ("哈哈，您这语音转的有点乱哦", [REPLY3], True, True),  # 长 partial 1 字漂移
    ("早餐在一楼餐厅6点到10点哦", ["早餐在一楼餐厅，6点到10点哦。"], True, True),  # 数字折叠回归
    ("探索附电的前台机器人", ["我是云端探索指定店的前台机器人小云"], True, False),  # 上轮重度漂移：非loose仍靠DTD
    # —— 用户 18:0x 真实自循环（长回声 final 迟到落入 IDLE/LISTENING）——
    # 运行时由 _utt_loose（起话时刻在播报中）保证走 loose；strict 逃逸可接受
    ("我是小云呀，云端探索酒店的前台机器人，有啥事，尽快说，入住顶早山问路都能帮你搞定哦。",
     ["我是小云呀，云端探索酒店的前台机器人，有啥事尽管说，入住订餐问路都能帮你搞定哦"],
     True, False),
    ("看到前台台有啥需要帮忙的？",
     ["您好呀，看到您来前台了，有啥需要帮忙的尽管说"],
     True, False),
    # —— 真人语料（必须放）——
    ("帮我", [REPLY_Q2], False, False),             # barge 首 partial
    ("帮我查一下，明天还有大床房吗？", [REPLY_Q1], False, False),  # barge 完整句
    ("住2晚", [REPLY_Q2], False, False),            # 3 字 0.67 陷阱（2/几不在易混表）
    ("你好，请问早餐是几点到几点？", [REPLY_Q2], False, False),
    ("在吗", [REPLY_Q1], False, False),             # 2 字高频插话
    ("行", [REPLY_Q1], True, True),                 # 1 字：既有句尾子串规则（bot 尾含"行"），非本次改动
]

fails = 0
for text, bots, want_loose, want_strict in CASES:
    got_loose = is_echo(text, bots, True)
    got_strict = is_echo(text, bots, False)
    ok = got_loose == want_loose and got_strict == want_strict
    if not ok:
        fails += 1
    tag = "OK " if ok else "FAIL"
    print(f"{tag} loose={got_loose}(want {want_loose}) strict={got_strict}(want {want_strict})  {text!r}")

# 已接受的权衡：loose 下乱序重复播报话题词会被误杀（仅限播报中/宽限窗）
tradeoff = is_echo("明天还有大床房吗", [REPLY_Q2], True)
print(f"INFO loose 乱序追问误杀(已知权衡)={tradeoff}  strict={is_echo('明天还有大床房吗', [REPLY_Q2], False)}(须False)")
print("FAILS:", fails)
