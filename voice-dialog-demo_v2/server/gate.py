"""非酒店业务不触发回复：ASR 可以上屏，但不插话。"""
import re

IGNORE_MARK = "[IGNORE]"

# 单独「喂」多半不是在答酒店问题
_PHONE_PICKUP = re.compile(r"^(喂)[啊呀哈]?[。！？.!?…\s]*$")

_HOTEL = re.compile(
    r"入住|续住|退房|早餐|wifi|WiFi|无线|停车|房间|房号|房型|"
    r"大床|双床|豪华|套房|前台|小云|身份证|办理|登记|发票|行李|"
    r"住\d|住[一两二三四五六七八九十几]|晚"
)

# 较长、又完全没有酒店业务词：旁白/闲聊/别人的话，不送模型以免插话
_OFFTOPIC_MIN = 16


def should_ignore_utt(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _PHONE_PICKUP.match(t):
        return True
    compact = re.sub(r"\s+", "", t)
    if len(compact) >= _OFFTOPIC_MIN and not _HOTEL.search(t):
        return True
    return False


def is_ignore_reply(text: str) -> bool:
    t = (text or "").strip()
    return t == IGNORE_MARK or t.startswith(IGNORE_MARK)
