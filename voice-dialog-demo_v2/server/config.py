import os

# ---------- LLM (DeepSeek) ----------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "200"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "你是云端探索酒店的前台机器人小云。说话像真人前台一样自然口语化：句子短，"
    "哦、呢、哈、呀这些语气词按情况偶尔用，别每句都带，也别完全不用；不要书面腔，不用列表和"
    " markdown，别每句都喊客人。每次回复不超过60字。只处理入住、续住、退房、"
    "早餐、WiFi、停车、房型、房号这些酒店业务，以及客人正在回答你的问题"
    "（几晚、房型、姓名、要不要）。语气温和亲切，带一点俏皮。"
    "客人的话是语音识别来的，可能有同音字错误：比如'晚'会错成'碗'或'万'，"
    "'住'会错成'做'。请结合酒店场景纠正理解（'住两碗'就是'住两晚'）；"
    "遇到明显不合常理的（如住几万晚），先礼貌跟客人确认再回答。"
    "电话、手机号、座机、房号、房间号只写阿拉伯数字（如 13812345678、1208），"
    "不要写成十一万、一千二百这种数量读法。"
    "只回应明确在跟你说话、并且和酒店服务有关的内容。"
    "如果听到的是打电话、旁边人说话、或者是播放的音频、或明显不是对你说的，"
    "或者是你理解不了的、没有上下文、你不明白的（怕是旁边干扰的声音），"
    "都只输出[IGNORE]，不要解释、不要提醒、不要插话、不要安慰。",
)

# ---------- Audio ----------
ASR_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000

# ---------- VAD (Silero) ----------
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.65"))
VAD_NEG_THRESHOLD = float(os.environ.get("VAD_NEG_THRESHOLD", "0.35"))
VAD_MIN_SILENCE_MS = int(os.environ.get("VAD_MIN_SILENCE_MS", "450"))  # 一句话中停顿超过此时长才判说完
VAD_SPEECH_PAD_MS = int(os.environ.get("VAD_SPEECH_PAD_MS", "600"))   # 起始前补偿：必须 > 起话确认窗，否则句首音节被切
VAD_MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", "300"))   # 起话确认，防远场闲聊/句尾残响误
# ---------- 打断 ----------
# immediate: VAD 起话立刻停播（避免人声和 TTS 叠进 ASR）
# confirm: 窗口内 ASR 出字才停播（更抗噪声，但叠音更久）
INTERRUPT_MODE = os.environ.get("INTERRUPT_MODE", "immediate")
INTERRUPT_CONFIRM_MS = int(os.environ.get("INTERRUPT_CONFIRM_MS", "900"))  # confirm 模式用
# 句尾回声宽限窗：扬声器真正停完后该时长内，≤2 字 final 直接丢（SenseVoice 残响碎片）
ECHO_TAIL_MS = int(os.environ.get("ECHO_TAIL_MS", "1000"))

# ---------- ASR (FunASR paraformer 流式 + 句末离线 SenseVoice / paraformer-zh) ----------
ASR_MODEL = os.environ.get(
    "ASR_MODEL",
    "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
)
ASR_MODEL_REVISION = os.environ.get("ASR_MODEL_REVISION", "v2.0.4")
# 句末整句识别。空字符串则退回用流式模型再解一次
# ASR_OFFLINE_MODEL = os.environ.get(
#     "ASR_OFFLINE_MODEL",
#     "/root/autodl-tmp/models/paraformer-offline",
# )
ASR_OFFLINE_MODEL = os.environ.get(
    "ASR_OFFLINE_MODEL",
    "/root/autodl-tmp/models/SenseVoiceSmall",
)
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")
ASR_CHUNK_MS = 600  # funasr online chunk_size=[0,10,5] => 600ms
# 句末用整句再解一次，纠正流式累积漂移；热词提高酒店场景召回
ASR_HOTWORD = os.environ.get(
    "ASR_HOTWORD",
    "入住 续住 退房 早餐 无线网 WiFi 停车 大床房 双床房 豪华间 前台 小云 住两晚 办理",
)

# ---------- 诊断 ----------
# DEBUG_AUDIO=1 时把每句送入 ASR 的音频存到 DEBUG_AUDIO_DIR，用于复盘识别错误
DEBUG_AUDIO = os.environ.get("DEBUG_AUDIO", "0") == "1"
DEBUG_AUDIO_DIR = os.environ.get("DEBUG_AUDIO_DIR", "/root/autodl-tmp/debug_audio")

# ---------- TTS (CosyVoice) ----------
TTS_ENGINE = os.environ.get("TTS_ENGINE", "cosyvoice3")   # cosyvoice2 | cosyvoice3
COSYVOICE_REPO = os.environ.get("COSYVOICE_REPO", "/root/autodl-tmp/CosyVoice")
COSYVOICE_MODEL = os.environ.get("COSYVOICE_MODEL", "/root/autodl-tmp/models/CosyVoice2-0.5B")
COSYVOICE3_MODEL = os.environ.get("COSYVOICE3_MODEL", "/root/autodl-tmp/models/Fun-CosyVoice3-0.5B")
TTS_V3_FP16 = os.environ.get("TTS_V3_FP16", "0") == "1"   # v3 官方示例 fp16=False（DiT TRT fp16 有性能告警）
TTS_INSTRUCT_V3 = os.environ.get("TTS_INSTRUCT_V3", "")   # v3 语气/语速指令（speed 参数在流式下无效；语速指令效果弱且可能干扰克隆）
TTS_MODE = os.environ.get("TTS_MODE", "zero_shot")       # zero_shot | sft
TTS_SPK = os.environ.get("TTS_SPK", "中文女")             # sft 模式用
TTS_FP16 = os.environ.get("TTS_FP16", "1") == "1"
#  加速栈开关 JIT, VLLM，TRT
TTS_LOAD_JIT = os.environ.get("TTS_LOAD_JIT", "0") == "1"
TTS_LOAD_VLLM = os.environ.get("TTS_LOAD_VLLM", "0") == "1"
TTS_LOAD_TRT = os.environ.get("TTS_LOAD_TRT", "0") == "1"
# ============================TTS音色  默认参考音====================================
PROMPT_WAV = os.environ.get(
    "PROMPT_WAV", "/root/autodl-tmp/models/longanhuan_dense.wav"  # 女声
)
PROMPT_TEXT = os.environ.get(
    "PROMPT_TEXT", "咱们一起去操场晒晒太阳吧，今天天气超级好。"
)  # 换音色就换这两个（参考音频+其文字稿）音频内容要和文字一致
# PROMPT_WAV = os.environ.get(
#     "PROMPT_WAV", "/root/autodl-tmp/models/xiaoen_dense.wav"    # 男声
# )
# PROMPT_TEXT = os.environ.get(
#     "PROMPT_TEXT", "今天是二零二五年十一月二十二日"
# )  # 换音色就换这两个（参考音频+其文字稿）音频内容要和文字一致
# ============================TTS音色  默认参考音====================================

# instruct2 语气指令。注意：当前 CosyVoice2-0.5B + 该仓库版本下，instruct2 会把
# 指令文本本身也念出来（prompt 文本无语音配对即被合成），故默认关闭，用 zero_shot。
TTS_INSTRUCT = os.environ.get("TTS_INSTRUCT", "")

# 首块 token 数：25 token ≈ 1s 语音。调小首音频更快，过小影响音质/稳定性。
# v2 实测 8 最优（首块 ~320ms）；v3 需匹配训练 static_chunk_size=25，勿调小。
TTS_FIRST_HOP = int(os.environ.get("TTS_FIRST_HOP", "8"))
TTS_FIRST_HOP_V3 = int(os.environ.get("TTS_FIRST_HOP_V3", "25"))

# 首句快速触发：LLM 输出凑够该长度或遇标点即送 TTS
FIRST_SENT_MIN_CHARS = int(os.environ.get("FIRST_SENT_MIN_CHARS", "6"))
