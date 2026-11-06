# 小云 · 全流式语音对话 Demo

浏览器麦克风 → (浏览器 AEC) → Silero VAD → FunASR 流式 ASR → DeepSeek 流式 LLM → CosyVoice2 流式 TTS → 浏览器播放。支持打断，全链路流式。实测（远程 4090D，热态）：**TTS 首音频 270-370ms（自 TTS 提交起算），端到端首音频 1.4-2.0s**（瓶颈为 DeepSeek API TTFT 0.9-1.4s）。

## TTS 引擎：CosyVoice3 / CosyVoice2 可切换

`TTS_ENGINE` 环境变量切换（run.sh 默认 `cosyvoice3`），v2 代码（`server/tts.py`）与 v3（`server/tts_v3.py`）完全独立，互不影响：

```bash
TTS_ENGINE=cosyvoice3 bash run.sh   # 用 v3（默认，音质/韵律更好，首块 ~510ms）
TTS_ENGINE=cosyvoice2 bash run.sh   # 回退 v2（首块更快 ~320ms）
```

v3 注意点：prompt 格式 `[指令]<|endofprompt|>参考稿`；语气指令用 `TTS_INSTRUCT_V3`（v3 自带 silent token 过滤，不会像 v2 instruct2 那样把指令念出来）；`token_hop_len` 需匹配训练 chunk（25），不要调小；官方建议 fp16=False（`TTS_V3_FP16` 默认 0）。

## TTS 加速栈（默认全开）

CosyVoice2 三段加速，对应 run.sh 默认 `TTS_LOAD_JIT=1 TTS_LOAD_VLLM=1 TTS_LOAD_TRT=1`：


| 组件                     | 加速                                                                            | 效果         |
| ---------------------- | ----------------------------------------------------------------------------- | ---------- |
| flow encoder           | TorchScript fp16（`load_jit`）                                                  | 编码器图编译     |
| 语音 LLM                 | vLLM 0.9（`load_vllm`，torch 2.7 + bf16 + cudagraph）                            | token 生成提速 |
| flow decoder estimator | TensorRT 10.7 fp16（`load_trt`，首次启动自动从 onnx 编译，~2min，引擎缓存在模型目录 `*.mygpu.plan`） | 扩散解码提速 ~3x |


未加速基线 TTS 首块 ~6.3s（CPU onnx）→ 仅 CUDA onnx ~1.4s → 全开 **~0.3s**。
注意：远程 CosyVoice 仓库 `cosyvoice/utils/file_utils.py` 打过两个 tensorrt 10.x 兼容补丁（EXPLICIT_BATCH / BuilderFlag.FP16 已从新 bindings 移除）；若重装 tensorrt 请用 `tensorrt==10.7.0`（cu12），不要装 11.x（cu13 且 API 不兼容）。

## 架构决策：为什么不用 pipecat

评估过 pipecat（v0.0.x）。结论：**本 demo 手写 asyncio 管道**，理由：

1. **延迟可控性**：300ms 首音频目标要求每一环可精确插桩/调参。pipecat 的 Frame 管线有一层抽象开销，且其 TTS 聚合器默认按句聚合，不利于"首句激进出声"策略。
2. **依赖风险**：pipecat 全家桶（daily/websocket transport + 各 service 适配）在远程裸机安装重，版本变动快；核心收益（VAD、打断）用 silero-vad + 100 行状态机即可覆盖。
3. **模型适配**：FunASR 流式 / CosyVoice2 流式在 pipecat 里没有现成 Service，仍要写自定义 FrameProcessor——那部分工作量与本方案相同。

借鉴了 pipecat 的两个设计：① 代际（generation）机制实现打断即废弃在途音频；② VAD 确认窗口防噪声误打断。

## 目录

```
server/            FastAPI 后端（pipeline.py 为编排核心）
static/            前端页面（AudioWorklet 采集 + 流式上屏 + 音频队列播放）
tools/             部署/测试工具
requirements.txt   Python 依赖
run.sh             启动脚本
```



## 协议（WebSocket /ws）

- C→S：binary = PCM16 16kHz 单声道音频帧
- S→C：binary = PCM16 24kHz TTS 音频块；JSON 文本事件：
`asr_partial / asr_final / llm_delta / llm_done / bot_start / bot_end / interrupted / metrics / state / error`



## 关键延迟策略


| 环节  | 策略                                                                                                                               |
| --- | -------------------------------------------------------------------------------------------------------------------------------- |
| VAD | 450ms 静音判停（可调 `VAD_MIN_SILENCE_MS`），160ms 起话确认防误触                                                                                |
| ASR | paraformer-zh **流式**出字 + 句末 **离线 paraformer-zh** 整句覆盖                                                                            |
| LLM | DeepSeek 流式；首句激进切分（逗号/≥18字即送 TTS）                                                                                                |
| TTS | CosyVoice2 `stream=True` 分块出音频；全局 worker 串行，打断按 gen 丢弃                                                                           |
| 打断  | 播报中 VAD 检话立刻停扬声器并丢弃在途 TTS（`INTERRUPT_MODE=immediate`，默认）。前端用 WebRTC 环回把 TTS 当成远端参考，交给浏览器 AEC 消回声；`confirm` 要等 ASR 出字，混音时容易停不下来 |


## 回声消除（AEC）

引擎仍是浏览器自带的 AEC（`getUserMedia({ echoCancellation: true })`，Chrome/Edge 里即 Chromium WebRTC AEC3）。本仓库没有再引入 Speex / WASM APM 等第三方 AEC。和 v1 的差别只在 **TTS 怎么接到 AEC 的参考通路上**。

**v2 播放路径（本仓库）：**

`TTS → AudioWorklet → MediaStreamDestination → RTCPeerConnection(pc1→pc2) → <audio>`

两个 `RTCPeerConnection` 都在本机，把 TTS 伪装成 WebRTC **远端音轨**。AEC3 是按远端参考设计的，这样才能稳定拿扬声器里的播报当回声模板。服务端另有 VAD 起话确认、短句/句尾回声文本过滤，防止漏网回声再进一轮对话。

| | v2（本仓库，WebRTC 环回） | v1（直出 `<audio>`） |
| --- | --- | --- |
| AEC 引擎 | 浏览器自带 | 同一套 |
| 参考音 | 本地环回成远端轨，AEC 认 | 本地 `MediaStream` 丢给 `<audio>`，Chrome 不一定当参考 |
| 回声消除 | 一般更稳，打断时更少把播报听成用户 | 看不清浏览器是否认真消，回声容易漏进 ASR |
| 链路 | 多一次本机 WebRTC 编解码/重采样 | 短、简单 |
| 副作用 | 多几十毫秒；环回接缝偶发浅爆破 | 音质更干净，回声可能更重 |

打断、回声漏进识别时用 v2 这条参考通路。回声不重、更在意句首/句号接缝时，v1 直出更省事。


## 安装部署

需要一块 NVIDIA GPU，驱动支持 **CUDA 12.6**。建议系统安装 `ffmpeg` 和 conda。

### 1. 安装 Python 依赖

```bash
conda create -n voice python=3.12 -y
conda activate voice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

若 `torch` 找不到 CUDA 12.6 的包，先装这三项再执行上面的 `pip install -r`：

```bash
pip install \
  https://mirrors.aliyun.com/pytorch-wheels/cu126/torch-2.7.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl \
  https://mirrors.aliyun.com/pytorch-wheels/cu126/torchaudio-2.7.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl \
  https://mirrors.aliyun.com/pytorch-wheels/cu126/torchvision-0.22.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl
```

若 `pynini` 安装失败，在同一环境补装：

```bash
conda install -c conda-forge pynini=2.1.7
```



### 2. 下载 CosyVoice 与模型

```bash
mkdir -p /root/autodl-tmp/models

# ASR
modelscope download --model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online \
  --local_dir /root/autodl-tmp/models/paraformer-online
modelscope download --model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  --local_dir /root/autodl-tmp/models/paraformer-offline
# 可选：句末离线识别换成 SenseVoice
modelscope download --model iic/SenseVoiceSmall \
  --local_dir /root/autodl-tmp/models/SenseVoiceSmall

# TTS
modelscope download --model iic/Fun-CosyVoice3-0.5B \
  --local_dir /root/autodl-tmp/models/Fun-CosyVoice3-0.5B
modelscope download --model iic/CosyVoice2-0.5B \
  --local_dir /root/autodl-tmp/models/CosyVoice2-0.5B

# CosyVoice 推理代码
cd /root/autodl-tmp
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice && git submodule update --init --recursive
```

参考音（克隆音色用）放到模型目录，例如 `longanhuan_dense.wav`，并在 `run.sh` 里填写对应文字稿。

首次开启 TensorRT 时会从 onnx 编译引擎（约 2 分钟），之后缓存在模型目录的 `*.mygpu.plan`。换 GPU 或驱动后会重新编译。

### 3. 配置并启动

按本机路径修改 `run.sh`：项目目录、模型目录、CosyVoice 仓库、参考音，以及开头的环境激活（`source .../activate voice`）。

```bash
bash run.sh          # 监听 0.0.0.0:8000
```

可用 `screen` 挂后台。首次加载 vLLM CUDA graph 约 1–2 分钟，`/healthz` 返回 `{"ready":true}` 后即可用。

本机用 SSH 隧道访问（麦克风需要 localhost 或 https）：

```bash
ssh -L 8000:127.0.0.1:8000 -p <端口> <用户>@<主机>
```

浏览器打开 [http://localhost:8000](http://localhost:8000) 。

## 远程无头测试

```bash
python tools/make_test_wav.py                       # 用 TTS 合成测试问句 wav
python tools/test_client.py testdata/q1.wav         # 单轮：看流式事件 + METRICS
python tools/test_client.py testdata/q1.wav --barge testdata/q2.wav   # 打断测试
```
