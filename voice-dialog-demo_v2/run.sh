#!/usr/bin/env bash
# 远程启动脚本
cd /root/autodl-tmp/voice-dialog-demo_v2
source /root/miniconda3/bin/activate voice

export ASR_MODEL=${ASR_MODEL:-/root/autodl-tmp/models/paraformer-online}
export ASR_OFFLINE_MODEL=${ASR_OFFLINE_MODEL:-/root/autodl-tmp/models/SenseVoiceSmall}
export COSYVOICE_REPO=${COSYVOICE_REPO:-/root/autodl-tmp/CosyVoice}
export TTS_ENGINE=${TTS_ENGINE:-cosyvoice3}
export COSYVOICE_MODEL=${COSYVOICE_MODEL:-/root/autodl-tmp/models/CosyVoice2-0.5B}
export COSYVOICE3_MODEL=${COSYVOICE3_MODEL:-/root/autodl-tmp/models/Fun-CosyVoice3-0.5B}
# 注意：PROMPT_WAV / PROMPT_TEXT 只以 server/config.py 为准，run.sh 不要 export
# （历史坑：run.sh 强制覆盖 config 导致音文错配，zero-shot 输出念参考稿+乱码）
export TTS_LOAD_JIT=${TTS_LOAD_JIT:-1}
export TTS_LOAD_VLLM=${TTS_LOAD_VLLM:-1}
export TTS_LOAD_TRT=${TTS_LOAD_TRT:-1}
export TTS_FIRST_HOP=${TTS_FIRST_HOP:-8}
export DEBUG_AUDIO=${DEBUG_AUDIO:-1}
export PYTHONUNBUFFERED=1

# onnxruntime-gpu 需要的 cu12 运行库（conda 环境内 torch 自带的 nvidia pip 包）
NV=${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cudnn/lib:$NV/cublas/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/cusparse/lib:${LD_LIBRARY_PATH:-}

exec python -m server.main
