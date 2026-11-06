#!/usr/bin/env bash
# 远程服务器一键环境准备：venv + 依赖 + 模型下载（modelscope）+ CosyVoice 代码
set -e
cd /root/autodl-tmp/voice-dialog-demo_v2

source /root/miniconda3/bin/activate
if [ ! -d .venv ]; then
  python -m venv --system-site-packages .venv
fi
source .venv/bin/activate

pip install -r requirements.txt

# ---- ASR 模型（paraformer-zh 流式 + 离线）----
mkdir -p /root/autodl-tmp/models
if [ ! -d /root/autodl-tmp/models/paraformer-online ]; then
  modelscope download --model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online \
    --local_dir /root/autodl-tmp/models/paraformer-online
fi
if [ ! -d /root/autodl-tmp/models/paraformer-offline ]; then
  modelscope download --model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --local_dir /root/autodl-tmp/models/paraformer-offline
fi
if [ ! -d /root/autodl-tmp/models/SenseVoiceSmall ]; then
  modelscope download --model iic/SenseVoiceSmall \
    --local_dir /root/autodl-tmp/models/SenseVoiceSmall
fi

# ---- CosyVoice2-0.5B 模型权重 ----
if [ ! -d /root/autodl-tmp/models/CosyVoice2-0.5B ]; then
  modelscope download --model iic/CosyVoice2-0.5B \
    --local_dir /root/autodl-tmp/models/CosyVoice2-0.5B
fi

# ---- CosyVoice 推理代码 ----
if [ ! -d /root/autodl-tmp/CosyVoice ]; then
  cd /root/autodl-tmp
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git || \
    git clone --recursive https://gitclone.com/github.com/FunAudioLLM/CosyVoice.git
  cd CosyVoice
  git submodule update --init --recursive
  cd /root/autodl-tmp/voice-dialog-demo_v2
fi

# CosyVoice 依赖（torch 已用系统 2.5.1+cu124）
pip install -r /root/autodl-tmp/CosyVoice/requirements.txt || \
  pip install HyperPyYAML soundfile librosa omegaconf einops onnxruntime-gpu \
              diffusers gradio matplotlib conformer wetextprocessing inflect \
              pyworld junction typer rich

echo "==== setup done ===="
python -c "import torch, funasr, fastapi, openai, silero_vad; print('imports ok')"
