#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

mkdir -p ../hf_cache/hub
mkdir -p ../hf_cache/transformers
mkdir -p ../hf_cache/torch

export HF_HUB_CACHE=../hf_cache/hub
export TRANSFORMERS_CACHE=../hf_cache/transformers
export TORCH_HOME=../hf_cache/torch

python train_opad.py \
  --instance_data_dir=../dreambooth/dataset/dog \
  --output_dir=outputs/opad_dog/dog \
  --instance_prompt="<new1> dog" \
  --modifier_token="<new1>" \
  --initializer_token=corgi \
  --validation_prompt="a <new1> dog in the jungle" \
  --ip_adapter_image_encoder_path=/path/to/IP-Adapter/models/image_encoder \
  --ip_adapter_ckpt=/path/to/IP-Adapter/models/ip-adapter_sd15.bin
