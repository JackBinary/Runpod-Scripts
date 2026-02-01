#!/bin/bash
set -e

pip install uv
uv pip install huggingface_hub hf_transfer
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
hf download zai-org/GLM-4.7-FP8
git clone https://github.com/JackBinary/DistillKit.git
cd DistillKit
pip install -e ".[capture]"

cat << 'EOF' > compression_config.yaml
d: 151552
k: 128
exact_k: 128
residual_bins: []
exact_dtype: bfloat16
polynomial_terms: []
delta_encoding: false
error_diffusion: false
EOF

HSA_NO_SCRATCH_RECLAIM=1 python -m distillkit.sample_logits_vllm \
  --model zai-org/GLM-4.7-FP8 \
  --dataset JackBinary/MixedText-2 \
  --output /workspace/GLM-4.7_logits-16384/ \
  --compression-config ./compression_config.yaml \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --max-seq-len 16384
