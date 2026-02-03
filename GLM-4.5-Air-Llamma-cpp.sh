#!/bin/bash
set -e

# Download Model
mkdir -p /workspace/GLM-4.5-Air-GGUF
curl -L -o /workspace/GLM-4.5-Air-GGUF/GLM-4.5-Air-UD-Q8_K_XL-00001-of-00003.gguf \
  "https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/resolve/main/UD-Q8_K_XL/GLM-4.5-Air-UD-Q8_K_XL-00001-of-00003.gguf"
curl -L -o /workspace/GLM-4.5-Air-GGUF/GLM-4.5-Air-UD-Q8_K_XL-00002-of-00003.gguf \
  "https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/resolve/main/UD-Q8_K_XL/GLM-4.5-Air-UD-Q8_K_XL-00002-of-00003.gguf"
curl -L -o /workspace/GLM-4.5-Air-GGUF/GLM-4.5-Air-UD-Q8_K_XL-00003-of-00003.gguf \
  "https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/resolve/main/UD-Q8_K_XL/GLM-4.5-Air-UD-Q8_K_XL-00003-of-00003.gguf"

# Run the model
llama-server \
  -m /workspace/GLM-4.5-Air-GGUF/GLM-4.5-Air-UD-Q8_K_XL-00001-of-00003.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 999 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  -c 131072
