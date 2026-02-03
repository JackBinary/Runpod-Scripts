#!/bin/bash
set -e

# install cmake
apt install -y cmake

# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build for AMD MI300X (gfx942)
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx942 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -- -j $(nproc)

# Download Model
mkdir -p /workspace/GLM-4.5-Air-GGUF
curl -L -o /workspace/GLM-4.5-Air-GGUF/GLM-4.5-Air-UD-Q8_K_XL-00001-of-00003.gguf \
  "https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/resolve/main/UD-Q8_K_XL/GLM-4.5-Air-UD-Q8_K_XL-00001-of-00003.gguf"
curl -L -o /workspace/GLM-4.5-Air-GGUF/GLM-4.5-Air-UD-Q8_K_XL-00002-of-00003.gguf \
  "https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/resolve/main/UD-Q8_K_XL/GLM-4.5-Air-UD-Q8_K_XL-00002-of-00003.gguf"
curl -L -o /workspace/GLM-4.5-Air-GGUF/GLM-4.5-Air-UD-Q8_K_XL-00003-of-00003.gguf \
  "https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/resolve/main/UD-Q8_K_XL/GLM-4.5-Air-UD-Q8_K_XL-00003-of-00003.gguf"

# run the model
/llama.cpp/build/bin/llama-server \
  -m /workspace/GLM-4.5-Air-GGUF/GLM-4.5-Air-UD-Q8_K_XL-00001-of-00003.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 999 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  -c 131072
