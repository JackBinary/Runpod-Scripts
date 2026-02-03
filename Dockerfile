FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    curl

# Install Python dependencies for model downloading
RUN pip install --no-cache-dir huggingface_hub hf_transfer

# Clone and build llama.cpp
WORKDIR /opt
RUN git clone https://github.com/ggml-org/llama.cpp && \
    cd llama.cpp && \
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx942 \
    -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -- -j $(nproc)

# Set working directory for model storage
WORKDIR /workspace

# Add llama.cpp to PATH
ENV PATH="/opt/llama.cpp/build/bin:${PATH}"

# Default command
CMD ["/bin/bash"]
