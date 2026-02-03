#!/bin/bash
set -e  # Exit on any error

# Environment variables with defaults
MODEL=${MODEL:-"zai-org/GLM-4.7-FP8"}
DATASET=${DATASET:-"JackBinary/MixedText-2"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/GLM-4.7_logits-16384"}
UPLOAD_REPO=${UPLOAD_REPO:-""}  # e.g. "JackBinary/GLM-4.7-logits"
UPLOAD_INTERVAL=${UPLOAD_INTERVAL:-3600}  # Upload every hour (in seconds)
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-16384}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.9}
DTYPE=${DTYPE:-"bfloat16"}
MACROBATCH_SIZE=${MACROBATCH_SIZE:-32}  # NEW: Configurable batch size (reduced from 256)

# Compression config parameters
COMPRESSION_D=${COMPRESSION_D:-151552}
COMPRESSION_K=${COMPRESSION_K:-128}
EXACT_K=${EXACT_K:-128}

# Install dependencies
pip install uv
uv pip install huggingface_hub hf_transfer
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/0.14.1/rocm700

# Download model (only if not already downloaded)
echo "Checking model: ${MODEL}"
hf download ${MODEL}

# Clone DistillKit (only if not already cloned)
if [ -d "DistillKit" ]; then
    echo "DistillKit directory already exists, skipping clone"
    cd DistillKit
    git pull || echo "Warning: Could not pull latest changes, using existing version"
else
    echo "Cloning DistillKit..."
    git clone https://github.com/JackBinary/DistillKit.git
    cd DistillKit
fi

pip install -e ".[capture]"

# Create compression config
cat << EOF > compression_config.yaml
d: ${COMPRESSION_D}
k: ${COMPRESSION_K}
exact_k: ${EXACT_K}
residual_bins: []
exact_dtype: bfloat16
polynomial_terms: []
delta_encoding: false
error_diffusion: false
EOF

# Background upload function
upload_task() {
    local output_dir=$1
    local repo_id=$2
    local interval=$3
    
    echo "Starting background upload task to ${repo_id} every ${interval} seconds"
    
    while true; do
        sleep ${interval}
        
        if [ -d "${output_dir}" ] && [ "$(ls -A ${output_dir})" ]; then
            echo "[$(date)] Uploading to HuggingFace: ${repo_id}"
            hf upload ${repo_id} ${output_dir} . \
                --repo-type dataset \
                --commit-message "Auto-upload at $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                --quiet || echo "[$(date)] Upload failed, will retry in ${interval} seconds"
        else
            echo "[$(date)] Output directory empty or doesn't exist yet, skipping upload"
        fi
    done
}

# Start background upload task if UPLOAD_REPO is set
if [ -n "${UPLOAD_REPO}" ]; then
    if [ -z "${HF_TOKEN}" ]; then
        echo "Warning: UPLOAD_REPO is set but HF_TOKEN is not. Uploads will fail."
    else
        # Create the dataset repo if it doesn't exist
        hf repo create ${UPLOAD_REPO} --repo-type dataset || true
        
        # Start upload task in background
        upload_task "${OUTPUT_DIR}" "${UPLOAD_REPO}" "${UPLOAD_INTERVAL}" &
        UPLOAD_PID=$!
        echo "Background upload process started with PID: ${UPLOAD_PID}"
        
        # Ensure upload process is killed when script exits
        trap "kill ${UPLOAD_PID} 2>/dev/null || true" EXIT
    fi
else
    echo "UPLOAD_REPO not set, skipping periodic uploads"
fi

# Set environment variables for memory optimization
export HSA_NO_SCRATCH_RECLAIM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the logit sampling with configurable batch size
python -m distillkit.sample_logits_vllm \
  --model ${MODEL} \
  --dataset ${DATASET} \
  --output ${OUTPUT_DIR} \
  --compression-config ./compression_config.yaml \
  --dtype ${DTYPE} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
  --max-seq-len ${MAX_SEQ_LEN} \
  --macrobatch-size ${MACROBATCH_SIZE}

# Final upload after processing completes
if [ -n "${UPLOAD_REPO}" ] && [ -n "${HF_TOKEN}" ]; then
    echo "Processing complete. Performing final upload..."
    hf upload ${UPLOAD_REPO} ${OUTPUT_DIR} . \
        --repo-type dataset \
        --commit-message "Final upload - processing complete" || echo "Final upload failed"
fi
