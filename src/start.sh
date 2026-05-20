#!/usr/bin/env bash

# Function to check if a directory exists and is writable
can_write_to() {
    local target="$1"
    [ -z "$target" ] && return 1

    if [ -d "$target" ]; then
        touch "$target/.write_test" 2> /dev/null || return 1
        rm -f "$target/.write_test"
    else
        mkdir -p "$target" 2> /dev/null || return 1
        touch "$target/.write_test" 2> /dev/null || return 1
        rm -f "$target/.write_test"
    fi

    return 0
}

# Determine NETWORK_VOLUME
if [ -n "${NETWORK_VOLUME-}" ] && can_write_to "$NETWORK_VOLUME"; then
    echo "Using provided NETWORK_VOLUME: $NETWORK_VOLUME"

elif can_write_to "/workspace"; then
    NETWORK_VOLUME="/workspace"
    echo "Defaulting to /workspace"

elif can_write_to "/runpod-volume"; then
    NETWORK_VOLUME="/runpod-volume"
    echo "Defaulting to /runpod-volume"

else
    NETWORK_VOLUME="$(pwd)"
    echo "Fallback to current dir: $NETWORK_VOLUME"
fi

mkdir -p "$NETWORK_VOLUME"
export NETWORK_VOLUME
sed -i '/^export NETWORK_VOLUME=/d' /etc/profile.d/container_env.sh
echo "export NETWORK_VOLUME=\"$NETWORK_VOLUME\"" >> /etc/profile.d/container_env.sh

mkdir -p "$NETWORK_VOLUME/logs"
STARTUP_LOG="$NETWORK_VOLUME/logs/startup.log"
echo "--- Startup log $(date) ---" >> "$STARTUP_LOG"

# Explicitly set the python path
PYTHON_BIN="/usr/bin/python3"

# Keep-alive loop to prevent connection timeout and monitor DNS
(
    echo "Starting network keep-alive service..."
    while true; do
        # Re-enforce DNS just in case the host overrode it
        echo -e "nameserver 8.8.8.8\nnameserver 1.1.1.1" > /etc/resolv.conf
        TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

        # 1. Try to ping Google Drive's API endpoint
        if curl -Is --connect-timeout 5 https://www.google.com > /dev/null 2>&1; then
            echo "[$TIMESTAMP] Internet: REACHABLE (HTTPS)"
        else
            echo "[$TIMESTAMP] Internet: UNREACHABLE"
            # Fallback to check raw DNS resolution via a simple tool like 'host' or 'nslookup'
            if nslookup google.com > /dev/null 2>&1; then
                echo "[$TIMESTAMP] Alert: DNS works, but HTTPS traffic is failing."
            else
                echo "[$TIMESTAMP] Alert: Total network/DNS failure."
            fi
        fi

        # Wait 15 minutes (900 seconds)
        sleep 900
    done
) > "$NETWORK_VOLUME/logs/network_keepalive.log" 2>&1 &

# Run a command quietly, logging output to STARTUP_LOG.
# Shows "Still working..." every 10 seconds.
# On failure, prints a warning with the log path.
run_quiet() {
    local label="$1"
    shift

    # 1. Log a header so you know which command is starting
    echo "====================================================" >> "$STARTUP_LOG"
    echo "BEGIN: $label ($(date))" >> "$STARTUP_LOG"
    echo "COMMAND: $*" >> "$STARTUP_LOG"
    echo "====================================================" >> "$STARTUP_LOG"

    (
        while true; do
            sleep 10
            echo "       Still working on $label..."
        done
    ) &
    local heartbeat_pid=$!

    # 2. Run command. Adding --progress-bar off for pip specifically
    "$@" >> "$STARTUP_LOG" 2>&1
    local exit_code=$?

    kill "$heartbeat_pid" 2> /dev/null
    wait "$heartbeat_pid" 2> /dev/null

    if [ $exit_code -ne 0 ]; then
        echo "       ❌ Warning: $label failed (Exit Code: $exit_code)."
        echo "       Check the end of $STARTUP_LOG for details."
        echo "END: $label (FAILED)" >> "$STARTUP_LOG"
    else
        echo "END: $label (SUCCESS)" >> "$STARTUP_LOG"
    fi

    echo -e "\n" >> "$STARTUP_LOG" # Add spacing between log entries
    return $exit_code
}

# Helper functions for cleaner output
status_msg() { echo -e "\n---> $1"; }

# ============================================================
# Try to find full tcmalloc first, fallback to minimal
# ============================================================

TCMALLOC_PATH=$(ldconfig -p 2> /dev/null | grep -E 'libtcmalloc\.so' | head -n1 | awk '{print $NF}')

if [ -z "$TCMALLOC_PATH" ]; then
    TCMALLOC_PATH=$(ldconfig -p 2> /dev/null | grep -E 'libtcmalloc_minimal\.so' | head -n1 | awk '{print $NF}')
fi

# Apply if found
if [ -n "$TCMALLOC_PATH" ]; then
    export LD_PRELOAD="$TCMALLOC_PATH"
    echo "Using tcmalloc: $TCMALLOC_PATH"
else
    echo "tcmalloc not found, skipping LD_PRELOAD"
fi

# ============================================================
# GPU detection
# ============================================================

if command -v nvidia-smi > /dev/null 2>&1; then

    readarray -t GPU_INFO < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2> /dev/null)

    DETECTED_GPU=$(echo "${GPU_INFO[0]}" | cut -d',' -f1 | xargs)

    CUDA_ARCH=$(printf "%s\n" "${GPU_INFO[@]}" \
        | cut -d',' -f2 \
        | sed 's/\.//g' \
        | sort -u \
        | xargs \
        | tr ' ' ';')

else
    DETECTED_GPU="Unknown GPU"
    CUDA_ARCH="80;86;89;90"
fi

# Final fallback
[ -z "$CUDA_ARCH" ] && CUDA_ARCH="80;86;89;90"

echo "$DETECTED_GPU" > /tmp/detected_gpu

# ============================================================
# Startup banner
# ============================================================
echo ""
echo "================================================"
echo "  Starting up..."
status_msg "Detected GPU: $DETECTED_GPU (Compute Capability: $CUDA_ARCH)"
echo "================================================"

# ---------------------------------------------------------
# Flash Attention 2.8.3
# ---------------------------------------------------------
# Check if already installed (Crucial for persistent environments)
if python -c "import flash_attn" &> /dev/null; then
    status_msg "Flash Attention already installed. Skipping."
else
    # Only install if architecture supports it (Ampere+)
    if echo "$CUDA_ARCH" | grep -Eq '(^|;)(80|86|89|90|100|120)($|;)'; then
        status_msg "Supported architecture detected ($CUDA_ARCH). Installing Flash Attention..."

        PYTHON_VER=$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
        TORCH_VER=$(python -c 'import torch; print(".".join(torch.__version__.split("+")[0].split(".")[:2]))')
        CUDA_VER="128"
        FLASH_ATTENTION_VER="2.8.3"

        FLASH_ATTN_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-${FLASH_ATTENTION_VER}+cu${CUDA_VER}torch${TORCH_VER}-cp${PYTHON_VER}-cp${PYTHON_VER}-linux_x86_64.whl"

        if pip install "$FLASH_ATTN_WHEEL_URL" --no-build-isolation >> "$STARTUP_LOG" 2>&1; then
            echo "FlashAttention installed via wheel" >> "$STARTUP_LOG"
        else
            echo "        -> Wheel install failed. Building from source in background..."
            (
                set -e
                cd /tmp
                rm -rf flash-attention
                git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
                cd flash-attention
                export FLASH_ATTN_CUDA_ARCHS="$CUDA_ARCH"
                export MAX_JOBS=$(nproc)
                export NVCC_THREADS=2
                pip install ninja packaging -q
                python setup.py install
                cd /tmp
                rm -rf flash-attention
            ) > "$NETWORK_VOLUME/logs/flash_attn_install.log" 2>&1 &

            FLASH_ATTN_PID=$!
            echo "$FLASH_ATTN_PID" > /tmp/flash_attn_pid
            echo "        -> Background build started (PID: $FLASH_ATTN_PID)"
        fi
    else
        status_msg "Unsupported architecture ($CUDA_ARCH). Skipping Flash Attention."
    fi
fi

# ---------------------------------------------------------
# Sage Attention 2.x
# ---------------------------------------------------------
if $PYTHON_BIN -c "import sageattention" &> /dev/null; then
    status_msg "SageAttention already installed. Skipping build."
    SAGE_ATTENTION_AVAILABLE=true
else
    # Only attempt install if NOT already installed AND architecture is supported
    if echo "$CUDA_ARCH" | grep -Eq '(^|;)(80|86|89|90|100|120)($|;)'; then
        status_msg "Supported architecture ($CUDA_ARCH) detected. Installing SageAttention 2..."
        run_quiet "SageAttention V2" pip install --no-cache-dir --no-build-isolation git+https://github.com/thu-ml/SageAttention.git@main

        # Link libcuda for the kernels
        ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
        SAGE_ATTENTION_AVAILABLE=true
    else
        status_msg "Unsupported architecture ($CUDA_ARCH). Skipping SageAttention."
        SAGE_ATTENTION_AVAILABLE=false
    fi
fi

# ============================================================
# Setting up workspace
# ============================================================
# This is in case there's any special installs or overrides that needs to occur when starting the machine before starting ComfyUI
if [ -f "$NETWORK_VOLUME/additional_params.sh" ]; then
    chmod +x "$NETWORK_VOLUME/additional_params.sh"
    echo "Executing additional_params.sh..."
    "$NETWORK_VOLUME/comfyui-qwen/src/additional_params.sh"
else
    echo "No additional_params.sh found. Skipping..."
fi

if ! which aria2 > /dev/null 2>&1; then
    echo "Installing aria2..."
    apt-get update && apt-get install -y aria2
else
    echo "aria2 is already installed"
fi

echo "Starting JupyterLab in $NETWORK_VOLUME"
jupyter-lab --ip=0.0.0.0 --allow-root --no-browser \
    --NotebookApp.token='' --NotebookApp.password='' \
    --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True \
    --notebook-dir="$NETWORK_VOLUME" &

# Check if NETWORK_VOLUME is /workspace and set up extra model paths (only if IS_DEV is true)
USE_EXTRA_MODEL_PATHS=false
if [ "$IS_DEV" = "true" ] && [ "$NETWORK_VOLUME" = "/workspace" ]; then
    echo "IS_DEV is true and NETWORK_VOLUME is /workspace. Setting up extra model paths..."

    # Create /models/diffusion_models directory
    mkdir -p /models/diffusion_models

    # Copy all .safetensors files from $NETWORK_VOLUME/ComfyUI/models/diffusion_models to /models/diffusion_models in background
    if [ -d "$NETWORK_VOLUME/ComfyUI/models/diffusion_models" ]; then
        echo "Copying .safetensors files from $NETWORK_VOLUME/ComfyUI/models/diffusion_models to /models/diffusion_models in background..."
        (
            find "$NETWORK_VOLUME/ComfyUI/models/diffusion_models" -name "*.safetensors" -type f | while read -r file; do
                filename=$(basename "$file")
                cp "$file" "/models/diffusion_models/disk_${filename}"
            done
            echo "✅ Finished copying .safetensors files to /models/diffusion_models"
        ) > /tmp/model_copy.log 2>&1 &
        USE_EXTRA_MODEL_PATHS=true
    else
        echo "⚠️  Source directory $NETWORK_VOLUME/ComfyUI/models/diffusion_models does not exist. Skipping copy."
    fi
else
    if [ "$IS_DEV" != "true" ]; then
        echo "IS_DEV is not set to true. Skipping extra model paths setup."
    elif [ "$NETWORK_VOLUME" != "/workspace" ]; then
        echo "NETWORK_VOLUME is not /workspace. Skipping extra model paths setup."
    fi
fi

# Define base paths
COMFYUI_DIR="$NETWORK_VOLUME/ComfyUI"
WORKFLOW_DIR="$NETWORK_VOLUME/ComfyUI/user/default/workflows"
CUSTOM_NODES_DIR="$NETWORK_VOLUME/ComfyUI/custom_nodes"
MODEL_WHITELIST_DIR="$NETWORK_VOLUME/ComfyUI/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt"
DIFFUSION_MODELS_DIR="$NETWORK_VOLUME/ComfyUI/models/diffusion_models"
LORAS_DIR="$NETWORK_VOLUME/ComfyUI/models/loras"
CIVITAI_GGUF="$NETWORK_VOLUME/ComfyUI/models/unet"
TEXT_ENCODERS_DIR="$NETWORK_VOLUME/ComfyUI/models/text_encoders"
VAE_DIR="$NETWORK_VOLUME/ComfyUI/models/vae"
UPSCALE_MODELS_DIR="$NETWORK_VOLUME/ComfyUI/models/upscale_models"
JOYCAPTION_DIR="$NETWORK_VOLUME/ComfyUI/models/LLavacheckpoints/llama-joycaption-beta-one-hf-llava"
FLORENCE2_DIR="$NETWORK_VOLUME/ComfyUI/models/florence2/base-PromptGen"
mkdir -p "$CUSTOM_NODES_DIR"

if [ ! -d "$COMFYUI_DIR" ]; then
    status_msg "First Boot: Moving ComfyUI to Volume..."
    mv /ComfyUI "$COMFYUI_DIR"
    echo "✨ Pristine image deployed to volume. Skipping sync update for faster first boot."
else
    status_msg "Restart detected: Syncing latest Image changes to Volume..."
    # Using . ensures hidden files are included, and -T treats destination as a directory
    cp -ruvT /ComfyUI "$COMFYUI_DIR"
    rm -rf /ComfyUI
    echo "✅ Sync complete."

    # 🔄 SMART SYNC: Only engage on persistent storage restarts
    echo "🔄 Persistent storage detected. Checking for updates and new dependencies..."
    find "$CUSTOM_NODES_DIR" -maxdepth 1 -type d -not -path "$CUSTOM_NODES_DIR" | while read -r node_path; do
        if [ -d "$node_path/.git" ]; then
            node_name=$(basename "$node_path")

            # Check the 'mtime' (modified time) of requirements.txt before pulling
            REQ_FILE="$node_path/requirements.txt"
            BEFORE_MOD=0
            [ -f "$REQ_FILE" ] && BEFORE_MOD=$(stat -c %Y "$REQ_FILE" 2> /dev/null || stat -f %m "$REQ_FILE" 2> /dev/null)

            # Perform the update
            (cd "$node_path" && git pull --ff-only -q > /dev/null 2>&1)

            # Check if requirements.txt exists and if it was updated
            if [ -f "$REQ_FILE" ]; then
                AFTER_MOD=$(stat -c %Y "$REQ_FILE" 2> /dev/null || stat -f %m "$REQ_FILE" 2> /dev/null)

                if [ "$BEFORE_MOD" != "$AFTER_MOD" ]; then
                    echo "📦 New dependencies detected for $node_name. Harmonizing and installing..."

                    # 🛡️ RE-APPLY DOCKERFILE HARMONIZATION PATCHES
                    sed -i -E 's/opencv-(python|contrib-python)(-headless)?(\[[a-zA-Z0-9_-]+\])?(==[0-9.]+)?/opencv-contrib-python-headless/g' "$REQ_FILE"
                    sed -i -E 's/bitsandbytes([>=<~= ]+[0-9.]+)?/bitsandbytes/g' "$REQ_FILE"
                    sed -i -E 's/protobuf([>=<~= ]+[0-9.]+)?/protobuf/g' "$REQ_FILE"
                    sed -i -E 's/^onnxruntime([>=<~= ]+[0-9.]+)?$/onnxruntime-gpu/g' "$REQ_FILE"
                    sed -i -E 's/^torch([>=<~= ]+[0-9.]+)?$/# torch already installed/g' "$REQ_FILE"
                    sed -i -E 's/^torchvision([>=<~= ]+[0-9.]+)?$/# torchvision already installed/g' "$REQ_FILE"
                    sed -i -E 's/^torchaudio([>=<~= ]+[0-9.]+)?$/# torchaudio already installed/g' "$REQ_FILE"
                    sed -i -E 's/^numpy([>=<~= ]+[0-9.]+)?$/# numpy already installed/g' "$REQ_FILE"
                    sed -i -E 's/^numba([>=<~= ]+[0-9.]+)?$/numba/g' "$REQ_FILE"
                    sed -i -E 's/^clip[-_]interrogator([>=<~= ]+[0-9.]+)?$/clip-interrogator/g' "$REQ_FILE"

                    # Use --no-cache-dir to save space on your volume
                    $PYTHON_BIN -m pip install --no-cache-dir -r "$REQ_FILE" > /dev/null 2>&1
                fi
            fi
        fi
    done
    echo "✅ All persistent nodes updated and dependencies verified."
fi

# Acquiring CivitAI Downloader and required models
echo "📥 Setting up CivitAI Downloader..."
if [ ! -f "/usr/local/bin/download_with_aria.py" ]; then
    $PYTHON_BIN -m pip install requests tqdm

    git clone "https://github.com/concreteshoes/CivitAI_Downloader.git" /tmp/CivitAI_Downloader || echo "Git clone failed"
    mv /tmp/CivitAI_Downloader/download_with_aria.py "/usr/local/bin/" || echo "Move failed"
    chmod +x "/usr/local/bin/download_with_aria.py" || echo "Chmod failed"
    rm -rf /tmp/CivitAI_Downloader
else
    echo "✅ CivitAI Downloader already exists."
fi

download_model() {
    local url="$1"
    local full_path="$2"

    local destination_dir=$(dirname "$full_path")
    local destination_file=$(basename "$full_path")

    mkdir -p "$destination_dir"

    # Simple corruption check: file < 10MB or .aria2 files
    if [ -f "$full_path" ]; then
        local size_bytes=$(stat -f%z "$full_path" 2> /dev/null || stat -c%s "$full_path" 2> /dev/null || echo 0)
        local size_mb=$((size_bytes / 1024 / 1024))

        if [ "$size_bytes" -lt 10485760 ]; then # Less than 10MB
            echo "🗑️  Deleting corrupted file (${size_mb}MB < 10MB): $full_path"
            rm -f "$full_path"
        else
            echo "✅ $destination_file already exists (${size_mb}MB), skipping download."
            return 0
        fi
    fi

    # Check for and remove .aria2 control files
    if [ -f "${full_path}.aria2" ]; then
        echo "🗑️  Deleting .aria2 control file: ${full_path}.aria2"
        rm -f "${full_path}.aria2"
        rm -f "$full_path" # Also remove any partial file
    fi

    echo "📥 Downloading $destination_file to $destination_dir..."
    aria2c -x 16 -s 16 -k 1M --continue=true --file-allocation=none -d "$destination_dir" -o "$destination_file" "$url" &

    echo "Download started in background for $destination_file"
}

# ==========================================
# CORE SHARED MODELS (Always Downloaded)
# ==========================================
download_model "https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x/resolve/main/Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors" "$VAE_DIR/Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors"
download_model "https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning/resolve/main/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors" "$LORAS_DIR/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors"
download_model "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors" "$LORAS_DIR/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
download_model "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-ITF-SkinDiffDetail-Lite-v1.pth" "$UPSCALE_MODELS_DIR/1x-ITF-SkinDiffDetail-Lite-v1.pth"

# ==========================================
# QWEN GENERATION (2512 & Edit-2511)
# ==========================================
if [ "${DOWNLOAD_QWEN_2512:-}" = "true" ]; then
    echo "📥 Downloading Qwen Image 2512..."
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_2512_bf16.safetensors" "$DIFFUSION_MODELS_DIR/qwen_image_2512_bf16.safetensors"
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors" "$TEXT_ENCODERS_DIR/qwen_2.5_vl_7b.safetensors"
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" "$VAE_DIR/qwen_image_vae.safetensors"
fi
if [ "${DOWNLOAD_QWEN_2512_GGUF:-}" = "true" ]; then
    echo "📥 Downloading Qwen Image 2512 GGUF (Q8)..."
    download_model "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q8_0.gguf" "$DIFFUSION_MODELS_DIR/qwen-image-2512-Q8_0.gguf"
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors" "$TEXT_ENCODERS_DIR/qwen_2.5_vl_7b.safetensors"
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" "$VAE_DIR/qwen_image_vae.safetensors"
fi
if [ "${DOWNLOAD_QWEN_EDIT_2511:-}" = "true" ]; then
    echo "📥 Downloading Qwen Image Edit 2511..."
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors" "$DIFFUSION_MODELS_DIR/qwen_image_edit_2511_bf16.safetensors"
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors" "$TEXT_ENCODERS_DIR/qwen_2.5_vl_7b.safetensors"
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" "$VAE_DIR/qwen_image_vae.safetensors"
fi
if [ "${DOWNLOAD_QWEN_EDIT_2511_GGUF:-}" = "true" ]; then
    echo "📥 Downloading Qwen Image Edit 2511 GGUF (Q8)..."
    download_model "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q8_0.gguf" "$DIFFUSION_MODELS_DIR/qwen-image-edit-2511-Q8_0.gguf"
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors" "$TEXT_ENCODERS_DIR/qwen_2.5_vl_7b.safetensors"
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" "$VAE_DIR/qwen_image_vae.safetensors"
fi
# ==========================================
# Z-IMAGE MODELS
# ==========================================
if [ "${DOWNLOAD_Z_IMAGE_BASE:-}" = "true" ]; then
    echo "📥 download_z_image_base is set to true. Downloading Z-Image Base models..."
    download_model "https://huggingface.co/Comfy-Org/z_image_base/resolve/main/split_files/diffusion_models/z_image_base_bf16.safetensors" "$DIFFUSION_MODELS_DIR/z_image_base_bf16.safetensors"
    download_model "https://huggingface.co/Comfy-Org/z_image_base/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "$TEXT_ENCODERS_DIR/qwen_3_4b.safetensors"
    download_model "https://huggingface.co/modelzpalace/ae.safetensors/resolve/main/ae.safetensors" "$VAE_DIR/z_image_ae.safetensors"
    echo "✅ Z-Image Base model downloads scheduled"
else
    echo "⏭️  download_z_image_base is not set to true. Skipping Z-Image Base model downloads."
fi
if [ "${DOWNLOAD_Z_IMAGE_BASE_GGUF:-}" = "true" ]; then
    echo "📥 Downloading Z-Image Base GGUF (Q8)..."
    download_model "https://huggingface.co/unsloth/Z-Image-GGUF/resolve/main/z-image-Q8_0.gguf" "$DIFFUSION_MODELS_DIR/z-image-Q8_0.gguf"
    # Text encoder and VAE are shared with non-GGUF — skip if already downloaded
    download_model "https://huggingface.co/Comfy-Org/z_image_base/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "$TEXT_ENCODERS_DIR/qwen_3_4b.safetensors"
    download_model "https://huggingface.co/modelzpalace/ae.safetensors/resolve/main/ae.safetensors" "$VAE_DIR/z_image_ae.safetensors"
    echo "✅ Z-Image Base GGUF model downloads scheduled"
else
    echo "⏭️  download_z_image_base_gguf is not set to true. Skipping Z-Image Base GGUF downloads."
fi
if [ "${DOWNLOAD_Z_IMAGE_TURBO:-}" = "true" ]; then
    echo "📥 download_z_image_turbo is set to true. Downloading Z-Image Turbo..."
    download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" "$DIFFUSION_MODELS_DIR/z_image_turbo_bf16.safetensors"
    # These will skip instantly if Base already downloaded them, but ensures Turbo works if Base was set to false
    download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "$TEXT_ENCODERS_DIR/qwen_3_4b.safetensors"
    download_model "https://huggingface.co/modelzpalace/ae.safetensors/resolve/main/ae.safetensors" "$VAE_DIR/z_image_ae.safetensors"
    echo "✅ Z-Image Turbo model downloads scheduled"
fi
if [ "${DOWNLOAD_Z_IMAGE_TURBO_GGUF:-}" = "true" ]; then
    echo "📥 Downloading Z-Image Turbo GGUF (Q8)..."
    download_model "https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/resolve/main/z-image-turbo-Q8_0.gguf" "$DIFFUSION_MODELS_DIR/z-image-turbo-Q8_0.gguf"
    # These will skip instantly if Base or non-GGUF Turbo already downloaded them
    download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "$TEXT_ENCODERS_DIR/qwen_3_4b.safetensors"
    download_model "https://huggingface.co/modelzpalace/ae.safetensors/resolve/main/ae.safetensors" "$VAE_DIR/z_image_ae.safetensors"
    echo "✅ Z-Image Turbo GGUF model downloads scheduled"
fi

# ==========================================
# CHROMA1 HD
# ==========================================
if [ "${DOWNLOAD_CHROMA1_HD:-}" = "true" ]; then
    echo "📥 Downloading Chroma1 HD..."
    download_model "https://huggingface.co/lodestones/Chroma1-HD/resolve/main/Chroma1-HD.safetensors" "$DIFFUSION_MODELS_DIR/Chroma1-HD.safetensors"
    download_model "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" "$TEXT_ENCODERS_DIR/t5xxl_fp16.safetensors"
    download_model "https://huggingface.co/lodestones/Chroma/resolve/main/ae.safetensors" "$VAE_DIR/chroma1_hd_ae.safetensors"
    echo "✅ Chroma1 HD model downloads scheduled"
fi
if [ "${DOWNLOAD_CHROMA1_HD_GGUF:-}" = "true" ]; then
    echo "📥 Downloading Chroma1 HD GGUF (Q8)..."
    download_model "https://huggingface.co/silveroxides/Chroma1-HD-GGUF/resolve/main/Chroma1-HD-Q8_0.gguf" "$DIFFUSION_MODELS_DIR/Chroma1-HD-Q8_0.gguf"
    download_model "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" "$TEXT_ENCODERS_DIR/t5xxl_fp16.safetensors"
    download_model "https://huggingface.co/lodestones/Chroma/resolve/main/ae.safetensors" "$VAE_DIR/chroma1_hd_ae.safetensors"
    echo "✅ Chroma1 HD GGUF model downloads scheduled"
fi

# ==========================================
# JOYCAPTION BETA ONE
# ==========================================
if [ "${DOWNLOAD_JOYCAPTION:-}" = "true" ]; then
    echo "📥 Downloading JoyCaption Beta One..."

    # 1. Config & Tokenizer Files
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/config.json" "$JOYCAPTION_DIR/config.json"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/generation_config.json" "$JOYCAPTION_DIR/generation_config.json"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/model.safetensors.index.json" "$JOYCAPTION_DIR/model.safetensors.index.json"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/preprocessor_config.json" "$JOYCAPTION_DIR/preprocessor_config.json"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/special_tokens_map.json" "$JOYCAPTION_DIR/special_tokens_map.json"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/tokenizer.json" "$JOYCAPTION_DIR/tokenizer.json"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/tokenizer_config.json" "$JOYCAPTION_DIR/tokenizer_config.json"

    # 2. Sharded Weights
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/model-00001-of-00004.safetensors" "$JOYCAPTION_DIR/model-00001-of-00004.safetensors"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/model-00002-of-00004.safetensors" "$JOYCAPTION_DIR/model-00002-of-00004.safetensors"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/model-00003-of-00004.safetensors" "$JOYCAPTION_DIR/model-00003-of-00004.safetensors"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/model-00004-of-00004.safetensors" "$JOYCAPTION_DIR/model-00004-of-00004.safetensors"

    echo "✅ JoyCaption Beta One model downloads scheduled"
fi

# ==========================================
# FLORENCE-2 NSFW V2
# ==========================================
if [ "${DOWNLOAD_FLORENCE2:-}" = "true" ]; then
    echo "📥 Downloading Florence-2 NSFW finetune..."

    # Base URL for the finetune
    NSFW_BASE_URL="https://huggingface.co/ljnlonoljpiljm/florence-2-base-nsfw-v2/resolve/main"

    # 1. Core Configuration & Tokenizer
    download_model "$NSFW_BASE_URL/config.json" "$FLORENCE2_DIR/config.json"
    download_model "$NSFW_BASE_URL/generation_config.json" "$FLORENCE2_DIR/generation_config.json"
    download_model "$NSFW_BASE_URL/preprocessor_config.json" "$FLORENCE2_DIR/preprocessor_config.json"
    download_model "$NSFW_BASE_URL/added_tokens.json" "$FLORENCE2_DIR/added_tokens.json"
    download_model "$NSFW_BASE_URL/merges.txt" "$FLORENCE2_DIR/merges.txt"
    download_model "$NSFW_BASE_URL/special_tokens_map.json" "$FLORENCE2_DIR/special_tokens_map.json"
    download_model "$NSFW_BASE_URL/tokenizer.json" "$FLORENCE2_DIR/tokenizer.json"
    download_model "$NSFW_BASE_URL/tokenizer_config.json" "$FLORENCE2_DIR/tokenizer_config.json"
    download_model "$NSFW_BASE_URL/vocab.json" "$FLORENCE2_DIR/vocab.json"

    # 2. The Weights
    download_model "$NSFW_BASE_URL/model.safetensors" "$FLORENCE2_DIR/model.safetensors"

    # 3. Microsoft Processor (Handles the actual image bounding boxes/cropping)
    download_model "https://huggingface.co/microsoft/Florence-2-base/resolve/main/processing_florence2.py" "$FLORENCE2_DIR/processing_florence2.py"

    # 4. APPLY THE KIJAI / LAYERSTYLE PATCH
    # We copy the patched modeling and config files directly from the custom node directory
    # to overwrite any missing or outdated files, ensuring transformers >= 4.45 compatibility.
    echo "🔧 Applying Transformers compatibility patch for Florence-2..."
    LAYERSTYLE_MODELS_DIR="/ComfyUI/custom_nodes/ComfyUI_LayerStyle_Advance/florence2_models"

    if [ -d "$LAYERSTYLE_MODELS_DIR" ]; then
        cp "$LAYERSTYLE_MODELS_DIR/modeling_florence2.py" "$FLORENCE2_DIR/"
        cp "$LAYERSTYLE_MODELS_DIR/configuration_florence2.py" "$FLORENCE2_DIR/"
        echo "✅ Florence-2 patched successfully."
    else
        echo "⚠️ WARNING: LayerStyle advance folder not found at $LAYERSTYLE_MODELS_DIR. Patch skipped."
    fi

    echo "✅ Florence-2 NSFW scheduled"
fi

# Download MultiAngle.safetensors to LORAS_DIR using wget
mkdir -p "$LORAS_DIR"
if [ ! -f "$LORAS_DIR/MultiAngle.safetensors" ]; then
    echo "📥 Downloading MultiAngle.safetensors to $LORAS_DIR..."
    wget -O "$LORAS_DIR/MultiAngle.safetensors" "https://huggingface.co/Hearmeman/MultiAngleQwen/resolve/main/MultiAngle.safetensors"
else
    echo "✅ MultiAngle.safetensors already exists, skipping download."
fi

# Download additional models
echo "📥 Starting additional model downloads..."

if [ ! -f "$NETWORK_VOLUME/ComfyUI/models/upscale_models/4xLSDIR.pth" ]; then
    if [ -f "/4xLSDIR.pth" ]; then
        mv "/4xLSDIR.pth" "$NETWORK_VOLUME/ComfyUI/models/upscale_models/4xLSDIR.pth"
        echo "Moved 4xLSDIR.pth to the correct location."
    else
        echo "4xLSDIR.pth not found in the root directory."
    fi
else
    echo "4xLSDIR.pth already exists. Skipping."
fi

echo "Finished downloading models!"

declare -A MODEL_CATEGORIES=(
    ["$NETWORK_VOLUME/ComfyUI/models/loras"]="$LORAS_IDS_TO_DOWNLOAD"
    ["$NETWORK_VOLUME/ComfyUI/models/checkpoints"]="$SDXL_MODEL_IDS_TO_DOWNLOAD"
)

# Counter to track background jobs
download_count=0

# Ensure directories exist and schedule downloads in background
for TARGET_DIR in "${!MODEL_CATEGORIES[@]}"; do
    mkdir -p "$TARGET_DIR"
    MODEL_IDS_STRING="${MODEL_CATEGORIES[$TARGET_DIR]}"

    # SKIP if variable is empty or contains the default placeholder
    if [[ -z "$MODEL_IDS_STRING" || "$MODEL_IDS_STRING" == "replace_with_ids" ]]; then
        echo "⏭️  Skipping downloads for $TARGET_DIR (No IDs provided)"
        continue
    fi

    IFS=',' read -ra MODEL_IDS <<< "$MODEL_IDS_STRING"
    for MODEL_ID in "${MODEL_IDS[@]}"; do
        # Strip potential whitespace
        CLEAN_ID=$(echo "$MODEL_ID" | xargs)
        [ -z "$CLEAN_ID" ] && continue

        echo "🚀 Scheduling download: $CLEAN_ID to $TARGET_DIR"
        (cd "$TARGET_DIR" && download_with_aria.py -m "$CLEAN_ID") &
        ((download_count++))
    done
done

echo "📋 Scheduled $download_count downloads in background"

# Wait for all downloads to complete
if pgrep -x "aria2c" > /dev/null; then
    echo "⏳ Waiting for downloads..."
    while pgrep -x "aria2c" > /dev/null; do
        sleep 5
    done
fi

echo "✅ All models downloaded successfully!"

SOURCE_DIR="/comfyui-qwen/workflows"

# Ensure destination directory exists
mkdir -p "$WORKFLOW_DIR"

# Loop over each file in the source directory
for file in "$SOURCE_DIR"/*; do
    # Skip if it's not a file
    [[ -f "$file" ]] || continue

    dest_file="$WORKFLOW_DIR/$(basename "$file")"

    if [[ -e "$dest_file" ]]; then
        echo "File already exists in destination. Deleting: $file"
        rm -f "$file"
    else
        echo "Moving: $file to $WORKFLOW_DIR"
        mv "$file" "$WORKFLOW_DIR"
    fi
done

# Workspace as main working directory
echo "cd $NETWORK_VOLUME" >> ~/.bashrc

echo "Updating default preview method..."
CONFIG_PATH="$NETWORK_VOLUME/ComfyUI/user/default/ComfyUI-Manager"
CONFIG_FILE="$CONFIG_PATH/config.ini"

# Ensure the directory exists
mkdir -p "$CONFIG_PATH"

# Create the config file if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating config.ini..."
    cat << EOL > "$CONFIG_FILE"
[default]
preview_method = auto
git_exe =
use_uv = False
channel_url = https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main
share_option = all
bypass_ssl = False
file_logging = True
component_policy = workflow
update_policy = stable-comfyui
windows_selector_event_loop_policy = False
model_download_by_agent = False
downgrade_blacklist =
security_level = normal
skip_migration_check = False
always_lazy_install = False
network_mode = public
db_mode = cache
EOL
else
    echo "config.ini already exists. Updating preview_method..."
    sed -i 's/^preview_method = .*/preview_method = auto/' "$CONFIG_FILE"
fi
echo "Config file setup complete!"
echo "Default preview method updated to 'auto'"

# GPU VRAM check
# Grabs the total memory of the first GPU in MB
GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
VRAM_THRESHOLD=32000 # 32GB in MB

# Start with base flags
LAUNCH_FLAGS="--listen --preview-method auto"

# Add FP8 flags if enabled
if [ "${USE_FP8_TEXT_ENC:-true}" = "true" ]; then
    LAUNCH_FLAGS="$LAUNCH_FLAGS --fp8_e4m3fn-text-enc"
    status_msg "FP8 text encoder enabled"
fi

if [ "${USE_FP8_MODEL:-}" = "true" ]; then
    LAUNCH_FLAGS="$LAUNCH_FLAGS --fp8_e4m3fn-unet"
    status_msg "FP8 model weight casting enabled (E4M3FN)"
fi

# Memory Optimization based on VRAM
if [ "$GPU_VRAM_MB" -ge "$VRAM_THRESHOLD" ]; then
    echo "🚀 High VRAM detected (32GB+). Enabling --highvram."
    LAUNCH_FLAGS="$LAUNCH_FLAGS --highvram"
else
    echo "⚖️ Standard VRAM detected."
    # ComfyUI natively uses --weight-dtype to force model precision
    LAUNCH_FLAGS="$LAUNCH_FLAGS --medvram"
fi

# Add SageAttention
if [ "$SAGE_ATTENTION_AVAILABLE" = "true" ]; then
    LAUNCH_FLAGS="$LAUNCH_FLAGS --use-sage-attention"
fi

# Final Command Construction
COMFYUI_CMD="$PYTHON_BIN $COMFYUI_DIR/main.py $LAUNCH_FLAGS"

# Launch
URL="http://127.0.0.1:8188"
status_msg "▶️ Starting ComfyUI with flags: $LAUNCH_FLAGS"
nohup $COMFYUI_CMD > "$NETWORK_VOLUME/comfyui_nohup.log" 2>&1 &
echo $! > /tmp/comfyui.pid # Save PID for restart

# Debugging mode
cat > /usr/local/bin/comfyui-restart << 'EOF'
#!/bin/bash

PYTHON_BIN="/usr/bin/python3"
COMFYUI_DIR="${NETWORK_VOLUME:-/workspace}/ComfyUI"
LOG_FILE="${NETWORK_VOLUME:-/workspace}/comfyui_nohup.log"

echo "Stopping ComfyUI..."
kill $(cat /tmp/comfyui.pid 2>/dev/null) 2>/dev/null
sleep 2

echo "Relaunching with debug flags..."
BASE_FLAGS="--listen --preview-method auto --use-sage-attention"

echo "Base flags: $BASE_FLAGS"
echo "Extra flags: $@"

nohup $PYTHON_BIN $COMFYUI_DIR/main.py \
    $BASE_FLAGS $@ \
    > "$LOG_FILE" 2>&1 &

echo $! > /tmp/comfyui.pid
echo "ComfyUI restarted PID $(cat /tmp/comfyui.pid)"
EOF

chmod +x /usr/local/bin/comfyui-restart

# Timeout logic
counter=0
max_wait=100 # safer for cold starts + model init

until curl --silent --fail "$URL" --output /dev/null; do
    if [ $counter -ge $max_wait ]; then
        echo "❌ Timeout: ComfyUI failed to start within ${max_wait}s."
        echo "📋 Check logs: tail -n 100 $NETWORK_VOLUME/comfyui_nohup.log"
        exit 1
    fi

    echo "🔄 ComfyUI Starting... (${counter}s/${max_wait}s)"
    sleep 5
    counter=$((counter + 5))
done

# Final Verification
if curl --silent --fail "$URL" --output /dev/null; then
    echo "🚀 ComfyUI is ready."
fi

echo ""
echo "================================================"
echo ""
echo "  Template ready!"
echo ""
echo "  To access JupyterLab from your local machine:"
echo ""
echo "  1) Use the SSH command provided by your host (Vast.ai / RunPod),"
echo "     and add port forwarding like this:"
echo ""
echo "     ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 8888:localhost:8888"
echo ""
echo "  2) Then open your browser:"
echo "     http://localhost:8888/lab"
echo ""
echo "  To access ComfyUI GUI on port 8188:"
echo ""
echo "     ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 8188:localhost:8188"
echo ""
echo "     Then open your browser to:"
echo "     http://localhost:8188"
echo ""
echo "  You can also access JupyterLab via the RunPod web interface if deployed there"
echo ""
echo "================================================"
echo ""

# ================================
# SSH Startup
# ================================

echo "🔐 Starting SSH server..."

mkdir -p /var/run/sshd
chmod 700 /root/.ssh

# If SSH_PUBLIC_KEY provided via env, append safely
if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    echo "Adding SSH_PUBLIC_KEY from environment..."
    touch /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys

    # Avoid duplicates
    grep -qxF "$SSH_PUBLIC_KEY" /root/.ssh/authorized_keys 2> /dev/null \
        || echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi

/usr/sbin/sshd

echo "✅ SSH ready."

status_msg "Initialization complete"

# Stream the log to the container output so 'docker logs' works
tail -f "$NETWORK_VOLUME/comfyui_nohup.log" &

sleep infinity
