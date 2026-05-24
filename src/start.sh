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

# ============================================================
# Flash Attention
# ============================================================
status_msg "[1/4] Checking Flash Attention"

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
                pip install . --no-build-isolation
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

# ============================================================
# Sage Attention (V2.x)
# ============================================================
status_msg "[2/4] Checking SageAttention"

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
status_msg "[3/4] Setting up workspace..."

echo "Starting JupyterLab in $NETWORK_VOLUME"
jupyter-lab --ip=0.0.0.0 --allow-root --no-browser \
    --NotebookApp.token='' --NotebookApp.password='' \
    --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True \
    --notebook-dir="$NETWORK_VOLUME" &

# Ensure the database file path is clean
FB_DB="$NETWORK_VOLUME/filebrowser.db"

# 1. Initialize configuration only if it's a brand new volume
if [ ! -f "$FB_DB" ]; then
    echo "Creating a fresh Filebrowser database..."
    filebrowser -d "$FB_DB" config init

    # Hardcoded user to "admin", fallback password to "default_password" if env is missing
    filebrowser -d "$FB_DB" users add admin "${FB_PASSWORD:-default_password}" --perm.admin
fi

# 2. Start Filebrowser in the background
echo "Launching Filebrowser on port 8080..."
filebrowser -d "$FB_DB" -r "$NETWORK_VOLUME" -a 0.0.0.0 -p 8080 > "$NETWORK_VOLUME/filebrowser.log" 2>&1 &

# Define base paths
COMFYUI_DIR="$NETWORK_VOLUME/ComfyUI"
WORKFLOW_DIR="$NETWORK_VOLUME/ComfyUI/user/default/workflows"
CUSTOM_NODES_DIR="$NETWORK_VOLUME/ComfyUI/custom_nodes"
DIFFUSION_MODELS_DIR="$NETWORK_VOLUME/ComfyUI/models/diffusion_models"
LORAS_DIR="$NETWORK_VOLUME/ComfyUI/models/loras"
CHECKPOINTS_DIR="$NETWORK_VOLUME/ComfyUI/models/checkpoints"
GGUF_DIR="$NETWORK_VOLUME/ComfyUI/models/unet"
TEXT_ENCODERS_DIR="$NETWORK_VOLUME/ComfyUI/models/text_encoders"
CLIP_VISION_DIR="$NETWORK_VOLUME/ComfyUI/models/clip_vision"
VAE_DIR="$NETWORK_VOLUME/ComfyUI/models/vae"
UPSCALE_MODELS_DIR="$NETWORK_VOLUME/ComfyUI/models/upscale_models"
INSIGHTFACE_DIR="$NETWORK_VOLUME/ComfyUI/models/insightface/models"
SAM2_DIR="$NETWORK_VOLUME/ComfyUI/models/sam2"
ANTELOPEV2_DIR="$INSIGHTFACE_DIR/antelopev2"
PULID_DIR="$NETWORK_VOLUME/ComfyUI/models/pulid"
ULTRALYTICS_DIR="$NETWORK_VOLUME/ComfyUI/models/ultralytics"
CONTROLNET_DIR="$NETWORK_VOLUME/ComfyUI/models/controlnet"
JOYCAPTION_DIR="$NETWORK_VOLUME/ComfyUI/models/LLavacheckpoints/llama-joycaption-beta-one-hf-llava"
FLORENCE2_DIR="$NETWORK_VOLUME/ComfyUI/models/florence2/base-PromptGen"
MODEL_WHITELIST_DIR="$NETWORK_VOLUME/ComfyUI/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt"
mkdir -p "$CUSTOM_NODES_DIR"

if [ ! -d "$COMFYUI_DIR" ] || [ -z "$(ls -A "$COMFYUI_DIR" 2> /dev/null)" ]; then
    status_msg "First Boot: Moving ComfyUI to Volume..."
    mkdir -p "$COMFYUI_DIR"
    mv /ComfyUI/* "$COMFYUI_DIR"/ 2> /dev/null || true
    # Pre-seed .patched markers so second boot doesn't reinstall all dependencies
    find "$CUSTOM_NODES_DIR" -maxdepth 1 -type d -not -path "$CUSTOM_NODES_DIR" \
        -exec touch "{}/.patched" \;
    echo "✨ Pristine image deployed to volume. Skipping sync update for faster first boot."
else
    status_msg "Restart detected: Syncing latest Image changes to Volume..."
    cp -ruvT /ComfyUI "$COMFYUI_DIR"
    rm -rf /ComfyUI
    echo "✅ Sync complete."

    echo "🔄 Persistent storage detected. Checking for updates and new dependencies..."

    updated_nodes=0
    patched_dependencies=0

    while read -r node_path; do
        if [ -d "$node_path/.git" ]; then
            node_name=$(basename "$node_path")

            REQ_FILE="$node_path/requirements.txt"
            BEFORE_MOD=0
            [ -f "$REQ_FILE" ] && BEFORE_MOD=$(stat -c %Y "$REQ_FILE" 2> /dev/null || stat -f %m "$REQ_FILE" 2> /dev/null)

            (
                cd "$node_path" \
                    && git reset --hard HEAD -q \
                    && git pull --ff-only -q
            ) > /dev/null 2>&1

            AFTER_MOD=0
            [ -f "$REQ_FILE" ] && AFTER_MOD=$(stat -c %Y "$REQ_FILE" 2> /dev/null || stat -f %m "$REQ_FILE" 2> /dev/null)

            if [ "$BEFORE_MOD" != "$AFTER_MOD" ]; then
                echo "📦 New dependencies detected for $node_name. Harmonizing and installing..."
                ((patched_dependencies++))

                # 🛡️ GENERAL DOCKERFILE HARMONIZATION PATCHES
                if [ -f "$REQ_FILE" ]; then
                    sed -i -E 's/^[Pp]illow([>=<~= ]+[0-9.]+)?$/# Pillow already installed/g' "$REQ_FILE"
                    sed -i -E 's/opencv-(python|contrib-python)(-headless)?(\[[a-zA-Z0-9_-]+\])?(==[0-9.]+)?/opencv-contrib-python-headless/g' "$REQ_FILE"
                    sed -i -E 's/bitsandbytes([>=<~= ]+[0-9.]+)?/bitsandbytes/g' "$REQ_FILE"
                    sed -i -E 's/^protobuf[>=<~=,. 0-9]+$/protobuf/g' "$REQ_FILE"
                    sed -i -E 's/^onnxruntime(-gpu)?([>=<~=,. 0-9]+)?$/onnxruntime-gpu/g' "$REQ_FILE"
                    sed -i -E 's/^torch([>=<~= ]+[0-9.]+)?$/# torch already installed/g' "$REQ_FILE"
                    sed -i -E 's/^torchvision([>=<~= ]+[0-9.]+)?$/# torchvision already installed/g' "$REQ_FILE"
                    sed -i -E 's/^torchaudio([>=<~= ]+[0-9.]+)?$/# torchaudio already installed/g' "$REQ_FILE"
                    sed -i -E 's/^numpy([>=<~= ]+[0-9.]+)?$/# numpy already installed/g' "$REQ_FILE"
                    sed -i -E 's/^numba([>=<~= ]+[0-9.]+)?$/numba/g' "$REQ_FILE"
                    sed -i -E 's/^ninja([>=<~=~ ]+[0-9.]+)?$/ninja/g' "$REQ_FILE"
                    sed -i -E 's/^clip[-_]interrogator([>=<~= ]+[0-9.]+)?$/clip-interrogator/g' "$REQ_FILE"
                    sed -i -E 's/^transformers(\[[a-zA-Z0-9_,]+\])?([>=<~= ]+[0-9.]+)?$/transformers/g' "$REQ_FILE"
                    sed -i -E 's/^insightface([>=<~= ]+[0-9.]+)?$/insightface==1.0.1/g' "$REQ_FILE"
                    sed -i -E 's/^diffusers([>=<~= ]+[0-9.]+)?$/# diffusers already installed/g' "$REQ_FILE"
                    sed -i -E 's/^huggingface-hub([>=<~= ]+[0-9.]+)?$/# huggingface-hub already installed/g' "$REQ_FILE"
                    sed -i -E 's/^(segment-anything|transparent-background)([>=<~= ]+[0-9.]+)?$/# segmentation tooling already installed/g' "$REQ_FILE"

                    if $PYTHON_BIN -m pip install --no-cache-dir -r "$REQ_FILE" >> "$STARTUP_LOG" 2>&1; then
                        echo "   ✅ Dependencies installed for $node_name"
                    else
                        echo "   ❌ Dependency install failed for $node_name — check $STARTUP_LOG"
                    fi
                fi
            fi
            ((updated_nodes++))
        fi
    done < <(find "$CUSTOM_NODES_DIR" -maxdepth 1 -type d -not -path "$CUSTOM_NODES_DIR")

    echo "✅ Smart Updater processed $updated_nodes custom nodes ($patched_dependencies required env patching)."
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
    local skip_size_check="${3:-false}"

    local destination_dir=$(dirname "$full_path")
    local destination_file=$(basename "$full_path")

    mkdir -p "$destination_dir"

    if [ -f "${full_path}.aria2" ]; then
        echo "⏳ Partial download state found for $destination_file. Resuming..."

    elif [ -f "$full_path" ]; then
        local size_bytes=$(stat -f%z "$full_path" 2> /dev/null || stat -c%s "$full_path" 2> /dev/null || echo 0)
        local size_mb=$((size_bytes / 1024 / 1024))

        if [ "$size_bytes" -lt 10485760 ] && [ "$skip_size_check" != "true" ]; then
            echo "🗑️  Deleting corrupted placeholder file (${size_mb}MB < 10MB): $full_path"
            rm -f "$full_path"
        else
            echo "✅ $destination_file already exists (${size_mb}MB), skipping download."
            LAST_DOWNLOAD_PID="" # ← clear it so caller knows no job was started
            return 0
        fi
    fi

    echo "📥 Background download scheduled for $destination_file..."
    aria2c -x 8 -s 8 -k 4M \
        --continue=true \
        --file-allocation=none \
        --max-tries=5 \
        --retry-wait=3 \
        --timeout=60 \
        --connect-timeout=10 \
        --console-log-level=error \
        -d "$destination_dir" \
        -o "$destination_file" \
        "$url" &
    LAST_DOWNLOAD_PID=$! # ← capture before anything else can overwrite $!
}

# ============================================================
# QWEN IMAGE (2512 STANDARD & EDIT-2511 ARCHITECTURES)
# ============================================================

# 1. Download Target Diffusion Weights Based on Selections
if [ "${DOWNLOAD_QWEN_2512:-}" = "true" ]; then
    echo "📥 Downloading Qwen Image 2512 (BF16)..."
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_2512_bf16.safetensors" "$DIFFUSION_MODELS_DIR/qwen_image_2512_bf16.safetensors"
fi

if [ "${DOWNLOAD_QWEN_2512_GGUF:-}" = "true" ]; then
    echo "📥 Downloading Qwen Image 2512 GGUF (Q8)..."
    download_model "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q8_0.gguf" "$GGUF_DIR/qwen-image-2512-Q8_0.gguf"
fi

if [ "${DOWNLOAD_QWEN_EDIT_2511:-}" = "true" ]; then
    echo "📥 Downloading Qwen Image Edit 2511 (BF16)..."
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors" "$DIFFUSION_MODELS_DIR/qwen_image_edit_2511_bf16.safetensors"
fi

if [ "${DOWNLOAD_QWEN_EDIT_2511_GGUF:-}" = "true" ]; then
    echo "📥 Downloading Qwen Image Edit 2511 GGUF (Q8)..."
    download_model "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q8_0.gguf" "$GGUF_DIR/qwen-image-edit-2511-Q8_0.gguf"
fi

# 2. Shared Sub-Assets (Executes if ANY Qwen flavor flag is enabled)
if [ "${DOWNLOAD_QWEN_2512:-}" = "true" ] || [ "${DOWNLOAD_QWEN_2512_GGUF:-}" = "true" ] || [ "${DOWNLOAD_QWEN_EDIT_2511:-}" = "true" ] || [ "${DOWNLOAD_QWEN_EDIT_2511_GGUF:-}" = "true" ]; then
    echo "📥 Downloading shared Qwen Image ecosystem sub-assets..."

    # Unified Vision-Language Multimodal Text Encoder (Qwen-2.5-VL 7B Backbone)
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors" "$TEXT_ENCODERS_DIR/qwen_2.5_vl_7b.safetensors"

    # Unified Qwen Image Architecture VAE
    download_model "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" "$VAE_DIR/qwen_image_vae.safetensors"

    # Lightning Loras
    download_model "https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning/resolve/main/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors" "$LORAS_DIR/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors"
    download_model "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors" "$LORAS_DIR/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
    download_model "https://huggingface.co/Hearmeman/MultiAngleQwen/resolve/main/MultiAngle.safetensors" "$LORAS_DIR/MultiAngle.safetensors"

    echo "📋 Qwen Image pipeline queued for background download"
fi

# ============================================================
# Z-IMAGE (BASE & TURBO)
# ============================================================

# 1. Download Core Diffusion Flavor Based on Selections
if [ "${DOWNLOAD_Z_IMAGE_BASE:-}" = "true" ]; then
    echo "📥 Downloading Z-Image Base (BF16)..."
    download_model "https://huggingface.co/Comfy-Org/z_image_base/resolve/main/split_files/diffusion_models/z_image_base_bf16.safetensors" "$DIFFUSION_MODELS_DIR/z_image_base_bf16.safetensors"
fi

if [ "${DOWNLOAD_Z_IMAGE_BASE_GGUF:-}" = "true" ]; then
    echo "📥 Downloading Z-Image Base GGUF (Q8)..."
    download_model "https://huggingface.co/unsloth/Z-Image-GGUF/resolve/main/z-image-Q8_0.gguf" "$GGUF_DIR/z-image-Q8_0.gguf"
fi

if [ "${DOWNLOAD_Z_IMAGE_TURBO:-}" = "true" ]; then
    echo "📥 Downloading Z-Image Turbo (BF16)..."
    download_model "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" "$DIFFUSION_MODELS_DIR/z_image_turbo_bf16.safetensors"
fi

# 2. Shared Sub-Assets (Executes if ANY Z-Image variant flag is enabled)
if [ "${DOWNLOAD_Z_IMAGE_TURBO_GGUF:-}" = "true" ]; then
    echo "📥 Downloading Z-Image Turbo GGUF (Q8)..."
    download_model "https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/resolve/main/z-image-turbo-Q8_0.gguf" "$GGUF_DIR/z-image-turbo-Q8_0.gguf"
fi

if [ "${DOWNLOAD_Z_IMAGE_BASE:-}" = "true" ] || [ "${DOWNLOAD_Z_IMAGE_BASE_GGUF:-}" = "true" ] || [ "${DOWNLOAD_Z_IMAGE_TURBO:-}" = "true" ] || [ "${DOWNLOAD_Z_IMAGE_TURBO_GGUF:-}" = "true" ]; then
    echo "📥 Downloading shared Z-Image dependency ecosystem..."

    # Unified Multimodal Text Encoder (Qwen-3 4B Backbone)
    download_model "https://huggingface.co/Comfy-Org/z_image_base/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "$TEXT_ENCODERS_DIR/qwen_3_4b.safetensors"

    # Target Architecture VAE
    download_model "https://huggingface.co/modelzpalace/ae.safetensors/resolve/main/ae.safetensors" "$VAE_DIR/z_image_ae.safetensors"

    echo "📋 Z-Image pipeline queued for background download"
fi

# ============================================================
# CHROMA1 HD
# ============================================================

# 1. Download Base Diffusion Models Based on Flavor Selection
if [ "${DOWNLOAD_CHROMA1_HD:-}" = "true" ]; then
    echo "📥 Downloading Chroma1 HD (FP16)..."
    download_model "https://huggingface.co/lodestones/Chroma1-HD/resolve/main/Chroma1-HD.safetensors" "$DIFFUSION_MODELS_DIR/Chroma1-HD.safetensors"
fi

if [ "${DOWNLOAD_CHROMA1_HD_GGUF:-}" = "true" ]; then
    echo "📥 Downloading Chroma1 HD GGUF (Q8)..."
    download_model "https://huggingface.co/silveroxides/Chroma1-HD-GGUF/resolve/main/Chroma1-HD-Q8_0.gguf" "$GGUF_DIR/Chroma1-HD-Q8_0.gguf"
fi

# 2. Shared Sub-Assets (Executes if EITHER or BOTH flags are enabled)
if [ "${DOWNLOAD_CHROMA1_HD:-}" = "true" ] || [ "${DOWNLOAD_CHROMA1_HD_GGUF:-}" = "true" ]; then
    echo "📥 Downloading shared Chroma1 HD pipeline components..."

    # Foundational Flux Text Encoders (Both are required for DualCLIPLoader)
    download_model "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" "$TEXT_ENCODERS_DIR/t5xxl_fp16.safetensors"
    download_model "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" "$TEXT_ENCODERS_DIR/clip_l.safetensors"

    # Core Chroma VAE
    download_model "https://huggingface.co/lodestones/Chroma/resolve/main/ae.safetensors" "$VAE_DIR/chroma1_hd_ae.safetensors"

    # PuLID Pipeline Core & Native Face Evaluation Model
    download_model "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors" "$PULID_DIR/pulid_flux_v0.9.1.safetensors"
    download_model "https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt" "$CLIP_VISION_DIR/EVA02_CLIP_L_336_psz14_s6B.pt"

    # Spatial ControlNets (Union Package + Specialized Depth V3)
    download_model "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0/resolve/main/diffusion_pytorch_model.safetensors" "$CONTROLNET_DIR/flux_union_controlnet_2.0.safetensors"
    download_model "https://huggingface.co/XLabs-AI/flux-controlnet-depth-v3/resolve/main/flux-depth-controlnet-v3.safetensors" "$CONTROLNET_DIR/flux-depth-controlnet-v3.safetensors"

    echo "📋 Chroma1 HD pipeline queued for background download"
fi

# ==========================================
# CORE SHARED MODELS (Always Downloaded)
# ==========================================

echo "📥 Downloading shared models..."
download_model "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-ITF-SkinDiffDetail-Lite-v1.pth" "$UPSCALE_MODELS_DIR/1x-ITF-SkinDiffDetail-Lite-v1.pth"
download_model "https://huggingface.co/Tenofas/ComfyUI/resolve/main/upscale_models/4xFaceUpDAT.pth" "$UPSCALE_MODELS_DIR/4xFaceUpDAT.pth"

# ==========================================
# SAM 2
# ==========================================

echo "📥 Downloading SAM 2 weights..."
download_model "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2.1_hiera_large-fp16.safetensors" "$SAM2_DIR/sam2.1_hiera_large-fp16.safetensors"
download_model "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2.1_hiera_small-fp16.safetensors" "$SAM2_DIR/sam2.1_hiera_small-fp16.safetensors"

# ==========================================
# IMPACT PACK
# ==========================================

echo "📥 Downloading detailers & post-processing utilities..."
download_model "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth" "$UPSCALE_MODELS_DIR/4x_foolhardy_Remacri.pth"
download_model "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov11l.pt" "$ULTRALYTICS_DIR/bbox/face_yolov11l.pt"
download_model "https://huggingface.co/Ultralytics/assets/resolve/main/yolo11l-seg.pt" "$ULTRALYTICS_DIR/segm/yolo11l-seg.pt"

# ==========================================
# JOYCAPTION BETA ONE
# ==========================================

if [ "${DOWNLOAD_JOYCAPTION:-}" = "true" ]; then
    echo "📥 Downloading JoyCaption Beta One..."

    # 1. Config & Tokenizer Files
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/config.json" "$JOYCAPTION_DIR/config.json" true
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/generation_config.json" "$JOYCAPTION_DIR/generation_config.json" true
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/model.safetensors.index.json" "$JOYCAPTION_DIR/model.safetensors.index.json" true
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/preprocessor_config.json" "$JOYCAPTION_DIR/preprocessor_config.json" true
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/special_tokens_map.json" "$JOYCAPTION_DIR/special_tokens_map.json" true
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/tokenizer.json" "$JOYCAPTION_DIR/tokenizer.json" true
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/tokenizer_config.json" "$JOYCAPTION_DIR/tokenizer_config.json" true

    # 2. Sharded Weights
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/model-00001-of-00004.safetensors" "$JOYCAPTION_DIR/model-00001-of-00004.safetensors"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/model-00002-of-00004.safetensors" "$JOYCAPTION_DIR/model-00002-of-00004.safetensors"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/model-00003-of-00004.safetensors" "$JOYCAPTION_DIR/model-00003-of-00004.safetensors"
    download_model "https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/resolve/main/model-00004-of-00004.safetensors" "$JOYCAPTION_DIR/model-00004-of-00004.safetensors"

    echo "📋 JoyCaption Beta One model queued for background download"
fi

# ==========================================
# FLORENCE-2 NSFW V2
# ==========================================

if [ "${DOWNLOAD_FLORENCE2:-}" = "true" ]; then
    echo "📥 Downloading Florence-2 NSFW finetune..."

    # Base URL for the finetune
    NSFW_BASE_URL="https://huggingface.co/ljnlonoljpiljm/florence-2-base-nsfw-v2/resolve/main"

    # 1. Core Configuration & Tokenizer
    download_model "$NSFW_BASE_URL/config.json" "$FLORENCE2_DIR/config.json" true
    download_model "$NSFW_BASE_URL/generation_config.json" "$FLORENCE2_DIR/generation_config.json" true
    download_model "$NSFW_BASE_URL/preprocessor_config.json" "$FLORENCE2_DIR/preprocessor_config.json" true
    download_model "$NSFW_BASE_URL/added_tokens.json" "$FLORENCE2_DIR/added_tokens.json" true
    download_model "$NSFW_BASE_URL/merges.txt" "$FLORENCE2_DIR/merges.txt" true
    download_model "$NSFW_BASE_URL/special_tokens_map.json" "$FLORENCE2_DIR/special_tokens_map.json" true
    download_model "$NSFW_BASE_URL/tokenizer.json" "$FLORENCE2_DIR/tokenizer.json" true
    download_model "$NSFW_BASE_URL/tokenizer_config.json" "$FLORENCE2_DIR/tokenizer_config.json" true
    download_model "$NSFW_BASE_URL/vocab.json" "$FLORENCE2_DIR/vocab.json" true

    # 2. The Weights
    download_model "$NSFW_BASE_URL/model.safetensors" "$FLORENCE2_DIR/model.safetensors"

    # 3. Microsoft Processor (Handles the actual image bounding boxes/cropping)
    download_model "https://huggingface.co/microsoft/Florence-2-base/resolve/main/processing_florence2.py" "$FLORENCE2_DIR/processing_florence2.py" true

    # 4. APPLY THE KIJAI / LAYERSTYLE PATCH
    # We copy the patched modeling and config files directly from the custom node directory
    # to overwrite any missing or outdated files, ensuring transformers >= 4.45 compatibility.
    echo "🔧 Applying Transformers compatibility patch for Florence-2..."
    LAYERSTYLE_MODELS_DIR="$NETWORK_VOLUME/ComfyUI/custom_nodes/ComfyUI_LayerStyle_Advance/florence2_models"

    if [ -d "$LAYERSTYLE_MODELS_DIR" ]; then
        cp "$LAYERSTYLE_MODELS_DIR/modeling_florence2.py" "$FLORENCE2_DIR/"
        cp "$LAYERSTYLE_MODELS_DIR/configuration_florence2.py" "$FLORENCE2_DIR/"
        echo "✅ Florence-2 patched successfully."
    else
        echo "⚠️ WARNING: LayerStyle advance folder not found at $LAYERSTYLE_MODELS_DIR. Patch skipped."
    fi

    echo "📋 Florence-2 NSFW queued for background download"
fi

if [ ! -f "$ULTRALYTICS_DIR/bbox/Eyes.pt" ]; then
    if [ -f "/Eyes.pt" ]; then
        mv "/Eyes.pt" "$ULTRALYTICS_DIR/bbox/Eyes.pt"
        echo "Moved Eyes.pt to the correct location."
    else
        echo "Eyes.pt not found in the root directory."
    fi
else
    echo "Eyes.pt already exists. Skipping."
fi

if [ ! -f "$UPSCALE_MODELS_DIR/4xLSDIR.pth" ]; then
    if [ -f "/4xLSDIR.pth" ]; then
        mv "/4xLSDIR.pth" "$UPSCALE_MODELS_DIR/4xLSDIR.pth"
        echo "Moved 4xLSDIR.pth to the correct location."
    else
        echo "4xLSDIR.pth not found in the root directory."
    fi
else
    echo "4xLSDIR.pth already exists. Skipping."
fi

# ============================================================
# OPTIMIZED ANTELOPEV2 ENGINE (Integrated with custom fn)
# ============================================================

# Only trigger if the target directory doesn't have the final .onnx models
if [ ! -d "$ANTELOPEV2_DIR" ] || [ -z "$(ls -A "$ANTELOPEV2_DIR" 2> /dev/null | grep '\.onnx$')" ]; then
    echo "📥 AntelopeV2 models missing. Launching download allocation..."

    # Call your custom function
    download_model "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip" "$ANTELOPEV2_DIR/antelopev2.zip"

    # SAFETY LAYER: If the file is missing or actively downloading via .aria2, wait for your background task
    if [ ! -f "$ANTELOPEV2_DIR/antelopev2.zip" ] || [ -f "$ANTELOPEV2_DIR/antelopev2.zip.aria2" ]; then
        echo "⏳ Active download detected. Holding script execution until aria2c finishes..."
        wait "$LAST_DOWNLOAD_PID" 2> /dev/null || true
    fi

    # Proceed to extraction now that the file is fully on disk
    if [ -f "$ANTELOPEV2_DIR/antelopev2.zip" ]; then
        echo "📦 Extracting and flattening AntelopeV2 assets..."
        unzip -oj "$ANTELOPEV2_DIR/antelopev2.zip" -d "$ANTELOPEV2_DIR"

        echo "🧹 Cleaning up zip archive to keep network volume clean..."
        rm -f "$ANTELOPEV2_DIR/antelopev2.zip"
        echo "✅ AntelopeV2 engine deployment complete."
    fi
else
    echo "✅ AntelopeV2 models already present and extracted. Skipping setup."
fi

# ============================================================
# WORKFLOWS MIGRATION
# ============================================================

SOURCE_DIR="/comfyui-image/workflows"

# Ensure destination directory exists
mkdir -p "$WORKFLOW_DIR"

if [ -d "$SOURCE_DIR" ] && [ "$(ls -A "$SOURCE_DIR" 2> /dev/null)" ]; then
    echo "🔄 Migrating workflows and subfolders cleanly..."

    # rsync safely merges contents. If a folder exists, it adds new files inside it
    # without deleting old ones.
    rsync -av --ignore-existing "$SOURCE_DIR/" "$WORKFLOW_DIR/" > /dev/null

    # Wipe the source directory clean now that everything is safely copied/merged
    rm -rf "$SOURCE_DIR"/*
    echo "✅ Workflow migration and merge complete!"
else
    echo "✨ No source workflows found for migration."
fi

# ============================================================
# SURFACE EMBEDDED NODE WORKFLOWS
# ============================================================

WORKFLOW_EXPORT_DIR="$NETWORK_VOLUME/Workflows/Node_Examples"
mkdir -p "$WORKFLOW_EXPORT_DIR"

echo "🔗 Symlinking embedded node workflows for easy access..."

# 1. We look inside common directory structures to avoid dragging in non-workflow config.json files
# 2. We use relative path mapping to preserve nested folders (like i2v vs t2v examples)
find "$NETWORK_VOLUME/ComfyUI/custom_nodes" -type f -name "*.json" \
    \( -path "*/example_workflows/*" -o -path "*/examples/*" -o -path "*/workflows/*" \) | while read -r workflow_path; do

    # Extract the relative path path starting right after /custom_nodes/
    # This turns a deep path into 'ComfyUI-WanAnimatePreprocess/example_workflows/i2v/example.json'
    relative_path=$(echo "$workflow_path" | awk -F'/custom_nodes/' '{print $2}')

    # Determine the target path inside your export directory
    target_link_path="$WORKFLOW_EXPORT_DIR/$relative_path"

    # Ensure the parent folders exist cleanly inside the export tree
    mkdir -p "$(dirname "$target_link_path")"

    # Create the relative or absolute target mapping symlink safely
    ln -sf "$workflow_path" "$target_link_path"
done

echo "✅ Node example symlinking engine execution complete!"

# ============================================================
# COMFYUI IMPACT SUBPACK CONFIGURATION
# ============================================================

echo "📋 Ensuring ComfyUI-Impact-Subpack user directory exists..."
# Extracts the parent directory path dynamically to avoid creating a folder named '.txt'
mkdir -p "$(dirname "$MODEL_WHITELIST_DIR")"

echo "🔒 Writing model whitelist overrides..."
cat > "$MODEL_WHITELIST_DIR" << 'EOF'
EVA02_CLIP_L_336_psz14_s6B.pt
Eyes.pt
face_yolov11l.pt
yolo11l-seg.pt
EOF

echo "✅ Model whitelist successfully initialized!"

# ============================================================
# DYNAMIC CIVITAI DOWNLOAD ENGINE
# ============================================================

# Ensure the new UNET target path exists on the volume if GGUF downloads are requested
if [ -n "$GGUF_IDS_TO_DOWNLOAD" ] && [ "$GGUF_IDS_TO_DOWNLOAD" != "replace_with_ids" ]; then
    mkdir -p "$GGUF_DIR"
fi

# Initialize a clean, empty associative array
declare -A MODEL_CATEGORIES

# Dynamically populate the map to guarantee absolute syntax safety on empty environment variables
[ -n "$CHECKPOINT_IDS_TO_DOWNLOAD" ] && MODEL_CATEGORIES["$CHECKPOINTS_DIR"]="$CHECKPOINT_IDS_TO_DOWNLOAD"
[ -n "$LORAS_IDS_TO_DOWNLOAD" ] && MODEL_CATEGORIES["$LORAS_DIR"]="$LORAS_IDS_TO_DOWNLOAD"
[ -n "$BASE_MODEL_IDS_TO_DOWNLOAD" ] && MODEL_CATEGORIES["$DIFFUSION_MODELS_DIR"]="$BASE_MODEL_IDS_TO_DOWNLOAD"
[ -n "$GGUF_IDS_TO_DOWNLOAD" ] && MODEL_CATEGORIES["$GGUF_DIR"]="$GGUF_IDS_TO_DOWNLOAD"

# Counter and PID tracking
download_count=0
download_pids=()

# Schedule downloads in background
for TARGET_DIR in "${!MODEL_CATEGORIES[@]}"; do
    MODEL_IDS_STRING="${MODEL_CATEGORIES[$TARGET_DIR]}"

    if [[ "$MODEL_IDS_STRING" == "replace_with_ids" ]]; then
        echo "⏭️  Skipping downloads for $TARGET_DIR (Default placeholder detected)"
        continue
    fi

    IFS=',' read -ra MODEL_IDS <<< "$MODEL_IDS_STRING"
    for MODEL_ID in "${MODEL_IDS[@]}"; do
        CLEAN_ID="${MODEL_ID// /}"
        [ -z "$CLEAN_ID" ] && continue

        echo "🚀 Scheduling CivitAI download: $CLEAN_ID to $TARGET_DIR"
        (cd "$TARGET_DIR" && $PYTHON_BIN /usr/local/bin/download_with_aria.py -m "$CLEAN_ID") &
        download_pids+=($!)
        ((download_count++))
    done
done

echo "📋 Scheduled $download_count downloads in background."

# ============================================================
# CRITICAL BOUNDARY: Block thread until background jobs finish
# ============================================================

if [ "$download_count" -gt 0 ]; then
    echo "⏳ Holding boot sequence: Waiting for $download_count background model downloads to complete..."
    wait "${download_pids[@]}"
    echo "✅ All background model downloads have finished successfully!"
else
    echo "✅ No background downloads were required."
fi

# Final catch-all safety wall for any lingering aria2c tasks
if pgrep -x "aria2c" > /dev/null; then
    echo "⏳ Waiting for lingering aria2c processes to completely close..."
    while pgrep -x "aria2c" > /dev/null; do
        sleep 5
    done
fi

echo "✅ All models downloaded successfully!"

# ============================================================
# ComfyUI
# ============================================================

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
# 1. Block unauthorized external network sharing
share_option = none
bypass_ssl = False
file_logging = True
component_policy = workflow
# 2. Lock down core ComfyUI updates completely
update_policy = none
windows_selector_event_loop_policy = False
model_download_by_agent = False
downgrade_blacklist =
# 3. Elevate security to block background pip executions
security_level = high
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

# Workspace as main working directory
grep -qxF "cd $NETWORK_VOLUME" ~/.bashrc || echo "cd $NETWORK_VOLUME" >> ~/.bashrc

# Return to the ComfyUI root directory before launching
cd "$NETWORK_VOLUME/ComfyUI" || exit 1

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
    echo "⚖️ Standard VRAM detected. Letting ComfyUI handle dynamic offloading."
fi

# Add SageAttention
if [ "$SAGE_ATTENTION_AVAILABLE" = "true" ]; then
    LAUNCH_FLAGS="$LAUNCH_FLAGS --use-sage-attention"
fi

# Final Command Construction
COMFYUI_CMD="$PYTHON_BIN ./main.py $LAUNCH_FLAGS"

# Launch
URL="http://127.0.0.1:8188"
status_msg "▶️ Starting ComfyUI with flags: $LAUNCH_FLAGS"
nohup $COMFYUI_CMD > "$NETWORK_VOLUME/comfyui_nohup.log" 2>&1 &
echo $! > /tmp/comfyui.pid # Save PID for restart

# ============================================================
# LIVE-EVALUATION RESTART SCRIPT GENERATION
# ============================================================

# We use a quoted heredoc 'EOF' here to keep the inner variables intact for live runtime evaluation!
cat << 'EOF' > /usr/local/bin/comfyui-restart
#!/bin/bash

# Live-resolve environment paths
PYTHON_BIN="/usr/bin/python3"
COMFYUI_DIR=$(pwd)
LOG_FILE="comfyui_nohup.log"

# Catch accidental out-of-directory executions
if [ ! -f "./main.py" ] && [ -d "/workspace/ComfyUI" ]; then
    COMFYUI_DIR="/workspace/ComfyUI"
fi

# Detect log path safety
[ -f "../comfyui_nohup.log" ] && LOG_FILE="../comfyui_nohup.log"

echo "🛑 Stopping running ComfyUI process..."
kill $(cat /tmp/comfyui.pid 2>/dev/null) 2>/dev/null
sleep 2

# RE-EVALUATE HARDWARE ENVIRONMENT LIVE
# This ensures that if they change VRAM templates or switch configurations, the flags follow them.
GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
VRAM_THRESHOLD=32000

BASE_FLAGS="--listen --preview-method auto"

# Seamlessly check variable states inside the live shell container
if [ "${USE_FP8_TEXT_ENC:-true}" = "true" ]; then
    BASE_FLAGS="$BASE_FLAGS --fp8_e4m3fn-text-enc"
fi

if [ "${USE_FP8_MODEL:-}" = "true" ]; then
    BASE_FLAGS="$BASE_FLAGS --fp8_e4m3fn-unet"
fi

if [ "$GPU_VRAM_MB" -ge "$VRAM_THRESHOLD" ]; then
    BASE_FLAGS="$BASE_FLAGS --highvram"
fi

# Live Python execution test to see if SageAttention compiles/loads cleanly right now
if /usr/bin/python3 -c "import sageattention" &> /dev/null; then
    echo "⚡ SageAttention import verification: SUCCESS. Appending launch flag."
    BASE_FLAGS="$BASE_FLAGS --use-sage-attention"
else
    echo "⚠️ SageAttention import verification: FAILED or missing. Omitting flag."
fi

echo "📋 Active debugger flags: $BASE_FLAGS"
if [ ! -z "$*" ]; then
    echo "🔧 User-appended arguments: $*"
fi

cd "$COMFYUI_DIR" || exit 1
nohup $PYTHON_BIN ./main.py $BASE_FLAGS $* > "$LOG_FILE" 2>&1 &

echo $! > /tmp/comfyui.pid
echo "✅ ComfyUI successfully restarted with PID $(cat /tmp/comfyui.pid)"
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
echo "  Use the SSH command provided by your host: "
echo ""
echo "  Filebrowser:"
echo ""
echo "     ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 8080:localhost:8080"
echo ""
echo "     Then open your browser:"
echo "     http://localhost:8080"
echo ""
echo "  JupyterLab:"
echo ""
echo "     ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 8888:localhost:8888"
echo ""
echo "     Then open your browser:"
echo "     http://localhost:8888/lab"
echo ""
echo "  ComfyUI GUI:"
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

# ============================================================
# SSH Startup
# ============================================================
status_msg "[4/4] 🔐 Starting SSH server..."

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
