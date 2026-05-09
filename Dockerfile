# Single-stage build to ensure C++ compilation tools remain available for custom nodes
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Consolidated environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    PATH="/opt/venv/bin:$PATH"

# 1. System Dependencies, SSH Setup
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip \
        curl unzip ffmpeg ninja-build git aria2 git-lfs wget vim rsync \
        libgl1 libglib2.0-0 libgoogle-perftools4 build-essential gcc openssh-server && \
    \
    # Setup defaults
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python3 -m venv /opt/venv && \
    \
    # Surgical SSH Config (applies changes whether commented or active)
    mkdir -p /root/.ssh /var/run/sshd && \
    chmod 700 /root/.ssh && \
    sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Stable PyTorch 2.9.1 Stack
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        torch==2.9.1+cu128 \
        torchvision==0.24.1+cu128 \
        torchaudio==2.9.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# 3. Core Tooling & Critical ML Libraries
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    packaging \
    setuptools \
    wheel \
    triton==3.5.1 \
    accelerate \
    "transformers>=4.48.0" \
    diffusers \
    peft \
    sentencepiece \
    einops \
    scipy \
    timm \
    gguf \
    bitsandbytes \
    protobuf

# 4. Runtime libraries & Comfy-CLI
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    pyyaml \
    comfy-cli \
    jupyterlab \
    jupyterlab-lsp \
    jupyter-server \
    jupyter-server-terminals \
    ipykernel \
    jupyterlab_code_formatter \
    opencv-contrib-python-headless \
    "qwen-vl-utils>=0.0.8" \
    onnxruntime-gpu \
    ultralytics \
    segment-anything \
    transparent-background

RUN curl -fsSL https://rclone.org/install.sh -o /tmp/rclone_install.sh && \
    bash /tmp/rclone_install.sh && \
    rm /tmp/rclone_install.sh

# Establishing workspace
WORKDIR /workspace

# ------------------------------------------------------------
# ComfyUI & Custom Nodes install (CircleCI Heartbeat Version)
# ------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    # 1. Force create directory and silence the analytics prompt
    mkdir -p /ComfyUI/custom_nodes && \
    comfy --workspace /ComfyUI install --non-interactive --yes && \
    set -e; \
    # 2. Move into the directory once
    cd /ComfyUI/custom_nodes; \
    for repo in \
        https://github.com/city96/ComfyUI-GGUF.git \
        https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git \
        https://github.com/kijai/ComfyUI-KJNodes.git \
        https://github.com/rgthree/rgthree-comfy.git \
        https://github.com/spacepxl/ComfyUI-VAE-Utils.git \
        https://github.com/obisin/ComfyUI-FSampler.git \
        https://github.com/JPS-GER/ComfyUI_JPS-Nodes.git \
        https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git \
        https://github.com/Jordach/comfy-plasma.git \
        https://github.com/ltdrdata/ComfyUI-Impact-Pack.git \
        https://github.com/ClownsharkBatwing/RES4LYF.git \
        https://github.com/yolain/ComfyUI-Easy-Use.git \
        https://github.com/WASasquatch/was-node-suite-comfyui.git \
        https://github.com/theUpsider/ComfyUI-Logic.git \
        https://github.com/cubiq/ComfyUI_essentials.git \
        https://github.com/chrisgoringe/cg-image-picker.git \
        https://github.com/chflame163/ComfyUI_LayerStyle.git \
        https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git \
        https://github.com/Jonseed/ComfyUI-Detail-Daemon.git \
        https://github.com/shadowcz007/comfyui-mixlab-nodes.git \
        https://github.com/chflame163/ComfyUI_LayerStyle_Advance.git \
        https://github.com/bash-j/mikey_nodes.git \
        https://github.com/chrisgoringe/cg-use-everywhere.git \
        https://github.com/PGCRT/CRT-Nodes.git \
        https://github.com/Fannovel16/comfyui_controlnet_aux.git \
        https://github.com/M1kep/ComfyLiterals.git; \
    do \
        repo_dir=$(basename "$repo" .git); \
        echo "CIRCLECI_HEARTBEAT: Installing $repo_dir into $(pwd)..."; \
        \
        # Clone with depth 1
        if [ "$repo" = "https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git" ]; then \
            git clone --depth 1 --recursive "$repo"; \
        else \
            git clone --depth 1 "$repo"; \
        fi; \
        \
        # 4. Harmonize and Install Requirements
        if [ -f "$repo_dir/requirements.txt" ]; then \
            echo "🛠️ Harmonizing Dependencies for $repo_dir..."; \
            \
            # 1. Harmonize OpenCV
            sed -i -E 's/opencv-(python|contrib-python)(-headless)?(==[0-9.]+)?/opencv-contrib-python-headless/g' "$repo_dir/requirements.txt"; \
            \
            # 2. Harmonize bitsandbytes (Strips versions like ==0.41.1 or >=0.35)
            sed -i -E 's/bitsandbytes([>=<~= ]+[0-9.]+)?/bitsandbytes/g' "$repo_dir/requirements.txt"; \
            \
            # 3. Harmonize protobuf
            sed -i -E 's/protobuf([>=<~= ]+[0-9.]+)?/protobuf/g' "$repo_dir/requirements.txt"; \
            \
            # 4. Harmonize onnxruntime
            sed -i -E 's/^onnxruntime$/onnxruntime-gpu/g' "$repo_dir/requirements.txt"; \
            \
            pip install --progress-bar off -v -r "$repo_dir/requirements.txt"; \
        fi; \
        \
        # 5. Run install.py if it exists
        if [ -f "$repo_dir/install.py" ]; then \
            python "$repo_dir/install.py"; \
        fi; \
    done

# 6. Final Assets & Entrypoint
COPY src/start_script.sh /start_script.sh
COPY docker-entrypoint.sh /docker-entrypoint.sh
COPY Eyes.pt /Eyes.pt
COPY 4xLSDIR.pth /4xLSDIR.pth

RUN chmod +x /start_script.sh /docker-entrypoint.sh

# Fix for JoyCaption / Protobuf compatibility
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/start_script.sh"]