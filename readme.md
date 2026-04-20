# ComfyUI Qwen2512, Edit-2511 & Z-Image Turbo/Base w/ Sage Attention for CUDA 12.8

### Variables Selection

Z-Image Base and Qwen 2512 download by default if you don't want them set the variables to `false`
the rest need to be toggled by setting them to `true`.

```env
DOWNLOAD_QWEN_FULL=""
DOWNLOAD_QWEN_EDIT_2511=""
DOWNLOAD_QWEN_2512=""
DOWNLOAD_Z_IMAGE_BASE=""
DOWNLOAD_Z_IMAGE_TURBO=""
```

### Auth Tokens

```env
CIVITAI_TOKEN=""
SSH_PUBLIC_KEY=""
```

### Ports

| Port | Service  |
|------|----------|
| 8188 | ComfyUI  |
| 8888 | Jupyter  |
| 22   | SSH      |

### Accessing the Instance

```bash 
If you are using custom SSH key location you might want to create a config file in
~/.ssh/config for Linux or $HOME\.ssh\config for Windows.
```
Linux:
```bash
Host *
    IdentityFile PATH/.ssh/id_ed25519
    IdentitiesOnly yes
```
Windows:
```bash 
Host *
    IdentityFile PATH\.ssh\id_ed25519
    IdentitiesOnly yes
```

You can transfer files using `rsync` and connect via SSH:

```bash
# Example: sync local dataset to remote
rsync -avP -e "ssh -p <SSH_PORT>" /path/to/local/dataset/ hostname@<SERVER_IP>:/path/to/remote/dataset/

# SSH with port forwarding for JupyterLab
ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 8888:localhost:8888
```
Then open your browser to:
```bash
http://localhost:8888/lab
```

#### Accessing ComfyUI GUI

ComfyUI runs its interface on port `8188`. To access it from your local browser, use SSH port forwarding:

```bash
# SSH with port forwarding for ComfyUI
ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 8188:localhost:8188
```
Then open your browser to:
```bash
http://localhost:8188
```
---

# Civitai Downloader

### 📖 Usage

Download a model using its ID:

```bash
./download_with_aria.py -m 123456

# Download to specific directory
./download_with_aria.py -m 123456 -o ./models

# Use custom filename
./download_with_aria.py -m 123456 --filename "my_custom_model.safetensors"

# Force re-download (ignore existing files)
./download_with_aria.py -m 123456 --force

# Provide token via command line (not recommended for security)
./download_with_aria.py -m 123456 --token "your_token_here"
```

### Command Line Arguments

| Argument     | Short | Description                          |
|--------------|-------|--------------------------------------|
| `--model-id` | `-m`  | CivitAI model version ID (required)  |
| `--output`   | `-o`  | Output directory                     |
| `--token`    | —     | CivitAI API token                    |
| `--filename` | —     | Override default filename            |
| `--force`    | —     | Force re-download                    |

---

### 🎯 Examples

**Download a LoRA model:**

```bash
./download_with_aria.py -m 245589

# Download character LoRA
./download_with_aria.py -m 245589 -o ./models/lora/characters

# Download style LoRA
./download_with_aria.py -m 234567 -o ./models/lora/styles

# Download checkpoint
./download_with_aria.py -m 345678 -o ./models/checkpoints
```

**Batch download with a simple script:**

```bash
#!/bin/bash
# download_batch.sh
models=(245589 234567 345678 456789)
for model_id in "${models[@]}"; do
    ./download_with_aria.py -m "$model_id" -o ./models
done
```
