#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
source "${PROJECT_ROOT}/activate.sh" 2>/dev/null || {
    echo "Please run setup.sh first"
    exit 1
}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_success() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"; }
log_warning() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"; }
log_error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1"; }

MODEL_REPO="moonshotai/Kimi-K2.5"
MODEL_DIR="${PROJECT_ROOT}/models/original"
LOCK_FILE="${MODEL_DIR}/.download.lock"
RESUME_FILE="${MODEL_DIR}/.resume"

cleanup() {
    if [[ -f "$LOCK_FILE" ]]; then
        rm -f "$LOCK_FILE"
        log "Cleanup: removed lock file"
    fi
}
trap cleanup EXIT

wait_for_space() {
    local required_gb=$1
    while true; do
        AVAILABLE_GB=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | tr -d 'G')
        if [[ $AVAILABLE_GB -ge $required_gb ]]; then
            return 0
        fi
        log_warning "Insufficient space. Required: ${required_gb}GB, Available: ${AVAILABLE_GB}GB"
        log "Waiting 60 seconds before retry..."
        sleep 60
    done
}

check_existing() {
    if [[ -d "$MODEL_DIR" ]]; then
        EXISTING_SIZE=$(du -sb "$MODEL_DIR" 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
        log "Found existing model directory: ${EXISTING_SIZE}GB"
        
        if [[ -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
            TOTAL_FILES=$(find "$MODEL_DIR" -name "*.safetensors" | wc -l)
            log "Found $TOTAL_FILES model files"
            
            if [[ $TOTAL_FILES -ge 64 ]]; then
                log_success "Model appears to be complete"
                read -p "Redownload? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    log "Skipping download"
                    exit 0
                fi
            fi
        fi
        
        log "Resuming download..."
        return 0
    fi
    
    return 1
}

download_with_hf_cli() {
    log "Starting model download with uvx hf CLI..."
    log "Model: $MODEL_REPO"
    log "Destination: $MODEL_DIR"
    log "Expected size: ~600GB (64 safetensors files)"
    log "This will take several hours..."
    log ""
    
    mkdir -p "$MODEL_DIR"
    touch "$LOCK_FILE"
    
    wait_for_space 700
    
    export HF_HUB_ENABLE_HF_TRANSFER=1
    
    log "Starting download with: uvx hf model download"
    log "Progress will be shown below:"
    log "=========================================="
    
    uvx hf model download "$MODEL_REPO" \
        --local-dir "$MODEL_DIR" \
        --resume-download 2>&1 | while read line; do
            echo "$line"
            echo "$(date '+%Y-%m-%d %H:%M:%S') $line" >> "${PROJECT_ROOT}/logs/download.log"
        done
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    log "=========================================="
    
    if [[ $EXIT_CODE -ne 0 ]]; then
        log_error "Download failed with exit code $EXIT_CODE"
        exit 1
    fi
    
    log_success "Download complete"
}

download_with_python() {
    log "Starting model download with Python API..."
    
    python3 << 'EOF'
import os
import sys
from huggingface_hub import snapshot_download
from pathlib import Path

project_root = os.environ.get('PROJECT_ROOT', '/mnt/external2.5tb/kimi-finetune')
model_dir = Path(project_root) / 'models' / 'original'
model_dir.mkdir(parents=True, exist_ok=True)

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

try:
    snapshot_download(
        repo_id="moonshotai/Kimi-K2.5",
        local_dir=str(model_dir),
        resume_download=True,
        local_dir_use_symlinks=False,
        max_workers=4
    )
    print("✓ Download complete")
except Exception as e:
    print(f"✗ Download failed: {e}")
    sys.exit(1)
EOF
}

verify_download() {
    log "Verifying downloaded model..."
    
    if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
        log_error "Model index file not found"
        return 1
    fi
    
    TOTAL_SIZE=$(du -sb "$MODEL_DIR" | awk '{print $1}')
    TOTAL_SIZE_GB=$((TOTAL_SIZE / 1024 / 1024 / 1024))
    
    log "Total downloaded size: ${TOTAL_SIZE_GB}GB"
    
    if [[ $TOTAL_SIZE_GB -lt 500 ]]; then
        log_warning "Model size seems small (${TOTAL_SIZE_GB}GB). Expected ~600GB"
        return 1
    fi
    
    FILE_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" | wc -l)
    log "Found $FILE_COUNT safetensors files"
    
    if [[ $FILE_COUNT -lt 60 ]]; then
        log_warning "Expected ~64 safetensors files, found $FILE_COUNT"
        return 1
    fi
    
    log_success "Model verification passed"
    return 0
}

create_model_info() {
    cat > "${MODEL_DIR}/model_info.txt" << EOF
Model: Kimi K2.5
Repository: $MODEL_REPO
Downloaded: $(date)
Location: $MODEL_DIR
Size: $(du -sh "$MODEL_DIR" | awk '{print $1}')
EOF
    
    log "Model info saved"
}

main() {
    log "Kimi K2.5 Model Download"
    log "========================"
    
    if [[ -f "$LOCK_FILE" ]]; then
        log_warning "Another download process is running or previous download was interrupted"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    if check_existing; then
        log "Resuming previous download..."
    fi
    
    log "Using uvx to run hf CLI..."
    download_with_hf_cli
    
    verify_download || {
        log_error "Verification failed. You may need to re-run the download"
        exit 1
    }
    
    create_model_info
    
    log_success "Model download complete!"
    log "Location: $MODEL_DIR"
    log "Size: $(du -sh "$MODEL_DIR" | awk '{print $1}')"
    log ""
    log "Next step: Run ./scripts/convert.sh to convert to GGUF format"
}

main "$@"
