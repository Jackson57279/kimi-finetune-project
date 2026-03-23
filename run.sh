#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"; }
warning() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1"; }

print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          Kimi K2.5 Fine-Tuning Orchestrator                  ║"
    echo "║              CPU-Optimized | 2.5TB Storage                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
}

show_help() {
    print_banner
    echo "Usage: ./run.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup       - Install dependencies and setup environment"
    echo "  download    - Download Kimi K2.5 model (~600GB)"
    echo "  convert     - Convert to GGUF formats (F16, Q4, Q5, Q8)"
    echo "  prepare     - Prepare massive training datasets"
    echo "  train       - Start LoRA fine-tuning"
    echo "  inference   - Run inference with trained model"
    echo "  all         - Run complete pipeline (setup → train)"
    echo "  status      - Show storage and training status"
    echo "  clean       - Clean temporary files"
    echo ""
    echo "Examples:"
    echo "  ./run.sh setup"
    echo "  ./run.sh download"
    echo "  ./run.sh prepare"
    echo "  ./run.sh train"
    echo "  ./run.sh inference --model models/quantized/kimi-k2.5-q4_k_m.gguf --prompt 'Hello'"
    echo ""
}

cmd_setup() {
    log "Running setup..."
    bash "${SCRIPT_DIR}/scripts/setup.sh"
}

cmd_download() {
    log "Downloading model..."
    bash "${SCRIPT_DIR}/scripts/download.sh"
}

cmd_convert() {
    log "Converting model..."
    bash "${SCRIPT_DIR}/scripts/convert.sh"
}

cmd_prepare() {
    log "Preparing datasets..."
    source "${PROJECT_ROOT}/activate.sh"
    python3 "${PROJECT_ROOT}/src/prepare_data.py" --config "${PROJECT_ROOT}/configs/config.yaml"
}

cmd_train() {
    log "Starting training..."
    source "${PROJECT_ROOT}/activate.sh"
    python3 "${PROJECT_ROOT}/src/train.py" --config "${PROJECT_ROOT}/configs/config.yaml"
}

cmd_inference() {
    source "${PROJECT_ROOT}/activate.sh"
    python3 "${PROJECT_ROOT}/src/inference.py" "$@"
}

cmd_status() {
    print_banner
    
    if [[ -f "${PROJECT_ROOT}/activate.sh" ]]; then
        success "Environment: Configured"
    else
        error "Environment: Not configured"
    fi
    
    if [[ -d "${PROJECT_ROOT}/models/original" ]]; then
        MODEL_SIZE=$(du -sh "${PROJECT_ROOT}/models/original" | awk '{print $1}')
        success "Original Model: ${MODEL_SIZE}"
    else
        warning "Original Model: Not downloaded"
    fi
    
    if [[ -d "${PROJECT_ROOT}/models/quantized" ]]; then
        QTY=$(ls -1 "${PROJECT_ROOT}/models/quantized" 2>/dev/null | wc -l)
        success "Quantized Models: ${QTY} files"
    else
        warning "Quantized Models: None"
    fi
    
    if [[ -f "${PROJECT_ROOT}/data/processed/training_data.txt" ]]; then
        SAMPLES=$(wc -l < "${PROJECT_ROOT}/data/processed/training_data.txt")
        success "Training Data: ${SAMPLES} samples"
    else
        warning "Training Data: Not prepared"
    fi
    
    echo ""
    log "Storage Usage:"
    df -h "${PROJECT_ROOT}" | tail -1 | awk '{print "  Total: " $2 " | Used: " $3 " | Available: " $4 " | Usage: " $5}'
}

cmd_clean() {
    log "Cleaning temporary files..."
    
    rm -rf "${PROJECT_ROOT}/cache"/* 2>/dev/null || true
    rm -rf "${PROJECT_ROOT}/logs"/*.log 2>/dev/null || true
    find "${PROJECT_ROOT}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    success "Cleanup complete"
}

cmd_all() {
    print_banner
    
    log "This will run the complete pipeline (skipping setup - already done)"
    log "Estimated time: Days to weeks on CPU"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
    
    cmd_download
    cmd_convert
    cmd_prepare
    cmd_train
    
    success "Pipeline complete!"
}

main() {
    print_banner
    
    case "${1:-}" in
        setup)
            cmd_setup
            ;;
        download)
            cmd_download
            ;;
        convert)
            cmd_convert
            ;;
        prepare)
            cmd_prepare
            ;;
        train)
            cmd_train
            ;;
        inference)
            shift
            cmd_inference "$@"
            ;;
        status)
            cmd_status
            ;;
        clean)
            cmd_clean
            ;;
        all)
            cmd_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: ${1:-}"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
