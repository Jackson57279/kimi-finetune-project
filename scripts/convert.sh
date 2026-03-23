#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
source "${PROJECT_ROOT}/activate.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_success() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"; }
log_warning() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"; }
log_error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1"; }

MODEL_DIR="${PROJECT_ROOT}/models/original"
CONVERTED_DIR="${PROJECT_ROOT}/models/converted"
QUANTIZED_DIR="${PROJECT_ROOT}/models/quantized"
LLAMA_CPP="${PROJECT_ROOT}/llama.cpp"

mkdir -p "$CONVERTED_DIR" "$QUANTIZED_DIR"

check_space() {
    local required=$1
    AVAILABLE=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | tr -d 'G')
    if [[ $AVAILABLE -lt $required ]]; then
        log_error "Insufficient space. Need ${required}GB, have ${AVAILABLE}GB"
        return 1
    fi
    return 0
}

convert_to_f16() {
    log "Converting to F16 format..."
    
    check_space 1300 || exit 1
    
    if [[ -f "${CONVERTED_DIR}/kimi-k2.5-f16.gguf" ]]; then
        log_warning "F16 model already exists"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Skipping F16 conversion"
            return 0
        fi
    fi
    
    local convert_script="${LLAMA_CPP}/convert_hf_to_gguf.py"
    if [[ ! -f "$convert_script" ]]; then
        convert_script="${LLAMA_CPP}/convert.py"
    fi
    
    log "Running conversion..."
    python3 "$convert_script" \
        "$MODEL_DIR" \
        --outfile "${CONVERTED_DIR}/kimi-k2.5-f16.gguf" \
        --outtype f16 \
        2>&1 | tee "${PROJECT_ROOT}/logs/convert_f16.log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log_success "F16 conversion complete"
        ls -lh "${CONVERTED_DIR}/kimi-k2.5-f16.gguf"
    else
        log_error "F16 conversion failed"
        exit 1
    fi
}

quantize_model() {
    local input_file=$1
    local quant_type=$2
    local output_name="kimi-k2.5-${quant_type,,}.gguf"
    local output_file="${QUANTIZED_DIR}/${output_name}"
    
    log "Quantizing to $quant_type..."
    
    if [[ -f "$output_file" ]]; then
        log_warning "$quant_type already exists"
        return 0
    fi
    
    local quantize_bin="${LLAMA_CPP}/build/bin/llama-quantize"
    if [[ ! -f "$quantize_bin" ]]; then
        quantize_bin="${LLAMA_CPP}/build/llama-quantize"
    fi
    
    "$quantize_bin" \
        "$input_file" \
        "$output_file" \
        "$quant_type" \
        2>&1 | tee "${PROJECT_ROOT}/logs/quantize_${quant_type}.log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log_success "$quant_type quantization complete"
        ls -lh "$output_file"
    else
        log_error "$quant_type quantization failed"
        return 1
    fi
}

main() {
    log "Kimi K2.5 Model Conversion"
    log "=========================="
    
    if [[ ! -d "$MODEL_DIR" ]]; then
        log_error "Original model not found at $MODEL_DIR"
        log "Run ./scripts/download.sh first"
        exit 1
    fi
    
    log "Step 1/4: Converting to F16..."
    convert_to_f16
    
    log ""
    log "Step 2/4: Quantizing to Q4_K_M (~400GB)..."
    quantize_model "${CONVERTED_DIR}/kimi-k2.5-f16.gguf" "Q4_K_M"
    
    log ""
    log "Step 3/4: Quantizing to Q5_K_M (~500GB)..."
    quantize_model "${CONVERTED_DIR}/kimi-k2.5-f16.gguf" "Q5_K_M"
    
    log ""
    log "Step 4/4: Quantizing to Q8_0 (~800GB)..."
    quantize_model "${CONVERTED_DIR}/kimi-k2.5-f16.gguf" "Q8_0"
    
    log ""
    log_success "All conversions complete!"
    log ""
    log "Model files:"
    ls -lh "$QUANTIZED_DIR"
    log ""
    log "Next step: Run ./scripts/prepare_data.sh to prepare training data"
}

main "$@"
