#!/bin/bash
set -euo pipefail

# Kimi K2.5 Fine-Tuning Setup Script
# Uses uv for Python environment management

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/configs/config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install uv if not present
install_uv() {
    log "Checking for uv..."
    if command_exists uv; then
        log_success "uv already installed: $(uv --version)"
        return 0
    fi
    
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    
    if command_exists uv; then
        log_success "uv installed successfully: $(uv --version)"
    else
        log_error "Failed to install uv"
        exit 1
    fi
}

# Check Python version
check_python() {
    log "Checking Python version..."
    
    if ! command_exists python3; then
        log_error "Python 3 not found. Please install Python 3.11+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log "Found Python $PYTHON_VERSION"
    
    if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc -l) -eq 0 ]]; then
        log_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
    
    log_success "Python version check passed"
}

install_build_deps() {
    log "Checking build dependencies..."
    
    if ! command_exists cmake; then
        log "Installing cmake..."
        if command_exists apt-get; then
            sudo apt-get update
            sudo apt-get install -y cmake build-essential libopenblas-dev
        elif command_exists yum; then
            sudo yum install -y cmake gcc-c++ openblas-devel
        elif command_exists pacman; then
            sudo pacman -S cmake base-devel openblas
        else
            log_error "Could not install cmake automatically. Please install cmake manually."
            exit 1
        fi
    fi
    
    log_success "Build dependencies installed: $(cmake --version | head -1)"
}

# Create uv virtual environment
setup_venv() {
    log "Setting up uv virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    if [[ -d ".venv" ]]; then
        log_warning "Virtual environment already exists. Removing..."
        rm -rf .venv
    fi
    
    uv venv --python python3.11
    
    log_success "Virtual environment created"
}

# Install Python dependencies
install_deps() {
    log "Installing Python dependencies..."
    
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    # Core dependencies
    uv pip install \
        torch==2.5.1 \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
    
    # HuggingFace ecosystem
    uv pip install \
        transformers>=4.49.0 \
        datasets>=2.18.0 \
        huggingface-hub>=0.21.0 \
        tokenizers>=0.15.0 \
        accelerate>=0.27.0 \
        peft>=0.9.0 \
        bitsandbytes>=0.42.0 \
        trl>=0.8.0
    
    # Training utilities
    uv pip install \
        wandb \
        tensorboard \
        scipy \
        scikit-learn \
        pandas \
        numpy \
        tqdm \
        pyyaml \
        pyarrow \
        psutil
    
    # Additional utilities
    uv pip install \
        sentencepiece \
        protobuf \
        tiktoken \
        sentence-transformers
    
    # Development tools
    uv pip install \
        black \
        ruff \
        mypy \
        pytest
    
    log_success "Dependencies installed"
}

# Install llama.cpp
setup_llamacpp() {
    log "Setting up llama.cpp..."
    
    cd "$PROJECT_ROOT"
    
    LLAMA_DIR="${PROJECT_ROOT}/llama.cpp"
    
    if [[ -d "$LLAMA_DIR" ]]; then
        log_warning "llama.cpp already exists. Updating..."
        cd "$LLAMA_DIR"
        git pull origin master
    else
        log "Cloning llama.cpp..."
        git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
        cd "$LLAMA_DIR"
    fi
    
    # Build llama.cpp with CPU optimizations
    log "Building llama.cpp with CPU optimizations..."
    mkdir -p build
    cd build
    
    cmake .. \
        -DGGML_CUDA=OFF \
        -DGGML_OPENBLAS=ON \
        -DGGML_NATIVE=ON \
        -DGGML_AVX=ON \
        -DGGML_AVX2=ON \
        -DGGML_FMA=ON \
        -DGGML_F16C=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_LTO=ON \
        -DLLAMA_BUILD_TESTS=OFF
    
    cmake --build . --config Release -j$(nproc)
    
    log_success "llama.cpp built successfully"
    
    # Create symlinks in project bin
    mkdir -p "${PROJECT_ROOT}/bin"
    for binary in llama-cli llama-quantize llama-finetune llama-export-lora convert_hf_to_gguf; do
        if [[ -f "bin/${binary}" ]]; then
            ln -sf "${LLAMA_DIR}/build/bin/${binary}" "${PROJECT_ROOT}/bin/${binary}"
        elif [[ -f "${binary}" ]]; then
            ln -sf "${LLAMA_DIR}/build/${binary}" "${PROJECT_ROOT}/bin/${binary}"
        fi
    done
}

# Create project directories
create_dirs() {
    log "Creating project directories..."
    
    cd "$PROJECT_ROOT"
    
    mkdir -p \
        models/original \
        models/converted \
        models/quantized \
        data/raw \
        data/processed \
        data/cache \
        output/checkpoints \
        output/final \
        output/logs \
        logs \
        cache/huggingface \
        cache/datasets \
        bin
    
    log_success "Directories created"
}

# Setup HuggingFace cache
setup_hf_cache() {
    log "Setting up HuggingFace cache..."
    
    cd "$PROJECT_ROOT"
    
    export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
    export HF_DATASETS_CACHE="${PROJECT_ROOT}/cache/datasets"
    
    # Add to activation script
    cat >> .venv/bin/activate << 'EOF'

# Kimi Fine-tuning Environment
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/cache/datasets"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
EOF
    
    log_success "HuggingFace cache configured"
}

# Check storage availability
check_storage() {
    log "Checking storage availability..."
    
    BASE_DIR=$(grep "base_dir:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
    
    if [[ ! -d "$BASE_DIR" ]]; then
        log_warning "Base directory $BASE_DIR does not exist"
        log "Creating directory..."
        mkdir -p "$BASE_DIR" || {
            log_error "Failed to create $BASE_DIR"
            exit 1
        }
    fi
    
    AVAILABLE_GB=$(df -BG "$BASE_DIR" | awk 'NR==2 {print $4}' | tr -d 'G')
    log "Available storage: ${AVAILABLE_GB}GB"
    
    if [[ $AVAILABLE_GB -lt 2500 ]]; then
        log_warning "Less than 2.5TB available. Recommended: 2.5TB+"
    else
        log_success "Storage check passed"
    fi
}

# Create activation script
create_activate_script() {
    log "Creating activation script..."
    
    cat > "${PROJECT_ROOT}/activate.sh" << EOF
#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "\${PROJECT_ROOT}/.venv/bin/activate"

export HF_HOME="\${PROJECT_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="\${PROJECT_ROOT}/cache/datasets"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PATH="\${PROJECT_ROOT}/.venv/bin:\${PROJECT_ROOT}/bin:\${PATH}"
export PYTHONPATH="\${PROJECT_ROOT}/src:\${PYTHONPATH:-}"

echo "✓ Kimi Fine-tuning environment activated"
echo "  Project: \$PROJECT_ROOT"
echo "  Python: \$(which python)"
echo "  HF Cache: \$HF_HOME"
EOF
    
    chmod +x "${PROJECT_ROOT}/activate.sh"
    log_success "Activation script created at ${PROJECT_ROOT}/activate.sh"
}

# Main setup function
main() {
    log "Starting Kimi K2.5 Fine-Tuning Setup"
    log "Project Root: $PROJECT_ROOT"
    
    check_python
    install_uv
    install_build_deps
    setup_venv
    install_deps
    setup_llamacpp
    create_dirs
    setup_hf_cache
    check_storage
    create_activate_script
    
    log_success "Setup complete!"
    log ""
    log "Next steps:"
    log "  1. Source the environment: source ${PROJECT_ROOT}/activate.sh"
    log "  2. Download model: ./scripts/download.sh"
    log "  3. Prepare data: ./scripts/prepare_data.sh"
    log "  4. Start training: ./scripts/train.sh"
    log ""
    log "Or run all steps: ./run.sh all"
}

main "$@"
