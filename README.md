# Kimi K2.5 CPU Fine-Tuning Project

Complete fine-tuning pipeline for Kimi K2.5 (1T parameter MoE model) optimized for CPU-only training with 2.5TB external storage.

## Features

- **Modular Architecture**: Separate scripts for each step
- **UV Package Manager**: Fast, reliable Python dependency management
- **Massive Datasets**: Automatically downloads and mixes large-scale HF datasets
- **Resume Support**: All downloads and training can be resumed
- **Storage Optimized**: 2.5TB limit with intelligent space management
- **CPU Optimized**: OpenBLAS, native compilation, multi-threading

## Storage Requirements

| Component | Size |
|-----------|------|
| Original INT4 Model | ~600 GB |
| F16 Conversion | ~1,200 GB |
| Q4_K_M Quantized | ~400 GB |
| Q5_K_M Quantized | ~500 GB |
| Training Data | ~100 GB |
| Checkpoints | ~300 GB |
| **Total** | **~3,100 GB** |

**With cleanup**: ~2,500 GB

## Quick Start

### 1. Mount External Drive

```bash
sudo mkdir -p /mnt/external2.5tb
sudo mount /dev/sdX1 /mnt/external2.5tb  # Replace sdX1 with your drive
```

### 2. Copy Project

```bash
cp -r kimi-finetune-project /mnt/external2.5tb/
cd /mnt/external2.5tb/kimi-finetune-project
```

### 3. Run Setup

```bash
./run.sh setup
```

This installs:
- uv (Python package manager)
- PyTorch, Transformers, PEFT, etc.
- llama.cpp (compiled for CPU)

### 4. Download Model

```bash
./run.sh download
```

Downloads ~600GB. Resumable if interrupted.

### 5. Convert Model

```bash
./run.sh convert
```

Creates:
- F16 (~1.2TB) - For training
- Q4_K_M (~400GB) - For inference
- Q5_K_M (~500GB) - Higher quality inference

### 6. Prepare Training Data

```bash
./run.sh prepare
```

Downloads and processes massive datasets:
- UltraChat 200k (100k samples)
- OpenOrca (500k samples)
- WizardLM (100k samples)
- OpenHermes 2.5 (100k samples)
- Dolphin (100k samples)

**Total: ~900k training samples**

### 7. Train

```bash
./run.sh train
```

Starts LoRA fine-tuning with:
- Rank: 32
- Alpha: 64
- Learning rate: 1e-4
- Batch size: 1 (with gradient accumulation: 16)
- Context length: 2048

**Warning**: Training on CPU is very slow (days/weeks).

### 8. Run Inference

```bash
./run.sh inference --model models/quantized/kimi-k2.5-q4_k_m.gguf --prompt "Hello, how are you?"
```

## Available Commands

```bash
./run.sh setup      # Setup environment
./run.sh download   # Download model
./run.sh convert    # Convert to GGUF
./run.sh prepare    # Prepare datasets
./run.sh train      # Start training
./run.sh status     # Check status
./run.sh clean      # Clean temp files
./run.sh all        # Run complete pipeline
```

## Project Structure

```
kimi-finetune-project/
├── configs/
│   └── config.yaml          # Main configuration
├── scripts/
│   ├── setup.sh            # Environment setup
│   ├── download.sh         # Model download
│   ├── convert.sh          # Model conversion
│   ├── prepare_data.sh     # Data preparation wrapper
│   ├── train.sh            # Training wrapper
│   └── inference.sh        # Inference wrapper
├── src/
│   ├── prepare_data.py     # Dataset preparation
│   ├── train.py            # Training script
│   └── inference.py        # Inference script
├── models/
│   ├── original/           # Downloaded model (~600GB)
│   ├── converted/          # F16 GGUF (~1.2TB)
│   └── quantized/          # Quantized models
├── data/
│   ├── raw/                # Raw datasets
│   ├── processed/          # Training data
│   └── cache/              # HF cache
├── output/
│   ├── checkpoints/        # Training checkpoints
│   ├── final/              # Final model
│   └── logs/               # Training logs
├── llama.cpp/              # llama.cpp source
├── cache/                  # HuggingFace cache
├── bin/                    # Compiled binaries
├── run.sh                  # Main orchestrator
└── activate.sh             # Environment activation
```

## Configuration

Edit `configs/config.yaml` to customize:

- Storage paths and limits
- Dataset selection and mixing
- LoRA hyperparameters
- Training settings

## Training Tips

1. **Start Small**: Test with a small dataset first
2. **Monitor Memory**: Training uses significant RAM
3. **Use Checkpoints**: Training can be resumed
4. **Be Patient**: CPU training is very slow
5. **Consider Cloud**: For faster training, use GPU cloud instances

## Troubleshooting

### Out of Space

```bash
./run.sh clean  # Remove temp files
# Or manually delete:
rm -rf cache/*
rm -rf models/converted/*.gguf  # Keep only quantized versions
```

### Resume Download

Download automatically resumes. Just re-run:

```bash
./run.sh download
```

### Resume Training

Training automatically resumes from checkpoints. Just re-run:

```bash
./run.sh train
```

## License

This project follows the same license as Kimi K2.5 (Modified MIT).

## Support

For issues:
1. Check logs in `output/logs/`
2. Run `./run.sh status` for diagnostics
3. Check available space: `df -h`
