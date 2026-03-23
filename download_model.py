#!/usr/bin/env python3
"""
Download Kimi K2.5 model without using huggingface-cli.
Uses huggingface_hub Python library with resume support.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Model configuration
MODEL_REPO = "moonshotai/Kimi-K2.5"
MODEL_DIR = Path(__file__).parent.parent / "models" / "original"


def check_space(required_gb=700):
    """Check if enough disk space is available."""
    import shutil

    stat = shutil.disk_usage(MODEL_DIR.parent)
    available_gb = stat.free / (1024**3)
    if available_gb < required_gb:
        print(
            f"⚠ Warning: Low disk space. Available: {available_gb:.1f}GB, Required: ~{required_gb}GB"
        )
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            sys.exit(1)
    return True


def main():
    print("=" * 60)
    print("Kimi K2.5 Model Download")
    print("=" * 60)
    print(f"Model: {MODEL_REPO}")
    print(f"Destination: {MODEL_DIR}")
    print(f"Expected size: ~600GB")
    print("")

    # Create directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Check space
    check_space(700)

    # Enable fast transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print("Starting download...")
    print("This will take several hours. Progress will be shown below.")
    print("-" * 60)

    try:
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=str(MODEL_DIR),
            resume_download=True,
            local_dir_use_symlinks=False,
            max_workers=4,
        )

        print("-" * 60)
        print("✓ Download complete!")

        # Get final size
        import subprocess

        result = subprocess.run(
            ["du", "-sh", str(MODEL_DIR)], capture_output=True, text=True
        )
        print(f"Final size: {result.stdout.split()[0]}")
        print(f"Location: {MODEL_DIR}")
        print("")
        print("Next step: Run ./scripts/convert.sh to convert to GGUF format")

    except KeyboardInterrupt:
        print("\n⚠ Download interrupted. Run again to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
