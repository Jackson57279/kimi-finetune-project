#!/usr/bin/env python3
"""
Inference Script for Kimi K2.5
Supports both base and fine-tuned models
"""

import os
import sys
import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KimiInference:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

    def run_llama_cpp(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        threads: int = 16,
    ):
        import subprocess

        project_root = Path(__file__).parent.parent
        llama_cli = project_root / "llama.cpp" / "build" / "bin" / "llama-cli"

        if not llama_cli.exists():
            llama_cli = project_root / "llama.cpp" / "build" / "llama-cli"

        if not llama_cli.exists():
            raise FileNotFoundError("llama-cli not found. Run setup.sh first")

        cmd = [
            str(llama_cli),
            "-m",
            str(self.model_path),
            "-p",
            prompt,
            "-n",
            str(max_tokens),
            "--temp",
            str(temperature),
            "--top-p",
            str(top_p),
            "--threads",
            str(threads),
            "-c",
            "4096",
            "--repeat-penalty",
            "1.1",
        ]

        logger.info(f"Running inference with {self.model_path.name}")
        logger.info(f"Prompt: {prompt[:100]}...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Inference failed: {result.stderr}")
            return None

        return result.stdout

    def run_huggingface(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.8
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"Loading model from {self.model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt")

        logger.info("Generating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def main():
    parser = argparse.ArgumentParser(description="Run inference with Kimi K2.5")
    parser.add_argument(
        "--model", required=True, help="Path to model file or directory"
    )
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument(
        "--backend",
        choices=["llama_cpp", "huggingface"],
        default="llama_cpp",
        help="Inference backend",
    )

    args = parser.parse_args()

    inference = KimiInference(args.model)

    if args.backend == "llama_cpp":
        output = inference.run_llama_cpp(args.prompt, args.max_tokens, args.temperature)
    else:
        output = inference.run_huggingface(
            args.prompt, args.max_tokens, args.temperature
        )

    if output:
        print("\n" + "=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(output)
        print("=" * 60)


if __name__ == "__main__":
    main()
