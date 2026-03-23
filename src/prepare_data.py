#!/usr/bin/env python3
"""
Massive Dataset Preparation for Kimi K2.5 Fine-Tuning
Downloads and processes large-scale datasets from HuggingFace
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm
import random
import hashlib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    name: str
    sample_size: int
    weight: float
    format: str
    subset: Optional[str] = None


class MassiveDatasetLoader:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.output_dir = (
            Path(self.config["project"]["base_dir"]) / "data" / "processed"
        )
        self.cache_dir = Path(self.config["project"]["base_dir"]) / "data" / "cache"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.datasets_config = [
            DatasetConfig(**ds) for ds in self.config["datasets"]["massive_datasets"]
        ]

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def load_dataset_with_retry(
        self, dataset_name: str, subset: Optional[str] = None, max_retries: int = 3
    ):
        from datasets import load_dataset

        for attempt in range(max_retries):
            try:
                if subset:
                    return load_dataset(
                        dataset_name, subset, cache_dir=str(self.cache_dir)
                    )
                else:
                    return load_dataset(dataset_name, cache_dir=str(self.cache_dir))
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {dataset_name}: {e}")
                if attempt == max_retries - 1:
                    raise
                import time

                time.sleep(5 * (attempt + 1))

    def format_alpaca(self, example: dict) -> str:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    def format_chatml(self, example: dict, tokenizer=None) -> str:
        messages = example.get("messages", [])
        if not messages and "conversations" in example:
            messages = [
                {
                    "role": "user" if turn.get("from") == "human" else "assistant",
                    "content": turn.get("value", ""),
                }
                for turn in example["conversations"]
            ]

        if tokenizer:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"\u003c|im_start|\u003e{role}\n{content}\u003c|im_end|\u003e\n"
            return text

    def format_openorca(self, example: dict) -> str:
        system_prompt = example.get("system_prompt", "You are a helpful assistant.")
        question = example.get("question", "")
        response = example.get("response", "")

        return f"### System:\n{system_prompt}\n\n### Instruction:\n{question}\n\n### Response:\n{response}"

    def process_dataset(self, config: DatasetConfig, tokenizer=None) -> List[str]:
        logger.info(f"Loading {config.name}...")

        try:
            dataset = self.load_dataset_with_retry(config.name, config.subset)
        except Exception as e:
            logger.error(f"Failed to load {config.name}: {e}")
            return []

        if "train" in dataset:
            dataset = dataset["train"]

        total_samples = len(dataset)
        sample_size = min(config.sample_size, total_samples)

        logger.info(f"Total samples: {total_samples}, Using: {sample_size}")

        if sample_size < total_samples:
            indices = random.sample(range(total_samples), sample_size)
            dataset = dataset.select(indices)

        formatted_samples = []

        for example in tqdm(dataset, desc=f"Processing {config.name}"):
            try:
                if config.format == "alpaca":
                    text = self.format_alpaca(example)
                elif config.format == "chatml":
                    text = self.format_chatml(example, tokenizer)
                elif config.format == "openorca":
                    text = self.format_openorca(example)
                else:
                    continue

                text = text.strip()
                if len(text) > 50 and len(text) < 8192:
                    formatted_samples.append(text)

            except Exception as e:
                logger.debug(f"Error processing example: {e}")
                continue

        logger.info(f"Processed {len(formatted_samples)} samples from {config.name}")
        return formatted_samples

    def deduplicate(self, samples: List[str]) -> List[str]:
        logger.info("Deduplicating samples...")

        seen = set()
        unique = []

        for sample in tqdm(samples, desc="Deduplicating"):
            sample_hash = hashlib.md5(sample.encode()).hexdigest()
            if sample_hash not in seen:
                seen.add(sample_hash)
                unique.append(sample)

        removed = len(samples) - len(unique)
        logger.info(
            f"Removed {removed} duplicates ({removed / len(samples) * 100:.1f}%)"
        )

        return unique

    def mix_datasets(self, dataset_samples: Dict[str, List[str]]) -> List[str]:
        logger.info("Mixing datasets...")

        mixed = []

        for config in self.datasets_config:
            if config.name not in dataset_samples:
                continue

            samples = dataset_samples[config.name]
            weight = config.weight

            n_samples = int(len(samples) * weight)
            selected = random.sample(samples, min(n_samples, len(samples)))

            mixed.extend(selected)
            logger.info(
                f"Added {len(selected)} samples from {config.name} (weight={weight})"
            )

        random.shuffle(mixed)
        logger.info(f"Total mixed samples: {len(mixed)}")

        return mixed

    def save_dataset(self, samples: List[str], output_file: Path):
        logger.info(f"Saving {len(samples)} samples to {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            for sample in tqdm(samples, desc="Saving"):
                f.write(sample + "\n")

        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {size_mb:.2f} MB")

    def run(self):
        logger.info("=" * 60)
        logger.info("MASSIVE DATASET PREPARATION")
        logger.info("=" * 60)

        from transformers import AutoTokenizer

        logger.info("Loading tokenizer...")
        model_dir = Path(self.config["project"]["base_dir"]) / "models" / "original"
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )

        dataset_samples = {}

        for ds_config in self.datasets_config:
            try:
                samples = self.process_dataset(ds_config, tokenizer)
                if samples:
                    dataset_samples[ds_config.name] = samples
            except Exception as e:
                logger.error(f"Failed to process {ds_config.name}: {e}")
                continue

        if not dataset_samples:
            logger.error("No datasets were successfully loaded")
            return

        logger.info("\n" + "=" * 60)
        logger.info("MIXING DATASETS")
        logger.info("=" * 60)

        mixed = self.mix_datasets(dataset_samples)

        if self.config["datasets"]["mixing"]["deduplicate"]:
            mixed = self.deduplicate(mixed)

        output_file = self.output_dir / "training_data.txt"
        self.save_dataset(mixed, output_file)

        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, "w") as f:
            json.dump(
                {
                    "total_samples": len(mixed),
                    "datasets": {k: len(v) for k, v in dataset_samples.items()},
                    "config": self.config["datasets"],
                },
                f,
                indent=2,
            )

        logger.info("\n" + "=" * 60)
        logger.info("DATASET PREPARATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output: {output_file}")
        logger.info(f"Stats: {stats_file}")
        logger.info(f"Total samples: {len(mixed)}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare massive datasets for training"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Config file path"
    )
    parser.add_argument("--output-dir", help="Override output directory")

    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    loader = MassiveDatasetLoader(args.config)

    if args.output_dir:
        loader.output_dir = Path(args.output_dir)
        loader.output_dir.mkdir(parents=True, exist_ok=True)

    loader.run()


if __name__ == "__main__":
    main()
