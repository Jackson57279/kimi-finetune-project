#!/usr/bin/env python3
"""
Kimi K2.5 LoRA Training Script
Optimized for CPU training with memory-efficient settings
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"


class KimiTrainer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.base_dir = Path(self.config["project"]["base_dir"])
        self.setup_directories()

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def setup_directories(self):
        dirs = [
            self.base_dir / "output" / "checkpoints",
            self.base_dir / "output" / "logs",
            self.base_dir / "output" / "final",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def verify_environment(self):
        logger.info("Verifying environment...")

        try:
            import torch

            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CPU threads: {torch.get_num_threads()}")

            import transformers

            logger.info(f"Transformers version: {transformers.__version__}")

            import peft

            logger.info(f"PEFT version: {peft.__version__}")

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            sys.exit(1)

        model_dir = self.base_dir / "models" / "original"
        if not model_dir.exists():
            logger.error(f"Model not found at {model_dir}")
            logger.info("Run download.sh and convert.sh first")
            sys.exit(1)

        data_file = self.base_dir / "data" / "processed" / "training_data.txt"
        if not data_file.exists():
            logger.error(f"Training data not found at {data_file}")
            logger.info("Run prepare_data.py first")
            sys.exit(1)

        logger.info("Environment verified ✓")

    def load_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        logger.info("Loading model and tokenizer...")

        model_dir = self.base_dir / "models" / "original"

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, padding_side="right"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        logger.info("Loading model in 4-bit (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            quantization_config=bnb_config,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        logger.info(
            f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B"
        )

        return model, tokenizer

    def setup_lora(self, model):
        from peft import LoraConfig, get_peft_model, TaskType

        logger.info("Setting up LoRA...")

        lora_config = LoraConfig(
            r=self.config["training"]["lora"]["rank"],
            lora_alpha=self.config["training"]["lora"]["alpha"],
            target_modules=self.config["training"]["lora"]["target_modules"],
            lora_dropout=self.config["training"]["lora"]["dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    def prepare_dataset(self, tokenizer):
        from datasets import Dataset

        data_file = self.base_dir / "data" / "processed" / "training_data.txt"

        logger.info(f"Loading dataset from {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(texts)} training examples")

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config["training"]["hyperparameters"]["max_seq_length"],
                padding="max_length",
            )

        dataset = Dataset.from_dict({"text": texts})
        tokenized = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        return tokenized

    def train(self):
        from transformers import (
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        import torch

        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)

        self.verify_environment()

        model, tokenizer = self.load_model_and_tokenizer()
        model = self.setup_lora(model)
        dataset = self.prepare_dataset(tokenizer)

        output_dir = self.base_dir / "output" / "checkpoints"
        logging_dir = self.base_dir / "output" / "logs"

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config["training"]["hyperparameters"]["num_epochs"],
            per_device_train_batch_size=self.config["training"]["hyperparameters"][
                "batch_size"
            ],
            gradient_accumulation_steps=self.config["training"]["hyperparameters"][
                "gradient_accumulation_steps"
            ],
            learning_rate=self.config["training"]["hyperparameters"]["learning_rate"],
            warmup_steps=self.config["training"]["hyperparameters"]["warmup_steps"],
            logging_steps=self.config["training"]["hyperparameters"]["logging_steps"],
            save_steps=self.config["training"]["hyperparameters"]["save_steps"],
            save_total_limit=3,
            max_grad_norm=self.config["training"]["hyperparameters"]["max_grad_norm"],
            group_by_length=self.config["training"]["hyperparameters"][
                "group_by_length"
            ],
            bf16=self.config["training"]["hyperparameters"]["bf16"],
            fp16=False,
            optim=self.config["training"]["hyperparameters"]["optim"],
            weight_decay=self.config["training"]["hyperparameters"]["weight_decay"],
            logging_dir=str(logging_dir),
            report_to=["tensorboard"],
            remove_unused_columns=False,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        logger.info("Starting training loop...")
        logger.info("This will take a long time on CPU. Be patient!")

        try:
            trainer.train()

            final_dir = self.base_dir / "output" / "final"
            logger.info(f"Saving final model to {final_dir}")
            model.save_pretrained(str(final_dir))
            tokenizer.save_pretrained(str(final_dir))

            logger.info("Training complete!")

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            logger.info("Saving checkpoint...")
            model.save_pretrained(str(output_dir / "interrupted"))
            raise


def main():
    parser = argparse.ArgumentParser(description="Train Kimi K2.5 with LoRA")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    trainer = KimiTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
