#!/usr/bin/env python3
"""Train a local QLoRA adapter from `lmx eval train prepare` SFT JSONL."""

from __future__ import annotations

import argparse
import json
import inspect
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a verifier-filtered conversational SFT adapter with TRL and PEFT."
    )
    parser.add_argument("--dataset", required=True, help="Path to sft.jsonl")
    parser.add_argument("--model", required=True, help="Loadable HuggingFace base model ID or path")
    parser.add_argument("--output", required=True, help="Adapter output directory")
    parser.add_argument(
        "--backend",
        choices=("trl", "unsloth"),
        default="trl",
        help="Model loading/training backend. Use unsloth for its patched QLoRA stack.",
    )
    parser.add_argument(
        "--load-in-16bit",
        action="store_true",
        help="Unsloth only: load the base model in 16-bit instead of 4-bit.",
    )
    parser.add_argument(
        "--target-modules",
        default="auto",
        help="Unsloth LoRA target suffixes, comma-separated; default: infer dense versus MoE.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-sequence-loss",
        action="store_true",
        help="Train on all tokens. Default is assistant-only loss and requires a supported chat template.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.is_file():
        raise SystemExit(f"dataset not found: {dataset}")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    if args.max_length < 256:
        raise SystemExit("--max-length must be at least 256")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("batch size and gradient accumulation must be positive")
    if args.lora_r < 1 or args.lora_alpha < 1:
        raise SystemExit("LoRA rank and alpha must be positive")
    return dataset


def validate_dataset(dataset_path: Path) -> int:
    examples = 0
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL at line {line_number}: {exc}") from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise SystemExit(f"line {line_number} has no conversational messages")
            roles = {message.get("role") for message in messages if isinstance(message, dict)}
            if "user" not in roles or "assistant" not in roles:
                raise SystemExit(f"line {line_number} needs both user and assistant messages")
            examples += 1
    if examples == 0:
        raise SystemExit("dataset contains no examples")
    return examples

def load_training_rows(dataset_path: Path) -> list[dict]:
    rows: list[dict] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL at line {line_number}: {exc}") from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise SystemExit(f"line {line_number} has no conversational messages")
            rows.append(row)
    if not rows:
        raise SystemExit("dataset contains no examples")
    return rows


def is_probable_moe(model_name: str, config: object | None = None) -> bool:
    if config is not None:
        model_type = str(getattr(config, "model_type", "")).lower()
        if "moe" in model_type or int(getattr(config, "num_experts", 0) or 0) > 1:
            return True
    normalized = model_name.lower()
    return "moe" in normalized or "-a3b" in normalized or "-a22b" in normalized


def unsloth_target_modules(
    requested: str, model_name: str, config: object | None = None
) -> list[str]:
    if requested != "auto":
        modules = [module.strip() for module in requested.split(",") if module.strip()]
        if not modules:
            raise SystemExit("--target-modules must contain at least one module")
        return modules
    if is_probable_moe(model_name, config):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize_training_rows(
    tokenizer: object, rows: list[dict], max_length: int, assistant_only: bool
) -> dict[str, list[list[int]]]:
    encoded = {"input_ids": [], "attention_mask": [], "labels": []}
    for row_index, row in enumerate(rows, 1):
        input_ids: list[int] = []
        labels: list[int] = []
        for message_index, message in enumerate(row["messages"], 1):
            try:
                current = tokenizer.apply_chat_template(
                    row["messages"][:message_index],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            except Exception as exc:
                raise SystemExit(
                    f"dataset row {row_index}: the model chat template could not render "
                    f"message {message_index}: {exc}"
                ) from exc
            if isinstance(current, dict):
                current = current.get("input_ids")
            if (
                isinstance(current, list)
                and len(current) == 1
                and isinstance(current[0], list)
            ):
                current = current[0]
            if not isinstance(current, list) or not all(
                isinstance(token, int) for token in current
            ):
                raise SystemExit(
                    f"dataset row {row_index}: chat template returned unsupported token IDs"
                )
            common = 0
            common_limit = min(len(input_ids), len(current))
            while common < common_limit and input_ids[common] == current[common]:
                common += 1
            labels = labels[:common]
            learn_tail = not assistant_only or message.get("role") == "assistant"
            labels.extend(current[common:] if learn_tail else [-100] * (len(current) - common))
            input_ids = current

        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        if not input_ids:
            raise SystemExit(f"dataset row {row_index}: chat template rendered no tokens")
        if assistant_only and all(label == -100 for label in labels):
            raise SystemExit(
                f"dataset row {row_index}: no assistant tokens remain within --max-length"
            )
        encoded["input_ids"].append(input_ids)
        encoded["attention_mask"].append([1] * len(input_ids))
        encoded["labels"].append(labels)
    return encoded


def compatible_keyword(callable_object: object, current: str, legacy: str) -> str:
    parameters = inspect.signature(callable_object).parameters
    if current in parameters:
        return current
    if legacy in parameters:
        return legacy
    raise SystemExit(
        f"installed training library supports neither {current!r} nor {legacy!r}"
    )


def main() -> None:
    args = parse_args()
    dataset_path = validate_args(args)
    example_count = validate_dataset(dataset_path)
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "unsloth":
        run_unsloth(args, dataset_path, output_dir, example_count)
    else:
        run_trl(args, dataset_path, output_dir, example_count)


def run_trl(
    args: argparse.Namespace, dataset_path: Path, output_dir: Path, example_count: int
) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "training dependencies are missing; install them with "
            "`python3 -m pip install 'trl[peft]' bitsandbytes datasets`"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA training requires a CUDA GPU visible to PyTorch")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
        assistant_only_loss=not args.full_sequence_loss,
        packing=False,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
    )
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    print(
        json.dumps(
            {
                "event": "training_start",
                "backend": "trl",
                "model": args.model,
                "dataset": str(dataset_path),
                "examples": example_count,
                "output": str(output_dir),
                "assistantOnlyLoss": not args.full_sequence_loss,
                "quantization": "4bit-nf4-double",
            }
        )
    )
    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        quantization_config=quantization_config,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    trainer.processing_class.save_pretrained(str(output_dir))
    print(
        json.dumps(
            {"event": "training_complete", "backend": "trl", "output": str(output_dir)}
        )
    )


def run_unsloth(
    args: argparse.Namespace, dataset_path: Path, output_dir: Path, example_count: int
) -> None:
    try:
        # Unsloth must patch Transformers and TRL before either library is imported.
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Unsloth training dependencies are missing. Install a current Unsloth build "
            "for this CUDA/PyTorch environment, plus trl, peft, bitsandbytes, and datasets."
        ) from exc

    if is_probable_moe(args.model) and not args.load_in_16bit:
        print(
            "warning: this looks like an MoE checkpoint; current Unsloth guidance does "
            "not recommend bitsandbytes 4-bit loading for MoE training. Use "
            "--load-in-16bit if the checkpoint fits in VRAM.",
            file=sys.stderr,
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_length,
        load_in_4bit=not args.load_in_16bit,
        load_in_8bit=False,
        full_finetuning=False,
    )
    targets = unsloth_target_modules(args.target_modules, args.model, model.config)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=targets,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    rows = load_training_rows(dataset_path)
    encoded = tokenize_training_rows(
        tokenizer,
        rows,
        max_length=args.max_length,
        assistant_only=not args.full_sequence_loss,
    )
    dataset = Dataset.from_dict(encoded)
    config_kwargs = {
        "output_dir": str(output_dir),
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "packing": False,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": "none",
        "seed": args.seed,
        "gradient_checkpointing": True,
    }
    config_kwargs[
        compatible_keyword(SFTConfig.__init__, "max_length", "max_seq_length")
    ] = args.max_length
    training_args = SFTConfig(**config_kwargs)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
    }
    trainer_kwargs[
        compatible_keyword(SFTTrainer.__init__, "processing_class", "tokenizer")
    ] = tokenizer

    print(
        json.dumps(
            {
                "event": "training_start",
                "backend": "unsloth",
                "dataset": str(dataset_path),
                "examples": example_count,
                "model": args.model,
                "output": str(output_dir),
                "loadIn4bit": not args.load_in_16bit,
                "targetModules": targets,
                "loss": "full_sequence" if args.full_sequence_loss else "assistant_only",
            }
        )
    )
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(
        json.dumps(
            {
                "event": "training_complete",
                "backend": "unsloth",
                "output": str(output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
