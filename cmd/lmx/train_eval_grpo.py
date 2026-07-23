#!/usr/bin/env python3
"""Validate and run online GRPO training prepared by ``lmx eval train rl``.

The module deliberately imports only the Python standard library at import time.
TRL, Transformers, Datasets, and the user-selected environment plugin are loaded
only after all manifest, dataset, output, and resume validation has succeeded.
"""

from __future__ import annotations

import argparse
import copy
import functools
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
MANIFEST_KIND = "localmaxxing.eval_rl_grpo"
ALGORITHM = "online_grpo"
DATASET_FORMAT = "trl_conversational_prompt_jsonl"
TRAINER_IMPLEMENTATION = "localmaxxing_trl_grpo"
REQUIRED_TRL_VERSION = "1.8.0"
TRANSFORMERS_MIN_VERSION = (5, 2, 0)
TRANSFORMERS_MAX_MAJOR = 6
INSTALL_COMMAND = (
    "python -m pip install 'trl==1.8.0' 'transformers>=5.2.0,<6'"
)


class TrainingValidationError(ValueError):
    """Raised when prepared training input is unsafe or inconsistent."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or run an lmx online-GRPO training manifest."
    )
    parser.add_argument("--manifest", required=True, help="RL manifest.json path")
    parser.add_argument(
        "--output-dir", required=True, help="Trainer output directory (may override manifest)"
    )
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or a checkpoint directory containing trainer_state.json",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs without importing the environment plugin or training packages",
    )
    return parser.parse_args(argv)


def _reject_json_constant(value: str) -> None:
    raise TrainingValidationError(f"non-finite JSON number {value!r} is not allowed")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_json(text: str, *, source: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except TrainingValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise TrainingValidationError(
            f"{source} is not valid JSON: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingValidationError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, required: set[str], name: str
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise TrainingValidationError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise TrainingValidationError(f"{name} has unknown fields: {', '.join(unknown)}")


def _require_string(value: Any, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise TrainingValidationError(f"{name} must be a string")
    if nonempty and not value.strip():
        raise TrainingValidationError(f"{name} must not be empty")
    return value


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingValidationError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise TrainingValidationError(f"{name} must be at least {minimum}")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingValidationError(f"{name} must be a boolean")
    return value


def _absolute_existing_directory(value: Any, name: str) -> Path:
    raw = _require_string(value, name)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise TrainingValidationError(f"{name} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TrainingValidationError(f"{name} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise TrainingValidationError(f"{name} must be a directory: {resolved}")
    return resolved


def _absolute_existing_file(value: Any, name: str) -> Path:
    raw = _require_string(value, name)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise TrainingValidationError(f"{name} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TrainingValidationError(f"{name} does not exist: {path}") from exc
    if not resolved.is_file():
        raise TrainingValidationError(f"{name} must be a file: {resolved}")
    return resolved


def _absolute_output_path(value: Any, name: str) -> Path:
    raw = _require_string(value, name)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise TrainingValidationError(f"{name} must be an absolute path")
    return path.resolve(strict=False)


def _validate_json_value(value: Any, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrainingValidationError(f"{name} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TrainingValidationError(f"{name} contains a non-string object key")
            _validate_json_value(item, f"{name}.{key}")
        return
    raise TrainingValidationError(f"{name} contains a non-JSON value")


_BOOL_GRPO_FIELDS = {
    "bf16",
    "disable_dropout",
    "fp16",
    "gradient_checkpointing",
    "greater_is_better",
    "include_tokens_per_second",
    "mask_truncated_completions",
    "shuffle_dataset",
    "sync_ref_model",
    "tf32",
    "use_liger_kernel",
    "use_vllm",
    "vllm_enable_sleep_mode",
    "use_transformers_paged",
}
_POSITIVE_INT_GRPO_FIELDS = {
    "eval_accumulation_steps",
    "generation_batch_size",
    "gradient_accumulation_steps",
    "num_generations_eval",
    "max_completion_length",
    "num_completions_to_print",
    "num_generations",
    "num_iterations",
    "per_device_train_batch_size",
    "ref_model_sync_steps",
    "save_total_limit",
    "steps_per_generation",
    "vllm_server_port",
    "vllm_server_timeout",
    "vllm_tensor_parallel_size",
}
_NONNEGATIVE_INT_GRPO_FIELDS = {
    "dataloader_num_workers",
    "data_seed",
    "seed",
    "warmup_steps",
}
_POSITIVE_NUMBER_GRPO_FIELDS = {
    "learning_rate",
    "logging_steps",
    "max_grad_norm",
    "num_train_epochs",
    "repetition_penalty",
    "save_steps",
    "temperature",
    "vllm_gpu_memory_utilization",
}
_NONNEGATIVE_NUMBER_GRPO_FIELDS = {
    "beta",
    "delta",
    "epsilon",
    "epsilon_high",
    "ref_model_mixup_alpha",
    "weight_decay",
}
_RATIO_GRPO_FIELDS = {"min_p", "top_p", "warmup_ratio"}
_NONEMPTY_STRING_GRPO_FIELDS = {
    "chat_template",
    "importance_sampling_level",
    "loss_type",
    "lr_scheduler_type",
    "optim",
    "optim_args",
    "save_strategy",
    "scale_rewards",
    "vllm_mode",
    "vllm_model_impl",
    "vllm_server_base_url",
    "vllm_server_host",
    "logging_strategy",
}
_OBJECT_GRPO_FIELDS = {
    "gradient_checkpointing_kwargs",
    "lr_scheduler_kwargs",
    "model_init_kwargs",
    "generation_kwargs",
}
_OPTIONAL_POSITIVE_INT_GRPO_FIELDS = {
    "max_tool_calling_iterations",
    "vllm_max_model_length",
}
_RESERVED_GRPO_FIELDS = {
    "environment_factory",
    "log_completions",
    "output_dir",
    "overwrite_output_dir",
    "report_to",
    "remove_unused_columns",
    "resume_from_checkpoint",
    "rollout_func",
    "tools",
    "reward_weights",
}


def _number(value: Any, name: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingValidationError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise TrainingValidationError(f"{name} must be finite")
    return value


def validate_grpo_config(value: Any) -> dict[str, Any]:
    """Validate and defensively copy the user-controlled GRPOConfig subset."""

    config = _require_object(value, "trainer.grpoConfig")
    validated: dict[str, Any] = {}
    for key, raw in config.items():
        name = f"trainer.grpoConfig.{key}"
        if key in _RESERVED_GRPO_FIELDS:
            raise TrainingValidationError(
                f"{name} is controlled by the runner and must not be set"
            )
        if key == "logging_first_step":
            validated[key] = _require_bool(raw, name)
        elif key in _BOOL_GRPO_FIELDS:
            validated[key] = _require_bool(raw, name)
        elif key == "num_generations":
            validated[key] = _require_int(raw, name, minimum=2)
        elif key in _POSITIVE_INT_GRPO_FIELDS:
            validated[key] = _require_int(raw, name, minimum=1)
        elif key in _NONNEGATIVE_INT_GRPO_FIELDS:
            validated[key] = _require_int(raw, name, minimum=0)
        elif key in _POSITIVE_NUMBER_GRPO_FIELDS:
            number = _number(raw, name)
            if number <= 0:
                raise TrainingValidationError(f"{name} must be greater than zero")
            validated[key] = number
        elif key in _NONNEGATIVE_NUMBER_GRPO_FIELDS:
            number = _number(raw, name)
            if number < 0:
                raise TrainingValidationError(f"{name} must not be negative")
            validated[key] = number
        elif key in _RATIO_GRPO_FIELDS:
            number = _number(raw, name)
            if number < 0 or number > 1:
                raise TrainingValidationError(f"{name} must be between zero and one")
            validated[key] = number
        elif key in _NONEMPTY_STRING_GRPO_FIELDS:
            validated[key] = _require_string(raw, name)
        elif key in _OBJECT_GRPO_FIELDS:
            if raw is None and key in {"gradient_checkpointing_kwargs", "model_init_kwargs"}:
                validated[key] = None
            else:
                validated[key] = copy.deepcopy(_require_object(raw, name))
        elif key in _OPTIONAL_POSITIVE_INT_GRPO_FIELDS:
            if raw is None:
                validated[key] = None
            else:
                validated[key] = _require_int(raw, name, minimum=1)
        elif key == "max_steps":
            step_count = _require_int(raw, name)
            if step_count != -1 and step_count < 1:
                raise TrainingValidationError(f"{name} must be -1 or a positive integer")
            validated[key] = step_count
        elif key == "top_k":
            validated[key] = _require_int(raw, name, minimum=0)
        else:
            raise TrainingValidationError(f"{name} is not an allowed GRPO setting")
    _validate_json_value(validated, "trainer.grpoConfig")
    if "generation_batch_size" in validated and "steps_per_generation" in validated:
        raise TrainingValidationError(
            "trainer.grpoConfig.generation_batch_size and steps_per_generation are mutually exclusive"
        )
    if "generation_batch_size" in validated and "num_generations" in validated:
        if validated["generation_batch_size"] % validated["num_generations"] != 0:
            raise TrainingValidationError(
                "trainer.grpoConfig.generation_batch_size must be divisible by num_generations"
            )
    return validated


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and strictly validate an RL manifest without importing plugins or ML packages."""

    manifest_path = Path(path).expanduser()
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrainingValidationError(f"could not read manifest {manifest_path}: {exc}") from exc
    manifest = _require_object(_read_json(text, source=str(manifest_path)), "manifest")
    _require_exact_keys(
        manifest,
        required={
            "schemaVersion",
            "kind",
            "algorithm",
            "baseModel",
            "source",
            "dataset",
            "environment",
            "trainer",
            "contamination",
        },
        name="manifest",
    )
    if _require_int(manifest["schemaVersion"], "manifest.schemaVersion") != SCHEMA_VERSION:
        raise TrainingValidationError(
            f"manifest.schemaVersion must be exactly {SCHEMA_VERSION}"
        )
    if manifest["kind"] != MANIFEST_KIND:
        raise TrainingValidationError(f"manifest.kind must be {MANIFEST_KIND!r}")
    if manifest["algorithm"] != ALGORITHM:
        raise TrainingValidationError(f"manifest.algorithm must be {ALGORITHM!r}")
    _require_string(manifest["baseModel"], "manifest.baseModel")

    source = _require_object(manifest["source"], "source")
    _require_exact_keys(source, required={"bundleRoot", "taskCount"}, name="source")
    _absolute_existing_directory(source["bundleRoot"], "source.bundleRoot")
    task_count = _require_int(source["taskCount"], "source.taskCount", minimum=1)

    dataset = _require_object(manifest["dataset"], "dataset")
    _require_exact_keys(
        dataset,
        required={"format", "path", "examples", "columns"},
        name="dataset",
    )
    if dataset["format"] != DATASET_FORMAT:
        raise TrainingValidationError(f"dataset.format must be {DATASET_FORMAT!r}")
    _absolute_existing_file(dataset["path"], "dataset.path")
    examples = _require_int(dataset["examples"], "dataset.examples", minimum=1)
    if examples != task_count:
        raise TrainingValidationError("dataset.examples must equal source.taskCount")
    if dataset["columns"] != ["prompt", "task_id", "bundle_ref"]:
        raise TrainingValidationError(
            "dataset.columns must be exactly ['prompt', 'task_id', 'bundle_ref']"
        )

    environment = _require_object(manifest["environment"], "environment")
    _require_exact_keys(
        environment,
        required={"contractVersion", "factory", "config"},
        name="environment",
    )
    if _require_int(environment["contractVersion"], "environment.contractVersion") != 1:
        raise TrainingValidationError("environment.contractVersion must be exactly 1")
    _validate_factory_spec(environment["factory"])
    environment_config = _require_object(environment["config"], "environment.config")
    _validate_json_value(environment_config, "environment.config")

    trainer = _require_object(manifest["trainer"], "trainer")
    _require_exact_keys(
        trainer,
        required={"implementation", "trlVersion", "outputDir", "grpoConfig"},
        name="trainer",
    )
    if trainer["implementation"] != TRAINER_IMPLEMENTATION:
        raise TrainingValidationError(
            f"trainer.implementation must be {TRAINER_IMPLEMENTATION!r}"
        )
    if trainer["trlVersion"] != REQUIRED_TRL_VERSION:
        raise TrainingValidationError(
            f"trainer.trlVersion must be exactly {REQUIRED_TRL_VERSION!r}"
        )
    _absolute_output_path(trainer["outputDir"], "trainer.outputDir")
    validate_grpo_config(trainer["grpoConfig"])

    contamination = _require_object(manifest["contamination"], "contamination")
    _require_exact_keys(
        contamination,
        required={"benchmarkDerived", "acknowledged", "warning"},
        name="contamination",
    )
    if _require_bool(
        contamination["benchmarkDerived"], "contamination.benchmarkDerived"
    ) is not True:
        raise TrainingValidationError("contamination.benchmarkDerived must be true")
    if _require_bool(
        contamination["acknowledged"], "contamination.acknowledged"
    ) is not True:
        raise TrainingValidationError("contamination.acknowledged must be true")
    _require_string(contamination["warning"], "contamination.warning")
    return manifest


def _validate_factory_spec(value: Any) -> tuple[str, tuple[str, ...]]:
    spec = _require_string(value, "environment.factory")
    if spec.count(":") != 1:
        raise TrainingValidationError(
            "environment.factory must use the form module:callable"
        )
    module_name, attribute_name = spec.split(":", 1)
    module_parts = module_name.split(".")
    attribute_parts = attribute_name.split(".")
    if not module_name or not all(part.isidentifier() for part in module_parts):
        raise TrainingValidationError("environment.factory has an invalid module name")
    if not attribute_name or not all(part.isidentifier() for part in attribute_parts):
        raise TrainingValidationError("environment.factory has an invalid callable name")
    return module_name, tuple(attribute_parts)


def _safe_bundle_path(bundle_root: Path, bundle_ref: str, *, row_number: int) -> Path:
    ref_path = Path(bundle_ref)
    if ref_path.is_absolute():
        raise TrainingValidationError(f"prompts row {row_number} bundle_ref must be relative")
    if bundle_ref != "." and (
        not bundle_ref or any(part in {"", ".", ".."} for part in ref_path.parts)
    ):
        raise TrainingValidationError(
            f"prompts row {row_number} bundle_ref is not a normalized relative path"
        )
    try:
        candidate = (bundle_root / ref_path).resolve(strict=True)
    except OSError as exc:
        raise TrainingValidationError(
            f"prompts row {row_number} bundle_ref does not exist: {bundle_ref!r}"
        ) from exc
    try:
        candidate.relative_to(bundle_root)
    except ValueError as exc:
        raise TrainingValidationError(
            f"prompts row {row_number} bundle_ref escapes source.bundleRoot"
        ) from exc
    if not candidate.is_dir():
        raise TrainingValidationError(
            f"prompts row {row_number} bundle_ref must identify a task bundle directory"
        )
    task_json = candidate / "task.json"
    if not task_json.is_file():
        raise TrainingValidationError(
            f"prompts row {row_number} bundle_ref has no task.json"
        )
    return candidate


def load_prompt_rows(
    dataset_path: str | os.PathLike[str],
    bundle_root: str | os.PathLike[str],
    expected_examples: int,
) -> list[dict[str, Any]]:
    """Strictly load conversational prompt rows and validate bundle containment."""

    dataset = Path(dataset_path).expanduser().resolve(strict=True)
    root = Path(bundle_root).expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    seen_bundle_paths: set[Path] = set()
    previous_task_id: str | None = None
    try:
        stream = dataset.open("r", encoding="utf-8")
    except OSError as exc:
        raise TrainingValidationError(f"could not read prompt dataset {dataset}: {exc}") from exc
    with stream:
        for row_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                raise TrainingValidationError(f"prompts row {row_number} must not be blank")
            row = _require_object(
                _read_json(raw_line, source=f"{dataset} row {row_number}"),
                f"prompts row {row_number}",
            )
            _require_exact_keys(
                row,
                required={"prompt", "task_id", "bundle_ref"},
                name=f"prompts row {row_number}",
            )
            task_id = _require_string(row["task_id"], f"prompts row {row_number} task_id")
            bundle_ref = _require_string(
                row["bundle_ref"], f"prompts row {row_number} bundle_ref"
            )
            prompt = row["prompt"]
            if not isinstance(prompt, list) or len(prompt) != 1:
                raise TrainingValidationError(
                    f"prompts row {row_number} prompt must contain exactly one message"
                )
            message = _require_object(prompt[0], f"prompts row {row_number} prompt[0]")
            _require_exact_keys(
                message,
                required={"role", "content"},
                name=f"prompts row {row_number} prompt[0]",
            )
            if message["role"] != "user":
                raise TrainingValidationError(
                    f"prompts row {row_number} prompt[0].role must be 'user'"
                )
            _require_string(
                message["content"], f"prompts row {row_number} prompt[0].content"
            )
            if task_id in seen_task_ids:
                raise TrainingValidationError(f"duplicate task_id {task_id!r}")
            if previous_task_id is not None and task_id <= previous_task_id:
                raise TrainingValidationError("prompt rows must be strictly sorted by task_id")
            bundle_path = _safe_bundle_path(root, bundle_ref, row_number=row_number)
            if bundle_path in seen_bundle_paths:
                raise TrainingValidationError(
                    f"duplicate bundle_ref target in prompts row {row_number}: {bundle_ref!r}"
                )
            try:
                bundle_task = _require_object(
                    _read_json(
                        (bundle_path / "task.json").read_text(encoding="utf-8"),
                        source=f"{bundle_path / 'task.json'}",
                    ),
                    f"bundle task for prompts row {row_number}",
                )
            except OSError as exc:
                raise TrainingValidationError(
                    f"could not read bundle task for prompts row {row_number}: {exc}"
                ) from exc
            if bundle_task.get("id") != task_id:
                raise TrainingValidationError(
                    f"prompts row {row_number} task_id does not match bundle task.json"
                )
            if bundle_task.get("instruction") != message["content"]:
                raise TrainingValidationError(
                    f"prompts row {row_number} content does not match bundle instruction"
                )
            seen_task_ids.add(task_id)
            seen_bundle_paths.add(bundle_path)
            previous_task_id = task_id
            rows.append(row)
    if len(rows) != expected_examples:
        raise TrainingValidationError(
            f"dataset contains {len(rows)} prompt rows; manifest declares {expected_examples}"
        )
    return rows


def resolve_environment_factory(
    spec: str, bundle_root: str | os.PathLike[str], config: Mapping[str, Any]
) -> functools.partial[Any]:
    """Import a trusted plugin and bind its runner-owned keyword arguments."""

    module_name, attribute_parts = _validate_factory_spec(spec)
    try:
        target: Any = importlib.import_module(module_name)
    except Exception as exc:
        raise TrainingValidationError(
            f"could not import environment factory module {module_name!r}: {exc}"
        ) from exc
    for part in attribute_parts:
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise TrainingValidationError(
                f"environment factory {spec!r} does not exist"
            ) from exc
    if not callable(target):
        raise TrainingValidationError(f"environment factory {spec!r} is not callable")
    root = Path(bundle_root).expanduser().resolve(strict=True)
    config_copy = copy.deepcopy(dict(config))
    try:
        inspect.signature(target).bind(bundle_root=root, config=config_copy)
    except (TypeError, ValueError) as exc:
        raise TrainingValidationError(
            "environment factory must accept keyword arguments bundle_root and config"
        ) from exc
    return functools.partial(target, bundle_root=root, config=config_copy)


def resolve_resume(
    output_dir: str | os.PathLike[str], selector: str = "auto"
) -> Path | None:
    """Resolve a resume selector without creating or modifying the output directory."""

    output = Path(output_dir).expanduser().resolve(strict=False)
    if output.exists() and not output.is_dir():
        raise TrainingValidationError(f"output directory path is not a directory: {output}")
    if selector == "none":
        if output.exists() and any(output.iterdir()):
            raise TrainingValidationError(
                f"--resume none requires an empty output directory: {output}"
            )
        return None
    if selector == "auto":
        if not output.exists() or not any(output.iterdir()):
            return None
        checkpoints: list[tuple[int, Path]] = []
        for child in output.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", child.name)
            if not match or not child.is_dir():
                continue
            if (child / "trainer_state.json").is_file():
                checkpoints.append((int(match.group(1)), child.resolve(strict=True)))
        if not checkpoints:
            raise TrainingValidationError(
                f"--resume auto found a nonempty output directory but no valid checkpoint-N: {output}"
            )
        return max(checkpoints, key=lambda item: item[0])[1]
    checkpoint = Path(selector).expanduser()
    try:
        checkpoint = checkpoint.resolve(strict=True)
    except OSError as exc:
        raise TrainingValidationError(f"resume checkpoint does not exist: {checkpoint}") from exc
    if not checkpoint.is_dir() or not (checkpoint / "trainer_state.json").is_file():
        raise TrainingValidationError(
            f"resume checkpoint must contain trainer_state.json: {checkpoint}"
        )
    return checkpoint


def _release_tuple(version: str) -> tuple[tuple[int, int, int], bool]:
    match = re.fullmatch(
        r"\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?([A-Za-z0-9.+_-]*)\s*",
        version,
    )
    if not match:
        raise TrainingValidationError(f"could not interpret installed version {version!r}")
    release = tuple(int(match.group(index) or 0) for index in range(1, 4))
    suffix = match.group(4).lower()
    if suffix and not re.fullmatch(
        r"(?:(?:a|b|rc)\d+|\.?post\d+|\.?dev\d+)?"
        r"(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?",
        suffix,
    ):
        raise TrainingValidationError(f"could not interpret installed version {version!r}")
    is_prerelease = bool(re.search(r"(?:a|b|rc)\d+|\.?dev\d+", suffix))
    return release, is_prerelease


def _require_training_versions() -> tuple[str, str]:
    try:
        trl_version = importlib.metadata.version("trl")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TrainingValidationError(
            f"TRL is not installed. Install the training dependencies with: {INSTALL_COMMAND}"
        ) from exc
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TrainingValidationError(
            f"Transformers is not installed. Install the training dependencies with: {INSTALL_COMMAND}"
        ) from exc
    if trl_version != REQUIRED_TRL_VERSION:
        raise TrainingValidationError(
            f"TRL {REQUIRED_TRL_VERSION} is required, but {trl_version} is installed. "
            f"Install the required versions with: {INSTALL_COMMAND}"
        )
    release, prerelease = _release_tuple(transformers_version)
    if release < TRANSFORMERS_MIN_VERSION or release[0] >= TRANSFORMERS_MAX_MAJOR or (
        release == TRANSFORMERS_MIN_VERSION and prerelease
    ):
        raise TrainingValidationError(
            "Transformers >=5.2.0,<6 is required, but "
            f"{transformers_version} is installed. Install the required versions with: "
            f"{INSTALL_COMMAND}"
        )
    return trl_version, transformers_version


def _validated_inputs(
    manifest_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    resume: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path | None, Path]:
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = load_manifest(manifest_file)
    source = manifest["source"]
    dataset = manifest["dataset"]
    rows = load_prompt_rows(
        dataset["path"], source["bundleRoot"], dataset["examples"]
    )
    output = Path(output_dir).expanduser().resolve(strict=False)
    selected_checkpoint = resolve_resume(output, resume)
    return manifest, rows, output, selected_checkpoint, manifest_file


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _resolved_run_metadata(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    output_dir: Path,
    checkpoint: Path | None,
    trl_version: str,
    transformers_version: str,
    examples: int,
) -> dict[str, Any]:
    environment = manifest["environment"]
    trainer = manifest["trainer"]
    return {
        "schemaVersion": 1,
        "kind": "localmaxxing.eval_rl_grpo.resolved_run",
        "resolvedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest": str(manifest_path),
        "outputDir": str(output_dir),
        "baseModel": manifest["baseModel"],
        "dataset": {
            "path": manifest["dataset"]["path"],
            "examples": examples,
        },
        "environment": {
            "contractVersion": environment["contractVersion"],
            "factory": environment["factory"],
        },
        "trainer": {
            "implementation": trainer["implementation"],
            "trlVersion": trl_version,
            "transformersVersion": transformers_version,
            "resumeFromCheckpoint": str(checkpoint) if checkpoint is not None else None,
        },
    }


def run_training(
    manifest_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    resume: str = "auto",
) -> dict[str, Any]:
    """Validate inputs, lazily load training code and plugin, then train and save."""

    manifest, rows, output, checkpoint, manifest_file = _validated_inputs(
        manifest_path, output_dir, resume
    )
    trl_version, transformers_version = _require_training_versions()
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise TrainingValidationError(
            f"Could not import TRL training dependencies: {exc}. Install them with: {INSTALL_COMMAND}"
        ) from exc

    environment = manifest["environment"]
    factory = resolve_environment_factory(
        environment["factory"],
        manifest["source"]["bundleRoot"],
        environment["config"],
    )
    config_values = validate_grpo_config(manifest["trainer"]["grpoConfig"])
    config_values.update(
        {
            "output_dir": str(output),
            "report_to": "none",
            "log_completions": False,
            "remove_unused_columns": False,
        }
    )
    dataset = Dataset.from_list(copy.deepcopy(rows))
    try:
        training_args = GRPOConfig(**config_values)
    except (TypeError, ValueError) as exc:
        raise TrainingValidationError(f"invalid TRL GRPO configuration: {exc}") from exc
    trainer = GRPOTrainer(
        model=manifest["baseModel"],
        args=training_args,
        train_dataset=dataset,
        environment_factory=factory,
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata = _resolved_run_metadata(
        manifest_path=manifest_file,
        manifest=manifest,
        output_dir=output,
        checkpoint=checkpoint,
        trl_version=trl_version,
        transformers_version=transformers_version,
        examples=len(rows),
    )
    _write_json_atomic(output / "resolved-run.json", metadata)
    trainer.train(
        resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None
    )
    trainer.save_model(str(output))
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.validate_only:
            manifest, rows, output, checkpoint, manifest_file = _validated_inputs(
                args.manifest, args.output_dir, args.resume
            )
            summary = {
                "valid": True,
                "manifest": str(manifest_file),
                "dataset": manifest["dataset"]["path"],
                "examples": len(rows),
                "outputDir": str(output),
                "resumeFromCheckpoint": (
                    str(checkpoint) if checkpoint is not None else None
                ),
            }
            print(json.dumps(summary, sort_keys=True))
            return 0
        run_training(args.manifest, args.output_dir, args.resume)
        return 0
    except (TrainingValidationError, OSError) as exc:
        print(f"train_eval_grpo: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
