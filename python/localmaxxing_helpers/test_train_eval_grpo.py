import contextlib
import copy
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from python.localmaxxing_helpers import train_eval_grpo as trainer


class GRPOFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.bundle_root = self.root / "bundles"
        self.bundle_root.mkdir()
        self.output_dir = self.root / "output"
        self.dataset_path = self.root / "prompts.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.instructions = {
            "task-a": "Repair task A without changing its public API.",
            "task-b": "Diagnose task B and submit the verified fix.",
        }
        self.make_bundle("bundle-a", "task-a", self.instructions["task-a"])
        self.make_bundle("bundle-b", "task-b", self.instructions["task-b"])
        self.rows = [
            {
                "prompt": [{"role": "user", "content": self.instructions["task-a"]}],
                "task_id": "task-a",
                "bundle_ref": "bundle-a",
            },
            {
                "prompt": [{"role": "user", "content": self.instructions["task-b"]}],
                "task_id": "task-b",
                "bundle_ref": "bundle-b",
            },
        ]
        self.write_rows(self.rows)
        self.manifest = {
            "schemaVersion": 1,
            "kind": trainer.MANIFEST_KIND,
            "algorithm": trainer.ALGORITHM,
            "baseModel": "example/model",
            "source": {
                "bundleRoot": str(self.bundle_root),
                "taskCount": 2,
            },
            "dataset": {
                "format": trainer.DATASET_FORMAT,
                "path": str(self.dataset_path),
                "examples": 2,
                "columns": ["prompt", "task_id", "bundle_ref"],
            },
            "environment": {
                "contractVersion": 1,
                "factory": "trusted_env:create_environment",
                "config": {"sandbox": {"timeout": 30}, "secret": "do-not-persist"},
            },
            "trainer": {
                "implementation": trainer.TRAINER_IMPLEMENTATION,
                "trlVersion": trainer.REQUIRED_TRL_VERSION,
                "outputDir": str(self.output_dir),
                "grpoConfig": {
                    "bf16": True,
                    "learning_rate": 0.00001,
                    "num_generations": 4,
                },
            },
            "contamination": {
                "benchmarkDerived": True,
                "acknowledged": True,
                "warning": "Benchmark-derived tasks must not be reported as uncontaminated evaluation.",
            },
        }
        self.write_manifest()

    def make_bundle(self, ref, task_id, instruction, *, parent=None):
        directory = (parent or self.bundle_root) / ref
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "task.json").write_text(
            json.dumps({"id": task_id, "instruction": instruction}) + "\n",
            encoding="utf-8",
        )
        return directory

    def write_rows(self, rows):
        self.dataset_path.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def write_manifest(self, manifest=None):
        value = self.manifest if manifest is None else manifest
        self.manifest_path.write_text(
            json.dumps(value, allow_nan=False) + "\n", encoding="utf-8"
        )

    def load_rows(self, expected=2):
        return trainer.load_prompt_rows(
            self.dataset_path, self.bundle_root, expected_examples=expected
        )


class ManifestValidationTest(GRPOFixture):
    def test_valid_manifest_is_accepted_without_importing_plugin(self):
        loaded = trainer.load_manifest(self.manifest_path)
        self.assertEqual(loaded, self.manifest)
        self.assertNotIn("trusted_env", sys.modules)

    def test_manifest_rejects_removed_max_prompt_length_setting(self):
        self.manifest["trainer"]["grpoConfig"]["max_prompt_length"] = 4096
        self.write_manifest()
        with self.assertRaisesRegex(
            trainer.TrainingValidationError,
            r"trainer\.grpoConfig\.max_prompt_length is not an allowed GRPO setting",
        ):
            trainer.load_manifest(self.manifest_path)


    def test_manifest_requires_exact_keys_at_every_level(self):
        containers = [
            (),
            ("source",),
            ("dataset",),
            ("environment",),
            ("trainer",),
            ("contamination",),
        ]
        for path in containers:
            with self.subTest(path=path):
                value = copy.deepcopy(self.manifest)
                target = value
                for part in path:
                    target = target[part]
                target["unexpected"] = True
                self.write_manifest(value)
                name = "manifest" if not path else path[-1]
                with self.assertRaisesRegex(
                    trainer.TrainingValidationError,
                    rf"{name} has unknown fields: unexpected",
                ):
                    trainer.load_manifest(self.manifest_path)

    def test_manifest_rejects_missing_fields_duplicate_keys_and_nonfinite_json(self):
        missing = copy.deepcopy(self.manifest)
        del missing["environment"]["factory"]
        self.write_manifest(missing)
        with self.assertRaisesRegex(
            trainer.TrainingValidationError, "environment is missing fields: factory"
        ):
            trainer.load_manifest(self.manifest_path)

        self.manifest_path.write_text(
            '{"schemaVersion":1,"schemaVersion":1}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(
            trainer.TrainingValidationError, "duplicate JSON object key 'schemaVersion'"
        ):
            trainer.load_manifest(self.manifest_path)

        raw = json.dumps(self.manifest).replace(
            '"timeout": 30', '"timeout": NaN'
        )
        self.manifest_path.write_text(raw + "\n", encoding="utf-8")
        with self.assertRaisesRegex(
            trainer.TrainingValidationError, "non-finite JSON number 'NaN'"
        ):
            trainer.load_manifest(self.manifest_path)

    def test_manifest_rejects_wrong_scalar_types_including_bool_as_int(self):
        cases = [
            (("schemaVersion",), True, "manifest.schemaVersion must be an integer"),
            (("baseModel",), 7, "manifest.baseModel must be a string"),
            (("source", "taskCount"), True, "source.taskCount must be an integer"),
            (("dataset", "examples"), 2.0, "dataset.examples must be an integer"),
            (
                ("environment", "contractVersion"),
                "1",
                "environment.contractVersion must be an integer",
            ),
            (
                ("trainer", "outputDir"),
                "relative/output",
                "trainer.outputDir must be an absolute path",
            ),
            (
                ("contamination", "acknowledged"),
                1,
                "contamination.acknowledged must be a boolean",
            ),
        ]
        for path, replacement, error in cases:
            with self.subTest(path=path, replacement=replacement):
                value = copy.deepcopy(self.manifest)
                target = value
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = replacement
                self.write_manifest(value)
                with self.assertRaisesRegex(
                    trainer.TrainingValidationError, error
                ):
                    trainer.load_manifest(self.manifest_path)

    def test_grpo_config_is_allowlisted_typed_finite_and_defensively_copied(self):
        original = {
            "bf16": True,
            "seed": 0,
            "num_generations": 2,
            "max_steps": -1,
            "top_p": 1.0,
            "generation_kwargs": {"stop_strings": ["END"]},
            "model_init_kwargs": None,
        }
        validated = trainer.validate_grpo_config(original)
        self.assertEqual(validated, original)
        validated["generation_kwargs"]["stop_strings"].append("MUTATED")
        self.assertEqual(original["generation_kwargs"], {"stop_strings": ["END"]})

        bad_cases = [
            ({"not_a_grpo_option": 1}, "not an allowed GRPO setting"),
            ({"output_dir": "/tmp/stolen"}, "controlled by the runner"),
            ({"bf16": 1}, "must be a boolean"),
            ({"seed": True}, "must be an integer"),
            ({"learning_rate": float("inf")}, "must be finite"),
            ({"beta": float("nan")}, "must be finite"),
            ({"num_generations": 1}, "must be at least 2"),
            ({"top_p": 1.01}, "must be between zero and one"),
            ({"generation_kwargs": []}, "must be a JSON object"),
            (
                {"generation_batch_size": 4, "steps_per_generation": 1},
                "mutually exclusive",
            ),
            (
                {"generation_batch_size": 3, "num_generations": 2},
                "must be divisible by num_generations",
            ),
        ]
        for value, error in bad_cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(trainer.TrainingValidationError, error):
                    trainer.validate_grpo_config(value)


class PromptRowsValidationTest(GRPOFixture):
    def test_prompt_rows_are_returned_exactly_without_normalization(self):
        loaded = self.load_rows()
        self.assertEqual(loaded, self.rows)
        self.assertIsInstance(loaded[0]["prompt"], list)
        self.assertEqual(
            loaded[0],
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Repair task A without changing its public API.",
                    }
                ],
                "task_id": "task-a",
                "bundle_ref": "bundle-a",
            },
        )

    def test_prompt_rows_require_exact_shape_and_count(self):
        cases = []
        unknown = copy.deepcopy(self.rows)
        unknown[0]["label"] = 1
        cases.append((unknown, "has unknown fields: label"))
        two_messages = copy.deepcopy(self.rows)
        two_messages[0]["prompt"].append({"role": "user", "content": "again"})
        cases.append((two_messages, "prompt must contain exactly one message"))
        extra_message_key = copy.deepcopy(self.rows)
        extra_message_key[0]["prompt"][0]["name"] = "caller"
        cases.append((extra_message_key, r"prompt\[0\] has unknown fields: name"))
        wrong_role = copy.deepcopy(self.rows)
        wrong_role[0]["prompt"][0]["role"] = "system"
        cases.append((wrong_role, r"prompt\[0\]\.role must be 'user'"))
        empty_content = copy.deepcopy(self.rows)
        empty_content[0]["prompt"][0]["content"] = "   "
        cases.append((empty_content, r"prompt\[0\]\.content must not be empty"))
        for rows, error in cases:
            with self.subTest(error=error):
                self.write_rows(rows)
                with self.assertRaisesRegex(trainer.TrainingValidationError, error):
                    self.load_rows()

        self.write_rows(self.rows[:1])
        with self.assertRaisesRegex(
            trainer.TrainingValidationError,
            "dataset contains 1 prompt rows; manifest declares 2",
        ):
            self.load_rows()

    def test_task_identity_instruction_and_sort_order_are_enforced(self):
        wrong_id = copy.deepcopy(self.rows)
        wrong_id[0]["task_id"] = "task-0"
        self.write_rows(wrong_id)
        with self.assertRaisesRegex(
            trainer.TrainingValidationError,
            "task_id does not match bundle task.json",
        ):
            self.load_rows()

        wrong_instruction = copy.deepcopy(self.rows)
        wrong_instruction[0]["prompt"][0]["content"] = "A rewritten label"
        self.write_rows(wrong_instruction)
        with self.assertRaisesRegex(
            trainer.TrainingValidationError,
            "content does not match bundle instruction",
        ):
            self.load_rows()

        self.write_rows(list(reversed(self.rows)))
        with self.assertRaisesRegex(
            trainer.TrainingValidationError,
            "prompt rows must be strictly sorted by task_id",
        ):
            self.load_rows()

    def test_duplicate_task_ids_and_bundle_targets_are_rejected_explicitly(self):
        duplicate_id = [copy.deepcopy(self.rows[0]), copy.deepcopy(self.rows[0])]
        self.write_rows(duplicate_id)
        with self.assertRaisesRegex(trainer.TrainingValidationError, "duplicate task_id"):
            self.load_rows()

        alias = self.bundle_root / "bundle-a-alias"
        try:
            alias.symlink_to(self.bundle_root / "bundle-a", target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")
        duplicate_target = copy.deepcopy(self.rows)
        duplicate_target[1]["bundle_ref"] = alias.name
        self.write_rows(duplicate_target)
        with self.assertRaisesRegex(
            trainer.TrainingValidationError, "duplicate bundle_ref target"
        ):
            self.load_rows()

    def test_bundle_refs_reject_traversal_absolute_and_symlink_escape(self):
        outside = self.root / "outside"
        self.make_bundle("escaped", "task-a", self.instructions["task-a"], parent=outside)
        cases = ["../outside/escaped", str(outside / "escaped"), "bundle-a/../bundle-a"]
        for bundle_ref in cases:
            with self.subTest(bundle_ref=bundle_ref):
                rows = copy.deepcopy(self.rows)
                rows[0]["bundle_ref"] = bundle_ref
                self.write_rows(rows)
                with self.assertRaisesRegex(
                    trainer.TrainingValidationError,
                    "must be relative|not a normalized relative path",
                ):
                    self.load_rows()

        link = self.bundle_root / "outside-link"
        try:
            link.symlink_to(outside / "escaped", target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")
        rows = copy.deepcopy(self.rows)
        rows[0]["bundle_ref"] = link.name
        self.write_rows(rows)
        with self.assertRaisesRegex(
            trainer.TrainingValidationError, "escapes source.bundleRoot"
        ):
            self.load_rows()


class FactoryResolutionTest(GRPOFixture):
    def test_factory_syntax_is_strict(self):
        cases = [
            ("module", "form module:callable"),
            ("module:factory:extra", "form module:callable"),
            ("bad-module:factory", "invalid module name"),
            ("module:bad-name", "invalid callable name"),
            (":factory", "invalid module name"),
            ("module:", "invalid callable name"),
        ]
        for spec, error in cases:
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(trainer.TrainingValidationError, error):
                    trainer._validate_factory_spec(spec)

    def test_factory_resolution_binds_resolved_root_and_defensive_config(self):
        module = types.ModuleType("trusted_test_environment")
        received = []

        def create(*, bundle_root, config):
            received.append((bundle_root, config))
            config["nested"]["values"].append("factory-mutated")
            return "environment-instance"

        module.namespace = types.SimpleNamespace(create=create)
        supplied = {"nested": {"values": ["original"]}}
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            factory = trainer.resolve_environment_factory(
                "trusted_test_environment:namespace.create",
                self.bundle_root / ".",
                supplied,
            )
            self.assertEqual(factory(), "environment-instance")
        self.assertEqual(received[0][0], self.bundle_root.resolve())
        self.assertEqual(received[0][1]["nested"]["values"], ["original", "factory-mutated"])
        self.assertEqual(supplied, {"nested": {"values": ["original"]}})

    def test_factory_resolution_reports_import_attribute_callable_and_signature_errors(self):
        module = types.ModuleType("bad_test_environment")
        module.not_callable = 3

        def wrong_signature(required_positional):
            return required_positional

        module.wrong_signature = wrong_signature
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(
                trainer.TrainingValidationError, "does not exist"
            ):
                trainer.resolve_environment_factory(
                    "bad_test_environment:missing", self.bundle_root, {}
                )
            with self.assertRaisesRegex(
                trainer.TrainingValidationError, "is not callable"
            ):
                trainer.resolve_environment_factory(
                    "bad_test_environment:not_callable", self.bundle_root, {}
                )
            with self.assertRaisesRegex(
                trainer.TrainingValidationError,
                "must accept keyword arguments bundle_root and config",
            ):
                trainer.resolve_environment_factory(
                    "bad_test_environment:wrong_signature", self.bundle_root, {}
                )
        with self.assertRaisesRegex(
            trainer.TrainingValidationError,
            "could not import environment factory module",
        ):
            trainer.resolve_environment_factory(
                "module_that_does_not_exist_for_grpo_test:create", self.bundle_root, {}
            )


class ResumeAndValidationOnlyTest(GRPOFixture):
    def test_resume_auto_selects_highest_valid_numeric_checkpoint(self):
        self.assertIsNone(trainer.resolve_resume(self.output_dir, "auto"))
        self.output_dir.mkdir()
        (self.output_dir / "notes.txt").write_text("ignore", encoding="utf-8")
        for name in ("checkpoint-2", "checkpoint-11", "checkpoint-100-invalid"):
            (self.output_dir / name).mkdir()
        for name in ("checkpoint-2", "checkpoint-11"):
            (self.output_dir / name / "trainer_state.json").write_text("{}", encoding="utf-8")
        selected = trainer.resolve_resume(self.output_dir, "auto")
        self.assertEqual(selected, (self.output_dir / "checkpoint-11").resolve())

    def test_resume_auto_rejects_nonempty_output_without_checkpoint(self):
        self.output_dir.mkdir()
        (self.output_dir / "artifact.bin").write_bytes(b"existing")
        with self.assertRaisesRegex(
            trainer.TrainingValidationError, "no valid checkpoint-N"
        ):
            trainer.resolve_resume(self.output_dir, "auto")

    def test_resume_none_requires_empty_output_and_explicit_requires_state(self):
        self.assertIsNone(trainer.resolve_resume(self.output_dir, "none"))
        self.output_dir.mkdir()
        self.assertIsNone(trainer.resolve_resume(self.output_dir, "none"))
        checkpoint = self.root / "manual-checkpoint"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
        self.assertEqual(
            trainer.resolve_resume(self.output_dir, str(checkpoint)), checkpoint.resolve()
        )
        (self.output_dir / "occupied").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(
            trainer.TrainingValidationError, "requires an empty output directory"
        ):
            trainer.resolve_resume(self.output_dir, "none")
        invalid = self.root / "invalid-checkpoint"
        invalid.mkdir()
        with self.assertRaisesRegex(
            trainer.TrainingValidationError, "must contain trainer_state.json"
        ):
            trainer.resolve_resume(self.output_dir, str(invalid))

    def test_validate_only_is_lazy_and_does_not_create_training_output(self):
        self.manifest["environment"]["factory"] = (
            "plugin_intentionally_absent_during_validation:create"
        )
        self.write_manifest()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = trainer.main(
                [
                    "--manifest",
                    str(self.manifest_path),
                    "--output-dir",
                    str(self.output_dir),
                    "--resume",
                    "none",
                    "--validate-only",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["examples"], 2)
        self.assertIsNone(summary["resumeFromCheckpoint"])
        self.assertFalse(self.output_dir.exists())
        self.assertNotIn("plugin_intentionally_absent_during_validation", sys.modules)
        self.assertNotIn("trl", sys.modules)
        self.assertNotIn("datasets", sys.modules)


class TrainingOrchestrationTest(GRPOFixture):
    def test_fake_training_stack_proves_online_environment_orchestration(self):
        checkpoint = self.output_dir / "checkpoint-7"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
        capture = {}

        datasets_module = types.ModuleType("datasets")

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self.column_names = list(rows[0])

            @classmethod
            def from_list(cls, rows):
                capture["dataset_rows"] = rows
                dataset = cls(rows)
                capture["dataset"] = dataset
                rows[0]["task_id"] = "dataset-private-mutation"
                return dataset

        datasets_module.Dataset = FakeDataset
        trl_module = types.ModuleType("trl")

        class FakeGRPOConfig:
            def __init__(self, **kwargs):
                capture["config"] = kwargs

        class FakeGRPOTrainer:
            def __init__(self, **kwargs):
                capture["trainer_kwargs"] = kwargs

            def train(self, **kwargs):
                capture["train_kwargs"] = kwargs
                capture["metadata_existed_at_train"] = (
                    self_output / "resolved-run.json"
                ).is_file()

            def save_model(self, path):
                capture["save_path"] = path

        trl_module.GRPOConfig = FakeGRPOConfig
        trl_module.GRPOTrainer = FakeGRPOTrainer
        environment_module = types.ModuleType("trusted_env")

        def create_environment(*, bundle_root, config):
            return {"root": bundle_root, "config": config}

        environment_module.create_environment = create_environment
        self_output = self.output_dir

        def installed_version(distribution):
            return {"trl": "1.8.0", "transformers": "5.2.3"}[distribution]

        fake_modules = {
            "datasets": datasets_module,
            "trl": trl_module,
            "trusted_env": environment_module,
        }
        with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
            importlib.metadata, "version", side_effect=installed_version
        ):
            metadata = trainer.run_training(
                self.manifest_path, self.output_dir, resume="auto"
            )

        self.assertEqual(
            capture["dataset_rows"][1], self.rows[1]
        )
        self.assertEqual(self.rows[0]["task_id"], "task-a")
        self.assertEqual(
            capture["config"],
            {
                "bf16": True,
                "learning_rate": 0.00001,
                "num_generations": 4,
                "output_dir": str(self.output_dir.resolve()),
                "report_to": "none",
                "log_completions": False,
                "remove_unused_columns": False,
            },
        )
        kwargs = capture["trainer_kwargs"]
        self.assertEqual(set(kwargs), {"model", "args", "train_dataset", "environment_factory"})
        self.assertNotIn("reward_funcs", kwargs)
        self.assertNotIn("reward_function", kwargs)
        self.assertEqual(kwargs["model"], "example/model")
        dataset = kwargs["train_dataset"]
        self.assertIs(dataset, capture["dataset"])
        self.assertEqual(dataset.column_names, ["prompt", "task_id", "bundle_ref"])
        self.assertEqual(
            [list(row) for row in dataset.rows],
            [["prompt", "task_id", "bundle_ref"]] * 2,
        )
        factory = kwargs["environment_factory"]
        self.assertIs(factory.func, create_environment)
        self.assertEqual(factory.keywords["bundle_root"], self.bundle_root.resolve())
        self.assertEqual(factory.keywords["config"], self.manifest["environment"]["config"])
        factory.keywords["config"]["sandbox"]["timeout"] = 999
        self.assertEqual(self.manifest["environment"]["config"]["sandbox"]["timeout"], 30)
        self.assertEqual(
            capture["train_kwargs"],
            {"resume_from_checkpoint": str(checkpoint.resolve())},
        )
        self.assertTrue(capture["metadata_existed_at_train"])
        self.assertEqual(capture["save_path"], str(self.output_dir.resolve()))
        self.assertEqual(metadata["trainer"]["trlVersion"], "1.8.0")
        self.assertEqual(metadata["trainer"]["transformersVersion"], "5.2.3")

        persisted = json.loads(
            (self.output_dir / "resolved-run.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted, metadata)
        self.assertNotIn("config", persisted["environment"])
        serialized = json.dumps(persisted, sort_keys=True)
        self.assertNotIn("do-not-persist", serialized)
        self.assertNotIn("timeout", serialized)


class VersionValidationTest(unittest.TestCase):
    def version_lookup(self, *, trl="1.8.0", transformers="5.2.0"):
        def lookup(distribution):
            value = {"trl": trl, "transformers": transformers}[distribution]
            if isinstance(value, BaseException):
                raise value
            return value

        return lookup

    def test_exact_trl_and_supported_transformers_versions_are_required(self):
        with mock.patch.object(
            importlib.metadata,
            "version",
            side_effect=self.version_lookup(transformers="5.9.1"),
        ):
            self.assertEqual(trainer._require_training_versions(), ("1.8.0", "5.9.1"))

        cases = [
            ("1.7.0", "5.2.0", "TRL 1.8.0 is required"),
            ("1.8.0", "5.1.9", "Transformers >=5.2.0,<6 is required"),
            ("1.8.0", "5.2.0rc1", "Transformers >=5.2.0,<6 is required"),
            ("1.8.0", "6.0.0", "Transformers >=5.2.0,<6 is required"),
            ("1.8.0", "not-a-version", "could not interpret installed version"),
        ]
        for trl_version, transformers_version, error in cases:
            with self.subTest(trl=trl_version, transformers=transformers_version):
                with mock.patch.object(
                    importlib.metadata,
                    "version",
                    side_effect=self.version_lookup(
                        trl=trl_version, transformers=transformers_version
                    ),
                ):
                    with self.assertRaisesRegex(trainer.TrainingValidationError, error):
                        trainer._require_training_versions()

    def test_missing_packages_have_actionable_errors(self):
        missing = importlib.metadata.PackageNotFoundError("missing")
        with mock.patch.object(
            importlib.metadata,
            "version",
            side_effect=self.version_lookup(trl=missing),
        ):
            with self.assertRaisesRegex(
                trainer.TrainingValidationError,
                "TRL is not installed.*python -m pip install",
            ):
                trainer._require_training_versions()
        with mock.patch.object(
            importlib.metadata,
            "version",
            side_effect=self.version_lookup(transformers=missing),
        ):
            with self.assertRaisesRegex(
                trainer.TrainingValidationError,
                "Transformers is not installed.*python -m pip install",
            ):
                trainer._require_training_versions()


if __name__ == "__main__":
    unittest.main()
