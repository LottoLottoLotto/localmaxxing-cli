import unittest

from python.localmaxxing_helpers import train_eval_sft as trainer


class FakeTokenizer:
    role_tokens = {"system": 10, "user": 20, "assistant": 30, "tool": 40}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.last_messages = messages
        tokens = []
        for message in messages:
            tokens.extend([self.role_tokens[message["role"]], len(message.get("content") or "")])
            if message.get("tool_calls"):
                tokens.append(50)
        return tokens if tokenize else "rendered"


class TrainEvalSFTTest(unittest.TestCase):
    def test_assistant_only_labels_preserve_tool_context(self):
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "fix"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "pwd"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": "/app"},
                    {"role": "assistant", "content": "done"},
                ]
            }
        ]

        encoded = trainer.tokenize_training_rows(
            FakeTokenizer(), rows, max_length=100, assistant_only=True
        )

        self.assertEqual(encoded["input_ids"][0], [10, 5, 20, 3, 30, 0, 50, 40, 4, 30, 4])
        self.assertEqual(
            encoded["labels"][0],
            [-100, -100, -100, -100, 30, 0, 50, -100, -100, 30, 4],
        )
        self.assertEqual(encoded["attention_mask"][0], [1] * 11)

    def test_full_sequence_labels_every_token(self):
        rows = [{"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}]
        encoded = trainer.tokenize_training_rows(
            FakeTokenizer(), rows, max_length=100, assistant_only=False
        )
        self.assertEqual(encoded["labels"], encoded["input_ids"])

    def test_assistant_tokens_must_survive_truncation(self):
        rows = [{"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}]
        with self.assertRaisesRegex(SystemExit, "no assistant tokens remain"):
            trainer.tokenize_training_rows(
                FakeTokenizer(), rows, max_length=2, assistant_only=True
            )

    def test_target_modules_distinguish_dense_and_moe(self):
        dense = trainer.unsloth_target_modules("auto", "Qwen/Qwen3-27B")
        moe = trainer.unsloth_target_modules("auto", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        self.assertIn("gate_proj", dense)
        self.assertNotIn("gate_up_proj", dense)
        self.assertIn("gate_up_proj", moe)
        self.assertNotIn("gate_proj", moe)

    def test_explicit_target_modules_are_trimmed(self):
        self.assertEqual(
            trainer.unsloth_target_modules(" q_proj, down_proj ", "anything"),
            ["q_proj", "down_proj"],
        )


if __name__ == "__main__":
    unittest.main()
