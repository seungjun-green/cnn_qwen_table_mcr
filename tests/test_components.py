from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from src.checkpointing import load_training_checkpoint, save_training_checkpoint
from src.config import ExperimentConfig, load_config
from src.data import MRCBatchCollator, Table, build_prompt_ids, truncate_table
from src.diagnostics import build_table_shuffled_examples, table_dependence_metrics
from src.model import TableCNNQwen
from src.pooling import TokenPooler
from src.train import train_model
from src.utils import mirror_directory


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [2 + (sum(token.encode("utf-8")) % 27) for token in str(text).split()]

    def __call__(
        self,
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    ):
        del add_special_tokens, padding, truncation, return_tensors
        encoded = [self.encode(text)[:max_length] for text in texts]
        width = max(max((len(ids) for ids in encoded), default=0), 1)
        input_ids = torch.zeros(len(encoded), width, dtype=torch.long)
        mask = torch.zeros_like(input_ids)
        for index, ids in enumerate(encoded):
            if ids:
                input_ids[index, : len(ids)] = torch.tensor(ids)
                mask[index, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


class DummyLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden):
        return hidden + torch.tanh(self.linear(hidden))


class DummyLM(nn.Module):
    def __init__(self, hidden_size=16, vocab_size=32, layers=3):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, pad_token_id=0)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = SimpleNamespace()
        self.model.layers = nn.ModuleList(
            [DummyLayer(hidden_size) for _ in range(layers)]
        )
        self.layers = self.model.layers
        self.head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        del attention_mask, kwargs
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        del kwargs
        sequence = input_ids
        mask = attention_mask
        for _ in range(max_new_tokens):
            logits = self.forward(sequence, mask).logits
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            sequence = torch.cat([sequence, next_token], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)
        return sequence


class ComponentTests(unittest.TestCase):
    def test_prompt_can_explicitly_disable_qwen_thinking(self):
        class ChatTokenizer(DummyTokenizer):
            chat_template = "template"
            observed_enable_thinking = None

            def apply_chat_template(self, messages, **kwargs):
                del messages
                self.observed_enable_thinking = kwargs.get("enable_thinking")
                return [7, 8]

        tokenizer = ChatTokenizer()
        prompt = build_prompt_ids(tokenizer, "question", enable_thinking=False)
        self.assertEqual(prompt, [7, 8])
        self.assertFalse(tokenizer.observed_enable_thinking)

    def test_table_shuffling_and_dependence_metrics(self):
        examples = [
            {
                "question": f"q{index}",
                "answers": [str(index)],
                "table": {"header": ["h"], "rows": [[str(index)]]},
            }
            for index in range(4)
        ]
        shuffled = build_table_shuffled_examples(examples)
        for original, replacement in zip(examples, shuffled):
            self.assertEqual(original["question"], replacement["question"])
            self.assertNotEqual(original["table"], replacement["table"])
        correct_records = [
            {"prediction": "yes", "correct": True},
            {"prediction": "no", "correct": False},
        ]
        shuffled_records = [
            {"prediction": "no", "correct": False},
            {"prediction": "no", "correct": False},
        ]
        metrics = table_dependence_metrics(correct_records, shuffled_records)
        self.assertEqual(metrics["exact_match_drop"], 0.5)
        self.assertEqual(metrics["prediction_change_rate"], 0.5)

    def test_full_checkpoint_restores_model_optimizer_and_progress(self):
        config = ExperimentConfig()
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(torch.ones(1, 3)).sum()
        loss.backward()
        optimizer.step()
        expected_weight = model.weight.detach().clone()
        history = {"epochs": [], "loss_history": [{"loss": 1.0}]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint_last.pt"
            save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                epoch=2,
                next_batch_index=17,
                global_step=23,
                history=history,
                best_metric=0.25,
                epochs_without_improvement=1,
                epoch_loss_sum=9.5,
                epoch_batches_seen=17,
            )
            with torch.no_grad():
                model.weight.zero_()
            restored = load_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                torch.device("cpu"),
            )
            self.assertTrue(torch.equal(model.weight, expected_weight))
            self.assertEqual(restored["epoch"], 2)
            self.assertEqual(restored["next_batch_index"], 17)
            self.assertEqual(restored["global_step"], 23)
            self.assertEqual(restored["optimizer_state"]["state"].keys(), {0, 1})

    def test_output_directory_mirroring(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "local_run"
            destination = root / "drive_run"
            (source / "validation_epoch_1").mkdir(parents=True)
            (source / "checkpoint_last.pt").write_bytes(b"checkpoint")
            (source / "validation_epoch_1" / "metrics.json").write_text(
                '{"accuracy": 0.5}', encoding="utf-8"
            )
            mirror_directory(source, destination)
            self.assertEqual(
                (destination / "checkpoint_last.pt").read_bytes(), b"checkpoint"
            )
            self.assertTrue(
                (destination / "validation_epoch_1" / "metrics.json").is_file()
            )
            (source / "checkpoint_last.pt").write_bytes(b"new-checkpoint")
            mirror_directory(source, destination)
            self.assertEqual(
                (destination / "checkpoint_last.pt").read_bytes(), b"new-checkpoint"
            )

    def test_pooling_masks(self):
        embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])
        self.assertTrue(
            torch.equal(
                TokenPooler(2, "mean")(embeddings, mask), torch.tensor([[2.0, 3.0]])
            )
        )
        self.assertTrue(
            torch.equal(
                TokenPooler(2, "max")(embeddings, mask), torch.tensor([[3.0, 4.0]])
            )
        )

    def test_truncation_includes_header_in_max_rows(self):
        table = Table(["a", "b", "c"], [["1", "2", "3"]] * 5)
        truncated = truncate_table(table, max_rows=3, max_cols=2)
        self.assertEqual(truncated.shape, (3, 2))

    def test_all_configs_load(self):
        for path in Path("configs").glob("*.yaml"):
            config = load_config(path)
            self.assertGreater(config.data.max_rows, 0)

    def test_end_to_end_gradients(self):
        tokenizer = DummyTokenizer()
        config = ExperimentConfig()
        config.data.max_rows = 4
        config.data.max_cols = 3
        config.data.max_cell_tokens = 4
        config.cell_encoder.cell_dim = 128
        config.cnn.channels = 128
        config.cnn.depth = 2
        config.cross_attention.insertion_layer = 1
        config.cross_attention.num_heads = 4
        language_model = DummyLM()
        model = TableCNNQwen(language_model, tokenizer, config)
        example = {
            "question": "which value",
            "answers": ["yes"],
            "table": {"header": ["name", "value"], "rows": [["x", "yes"]]},
        }
        batch = MRCBatchCollator(tokenizer, "cnn", 4, 3, 32, 8, training=True)(
            [example]
        )
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            tables=batch["tables"],
        )
        output.loss.backward()
        for module in [
            model.cell_encoder,
            model.table_cnn,
            model.projector,
            model.cross_attention,
        ]:
            grads = [
                p.grad
                for p in module.parameters()
                if p.requires_grad and p.grad is not None
            ]
            self.assertTrue(grads)
            self.assertTrue(any(torch.count_nonzero(grad).item() for grad in grads))
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            tables=batch["tables"],
            max_new_tokens=2,
        )
        self.assertEqual(generated.shape[1], batch["input_ids"].shape[1] + 2)

    def test_early_stopping_and_automatic_completed_run_resume(self):
        tokenizer = DummyTokenizer()
        config = ExperimentConfig()
        config.data.max_rows = 4
        config.data.max_cols = 3
        config.data.max_cell_tokens = 4
        config.cell_encoder.cell_dim = 128
        config.cnn.channels = 128
        config.cross_attention.insertion_layer = 1
        config.cross_attention.num_heads = 4
        config.training.bf16 = False
        config.training.batch_size = 1
        config.training.gradient_accumulation_steps = 1
        config.training.epochs = 3
        config.training.checkpoint_every_steps = 1
        config.training.early_stopping_patience = 1
        config.training.log_every = 1
        example = {
            "question": "which value",
            "answers": ["yes"],
            "table": {"header": ["name", "value"], "rows": [["x", "yes"]]},
        }
        fake_metrics = {
            "exact_match": 0.5,
            "number_evaluated": 1,
            "number_correct": 0,
            "accuracy": 0.5,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            config.training.output_dir = temporary_directory
            with patch("src.train.evaluate_model", return_value=(fake_metrics, [])):
                model = TableCNNQwen(DummyLM(), tokenizer, config)
                history = train_model(
                    model,
                    tokenizer,
                    [example],
                    [example],
                    config,
                    torch.device("cpu"),
                )
                self.assertEqual(history["status"], "early_stopped")
                self.assertEqual(len(history["epochs"]), 2)
                self.assertTrue(
                    (Path(temporary_directory) / "checkpoint_best.pt").is_file()
                )
                resumed_model = TableCNNQwen(DummyLM(), tokenizer, config)
                resumed_history = train_model(
                    resumed_model,
                    tokenizer,
                    [example],
                    [example],
                    config,
                    torch.device("cpu"),
                )
                self.assertEqual(resumed_history["status"], "early_stopped")
                self.assertEqual(len(resumed_history["resume_events"]), 1)


if __name__ == "__main__":
    unittest.main()
