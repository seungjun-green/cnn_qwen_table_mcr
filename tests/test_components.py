from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from src.config import ExperimentConfig, load_config
from src.data import MRCBatchCollator, Table, truncate_table
from src.model import TableCNNQwen
from src.pooling import TokenPooler
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


if __name__ == "__main__":
    unittest.main()
