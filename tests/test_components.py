from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from src.checkpointing import (
    architecture_signature,
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.config import ExperimentConfig, load_config
from src.data import (
    MRCBatchCollator,
    Table,
    build_prompt_ids,
    build_serialized_prompt_with_cell_alignment,
    select_table_for_question,
    serialize_answers,
    truncate_table,
)
from src.diagnostics import build_table_shuffled_examples, table_dependence_metrics
from src.model import (
    ContinuousPrefixQwen,
    SerializedCNNResidualQwen,
    Structured2DQwen,
    TableCNNQwen,
    build_model,
)
from src.pooling import TokenPooler
from src.train import train_model
from src.utils import mirror_directory
from src.wtq_evaluation import (
    OfficialTarget,
    check_denotation,
    load_official_targets,
    normalize_wtq_string,
    score_prediction,
    split_prediction_items,
    to_wtq_values,
    truncation_coverage,
)


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
        max_length=4096,
        return_tensors="pt",
        return_offsets_mapping=False,
    ):
        del add_special_tokens, padding, truncation, return_tensors
        if isinstance(texts, str):
            matches = list(re.finditer(r"\S+", texts))
            result = {
                "input_ids": [
                    2 + (sum(match.group().encode("utf-8")) % 27)
                    for match in matches
                ][:max_length]
            }
            if return_offsets_mapping:
                result["offset_mapping"] = [
                    (match.start(), match.end()) for match in matches[:max_length]
                ]
            return result
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
        positions = torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype
        ).view(1, -1, 1)
        causal_context = hidden.cumsum(dim=1) / positions
        return hidden + torch.tanh(self.linear(causal_context))


class DummyLM(nn.Module):
    def __init__(self, hidden_size=16, vocab_size=32, layers=3):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, pad_token_id=0)
        self.generation_config = SimpleNamespace()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = SimpleNamespace()
        self.model.layers = nn.ModuleList(
            [DummyLayer(hidden_size) for _ in range(layers)]
        )
        self.layers = self.model.layers
        self.head = nn.Linear(hidden_size, vocab_size)
        self.last_attention_mask = None
        self.last_labels = None
        self.last_inputs_embeds = None

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        **kwargs,
    ):
        del kwargs
        self.last_attention_mask = attention_mask.detach().clone()
        self.last_labels = None if labels is None else labels.detach().clone()
        self.last_inputs_embeds = inputs_embeds
        hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        if hasattr(self, "lora_adapter"):
            hidden = hidden + self.lora_adapter(hidden)
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        max_new_tokens=1,
        **kwargs,
    ):
        del kwargs
        if inputs_embeds is not None:
            self.last_inputs_embeds = inputs_embeds
            self.last_attention_mask = attention_mask.detach().clone()
            return torch.full(
                (inputs_embeds.shape[0], max_new_tokens),
                2,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        sequence = input_ids
        mask = attention_mask
        for _ in range(max_new_tokens):
            logits = self.forward(sequence, mask).logits
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            sequence = torch.cat([sequence, next_token], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)
        return sequence


class ComponentTests(unittest.TestCase):
    def test_lora_build_keeps_only_adapters_and_table_path_trainable(self):
        tokenizer = DummyTokenizer()
        config = ExperimentConfig(experiment_type="continuous_prefix")
        config.lora.enabled = True
        config.data.max_rows = 4
        config.data.max_cols = 3
        config.cell_encoder.cell_dim = 128
        config.cnn.channels = 128
        observed: dict[str, object] = {}

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                observed["model_load"] = (args, kwargs)
                return DummyLM()

        class FakePeftConfig:
            def __init__(self, **kwargs):
                observed["lora_config"] = kwargs

        class FakeTaskType:
            CAUSAL_LM = "CAUSAL_LM"

        def fake_get_peft_model(language_model, lora_config):
            del lora_config
            language_model.requires_grad_(False)
            language_model.lora_adapter = nn.Linear(16, 16, bias=False)
            return language_model

        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = FakeAutoModel
        fake_peft = ModuleType("peft")
        fake_peft.LoraConfig = FakePeftConfig
        fake_peft.TaskType = FakeTaskType
        fake_peft.get_peft_model = fake_get_peft_model
        with patch.dict(
            sys.modules,
            {"transformers": fake_transformers, "peft": fake_peft},
        ):
            model = build_model(
                config,
                tokenizer,
                torch.device("cpu"),
                torch.float32,
            )

        self.assertIsInstance(model, ContinuousPrefixQwen)
        self.assertFalse(model.language_model.embed.weight.requires_grad)
        self.assertTrue(model.language_model.lora_adapter.weight.requires_grad)
        self.assertEqual(
            observed["lora_config"]["target_modules"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.assertEqual(observed["lora_config"]["r"], 16)
        example = {
            "question": "which value",
            "answers": ["yes"],
            "table": {"header": ["name", "value"], "rows": [["x", "yes"]]},
        }
        batch = MRCBatchCollator(
            tokenizer, "continuous_prefix", 4, 3, 32, 8, training=True
        )([example])
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            tables=batch["tables"],
        )
        output.loss.backward()
        adapter_gradient = model.language_model.lora_adapter.weight.grad
        self.assertIsNotNone(adapter_gradient)
        self.assertGreater(torch.count_nonzero(adapter_gradient).item(), 0)

    def test_disabled_lora_preserves_legacy_checkpoint_signature(self):
        baseline = load_config("configs/baseline.yaml")
        self.assertEqual(
            architecture_signature(baseline),
            "0c07a3ebeb08140b649107a3914e747090ebc6060be3b8eacefdcb6cbd1b475c",
        )
        lora_config = load_config("configs/continuous_prefix_lora.yaml")
        self.assertNotEqual(
            architecture_signature(lora_config), architecture_signature(baseline)
        )

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
        example = {
            "question": "question",
            "answers": ["answer"],
            "table": {"header": ["column"], "rows": [["value"]]},
        }
        MRCBatchCollator(tokenizer, "continuous_prefix", 4, 3, 32, 8, training=False)(
            [example]
        )
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

    def test_official_denotation_scoring_uses_complete_answer_sets(self):
        target = OfficialTarget(
            original_strings=("Café [1]", "2"),
            canonical_strings=("Café [1]", "2.0"),
            values=to_wtq_values(("Café [1]", "2"), ("Café [1]", "2.0")),
        )
        self.assertEqual(normalize_wtq_string('"Café [1]."'), "cafe [1]")
        self.assertTrue(score_prediction("2.0 | cafe", target))
        self.assertFalse(score_prediction("cafe", target))
        self.assertTrue(
            check_denotation(to_wtq_values(["1"]), to_wtq_values(["1.0"]))
        )
        self.assertEqual(split_prediction_items("New York, NY; Boston"), [
            "New York, NY",
            "Boston",
        ])

    def test_official_target_loader_falls_back_when_canonical_field_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            tagged_path = Path(temporary_directory) / "sample.tagged"
            tagged_path.write_text(
                "id\ttargetValue\ttargetCanon\n"
                "good\t2\t2.0\n"
                "short\tanswer\n",
                encoding="utf-8",
            )
            targets = load_official_targets(temporary_directory)
        self.assertEqual(targets["good"].canonical_strings, ("2.0",))
        self.assertEqual(targets["short"].canonical_strings, ("answer",))
        self.assertTrue(score_prediction("answer", targets["short"]))

    def test_truncation_audit_only_counts_answers_removed_by_truncation(self):
        targets = {
            "kept-out": OfficialTarget(
                ("answer",), ("answer",), to_wtq_values(["answer"])
            ),
            "computed": OfficialTarget(("3",), ("3",), to_wtq_values(["3"])),
        }
        examples = [
            {
                "id": "kept-out",
                "question": "what is last?",
                "table": {
                    "header": ["name", "value"],
                    "rows": [["first", "x"], ["last", "answer"]],
                },
            },
            {
                "id": "computed",
                "question": "how many?",
                "table": {"header": ["name"], "rows": [["a"], ["b"]]},
            },
        ]
        audit = truncation_coverage(examples, targets, max_rows=2, max_cols=2)
        self.assertEqual(audit["number_evaluated"], 2)
        self.assertEqual(audit["full_table_direct_coverage_count"], 1)
        self.assertEqual(audit["truncation_removed_direct_answer_count"], 1)
        self.assertEqual(audit["removed_samples"][0]["id"], "kept-out")

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

    def test_question_relevance_selection_keeps_matching_row_and_neighbor(self):
        rows = [[str(index), f"Player {index}", "Back", "School"] for index in range(45)]
        rows[40][1] = "Frank Burns"
        rows[41][1] = "Frank Ziegler"
        table = Table(["Pick", "Player", "Position", "School"], rows)
        selected = select_table_for_question(
            table,
            "who was picked after Frank Burns?",
            max_rows=4,
            max_cols=3,
            neighbor_radius=1,
        )
        flattened = {cell for row in selected.rows for cell in row}
        self.assertEqual(selected.shape, (4, 3))
        self.assertIn("Frank Burns", flattened)
        self.assertIn("Frank Ziegler", flattened)

    def test_all_answer_targets_use_explicit_separator(self):
        self.assertEqual(
            serialize_answers(["first", "second"], "all", " | "),
            "first | second",
        )
        tokenizer = DummyTokenizer()
        example = {
            "question": "list both",
            "answers": ["first", "second"],
            "table": {"header": ["value"], "rows": [["first"], ["second"]]},
        }
        batch = MRCBatchCollator(
            tokenizer,
            "serialized",
            4,
            3,
            64,
            16,
            training=True,
            answer_mode="all",
        )([example])
        expected_answer_ids = tokenizer.encode("first | second")
        supervised = batch["labels"][0][batch["labels"][0] != -100].tolist()
        self.assertEqual(supervised, expected_answer_ids + [tokenizer.eos_token_id])

    def test_all_configs_load(self):
        for path in Path("configs").glob("*.yaml"):
            config = load_config(path)
            self.assertGreater(config.data.max_rows, 0)

    def test_serialized_cnn_residual_aligns_cells_and_backpropagates(self):
        tokenizer = DummyTokenizer()
        config = ExperimentConfig(experiment_type="serialized_cnn_residual")
        config.data.max_rows = 4
        config.data.max_cols = 3
        config.data.max_cell_tokens = 4
        config.data.max_question_tokens = 128
        config.cell_encoder.cell_dim = 128
        config.cell_encoder.mlp_type = "deep"
        config.cell_encoder.deep_hidden_dim = 256
        config.cnn.channels = 128
        config.cnn.depth = 2
        config.cnn_residual.insertion_layer = 1
        config.cnn_residual.gate_init = 0.1
        language_model = DummyLM()
        model = SerializedCNNResidualQwen(language_model, tokenizer, config)
        example = {
            "question": "which value",
            "answers": ["yes"],
            "table": {"header": ["name", "value"], "rows": [["x", "yes"]]},
        }
        batch = MRCBatchCollator(
            tokenizer,
            "serialized_cnn_residual",
            4,
            3,
            128,
            8,
            training=True,
        )([example])
        prompt_ids, alignment = build_serialized_prompt_with_cell_alignment(
            tokenizer,
            example["question"],
            example["table"],
            4,
            3,
        )
        self.assertEqual(
            prompt_ids,
            build_prompt_ids(
                tokenizer,
                example["question"],
                example["table"],
                "serialized",
                4,
                3,
                False,
            ),
        )
        self.assertIn(0, alignment)
        self.assertIn(1, alignment)
        self.assertIn(3, alignment)
        self.assertIn(4, alignment)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            tables=batch["tables"],
            table_cell_indices=batch["table_cell_indices"],
        )
        output.loss.backward()
        for module in [model.cell_encoder, model.table_cnn, model.projector]:
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(
                any(torch.count_nonzero(gradient).item() for gradient in gradients)
            )
        self.assertIsNotNone(model.residual_gate.grad)
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            tables=batch["tables"],
            table_cell_indices=batch["table_cell_indices"],
            max_new_tokens=2,
        )
        self.assertEqual(generated.shape[1], batch["input_ids"].shape[1] + 2)

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

    def test_continuous_prefix_masks_loss_and_generates_with_one_decoder(self):
        tokenizer = DummyTokenizer()
        config = ExperimentConfig(experiment_type="continuous_prefix")
        config.data.max_rows = 4
        config.data.max_cols = 3
        config.data.max_cell_tokens = 4
        config.cell_encoder.cell_dim = 128
        config.cnn.channels = 128
        language_model = DummyLM()
        model = ContinuousPrefixQwen(language_model, tokenizer, config)
        self.assertFalse(hasattr(model, "cross_attention"))
        example = {
            "question": "which value",
            "answers": ["yes"],
            "table": {"header": ["name", "value"], "rows": [["x", "yes"]]},
        }
        batch = MRCBatchCollator(
            tokenizer, "continuous_prefix", 4, 3, 32, 8, training=True
        )([example])
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            tables=batch["tables"],
        )
        prefix_length = 4
        self.assertEqual(
            language_model.last_inputs_embeds.shape[1],
            prefix_length + batch["input_ids"].shape[1],
        )
        self.assertTrue(
            torch.equal(
                language_model.last_labels[:, :prefix_length],
                torch.full((1, prefix_length), -100),
            )
        )
        self.assertTrue(
            torch.equal(language_model.last_labels[:, prefix_length:], batch["labels"])
        )
        self.assertTrue(
            torch.equal(
                language_model.last_attention_mask[:, :prefix_length],
                torch.ones(1, prefix_length, dtype=torch.long),
            )
        )
        output.loss.backward()
        for module in [model.cell_encoder, model.table_cnn, model.projector]:
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(
                any(torch.count_nonzero(gradient).item() for gradient in gradients)
            )
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            tables=batch["tables"],
            max_new_tokens=2,
        )
        self.assertEqual(generated.shape[1], batch["input_ids"].shape[1] + 2)

    def test_structured_2d_uses_lexical_tokens_and_trainable_structure(self):
        tokenizer = DummyTokenizer()
        config = ExperimentConfig(experiment_type="structured_2d")
        config.data.max_rows = 4
        config.data.max_cols = 3
        config.data.max_table_tokens = 64
        language_model = DummyLM()
        model = Structured2DQwen(language_model, tokenizer, config)
        example = {
            "question": "which value",
            "answers": ["yes"],
            "table": {"header": ["name", "value"], "rows": [["x", "yes"]]},
        }
        batch = MRCBatchCollator(
            tokenizer, "structured_2d", 4, 3, 32, 8, training=True
        )([example])
        prefix, prefix_mask, shapes = model.encode_tables(batch["tables"])
        self.assertEqual(shapes["table_prefix"], tuple(prefix.shape))
        self.assertGreater(int(prefix_mask.sum()), 4)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            tables=batch["tables"],
        )
        prefix_length = prefix.shape[1]
        self.assertTrue(
            torch.equal(
                language_model.last_labels[:, :prefix_length],
                torch.full((1, prefix_length), -100),
            )
        )
        output.loss.backward()
        for module in [
            model.row_embeddings,
            model.column_embeddings,
            model.cell_type_embeddings,
        ]:
            self.assertIsNotNone(module.weight.grad)
            self.assertGreater(torch.count_nonzero(module.weight.grad).item(), 0)
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

    def test_denotation_accuracy_controls_best_checkpoint_and_early_stopping(self):
        tokenizer = DummyTokenizer()
        config = ExperimentConfig()
        config.data.max_rows = 4
        config.data.max_cols = 3
        config.data.max_cell_tokens = 4
        config.cell_encoder.cell_dim = 128
        config.cnn.channels = 128
        config.cross_attention.insertion_layer = 1
        config.cross_attention.num_heads = 4
        config.evaluation.primary_metric = "denotation_accuracy"
        config.training.bf16 = False
        config.training.batch_size = 1
        config.training.gradient_accumulation_steps = 1
        config.training.epochs = 3
        config.training.checkpoint_every_steps = 0
        config.training.early_stopping_patience = 1
        example = {
            "id": "example",
            "question": "which value",
            "answers": ["yes"],
            "table": {"header": ["name", "value"], "rows": [["x", "yes"]]},
        }
        fake_metrics = {
            "exact_match": 0.9,
            "denotation_accuracy": 0.25,
            "number_evaluated": 1,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            config.training.output_dir = temporary_directory
            with (
                patch("src.train._prepare_official_targets", return_value={}),
                patch("src.train.evaluate_model", return_value=(fake_metrics, [])),
            ):
                history = train_model(
                    TableCNNQwen(DummyLM(), tokenizer, config),
                    tokenizer,
                    [example],
                    [example],
                    config,
                    torch.device("cpu"),
                )
        self.assertEqual(history["status"], "early_stopped")
        self.assertEqual(history["primary_metric"], "denotation_accuracy")
        self.assertEqual(history["best_metric"], 0.25)
        self.assertEqual(history["best_denotation_accuracy"], 0.25)


if __name__ == "__main__":
    unittest.main()
