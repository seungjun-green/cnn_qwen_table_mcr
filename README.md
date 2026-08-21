# Table-CNN MRC

A config-driven research implementation for table question answering on
WikiTableQuestions. It retains the original 2D-CNN experiments and now includes a
structure-aware lexical-token model developed from their diagnostic results.

The repository supports two historical Table-CNN fusion strategies. The original model
keeps the two modalities separate:

```text
question ───────────────────────────────→ Qwen3-1.7B decoder
                                                ↑
table → Qwen embeddings → cell pooling → MLP → 2D CNN → projector
                                                │
                                      gated cross-attention
```

The decoder prompt contains the question only. The table becomes cross-attention
memory after decoder layer 13 by default.

The continuous-prefix experiment instead uses one ordinary causal
decoder and no cross-attention module:

```text
table → Qwen embeddings → cell pooling → MLP → 2D CNN → projector ─┐
                                                                  ├→ Qwen → answer
question → Qwen token embeddings ─────────────────────────────────┘
```

Its causal input is `[table-prefix embeddings][question tokens][answer tokens]`.
Loss is masked for every table and question position, so only answer tokens are
training targets. A separate serialized-table baseline remains available in
`configs/serialized_table.yaml`.

Its LoRA config adds adapters to Qwen's self-attention while
keeping its pretrained weights frozen. The table encoder, projector, and adapters
are optimized jointly:

```text
[table-prefix embeddings][question tokens][answer tokens]
                         ↓
       Qwen + LoRA(q_proj, k_proj, v_proj, o_proj)
                         ↓
                       answer
```

The current proposed model preserves the table's lexical tokens and adds lightweight
2D structure instead of compressing cells through a CNN:

```text
question ──→ deterministic relevant row/column selection ─┐
table ─────→ lexical table tokens                          │
              + learned row embeddings                    ├→ Qwen + LoRA → answer set
              + learned column embeddings                 │
              + learned header/data embeddings ───────────┘
```

`configs/serialized_table_lora.yaml` is the corrected standard baseline,
`configs/serialized_retrieval_lora.yaml` isolates the selector contribution, and
`configs/structured_2d_lora.yaml` is the proposed model. All three train on every
gold denotation item and choose checkpoints using official WTQ denotation accuracy.

The cell-aligned CNN residual experiment keeps the exact serialized baseline prompt
and adds a parallel structural path:

```text
question + serialized table → Qwen decoder layers ───────────────→ answer
                              middle hidden state + CNN residual ↑
table → cell pooling → row/column/type embeddings → 2D CNN ──────┘
```

Each CNN cell vector is projected to Qwen's hidden size and added only to the
serialized tokens belonging to that cell. A zero-initialized learned gate controls
the residual, and the frozen Qwen backbone, LoRA adapters, and CNN path retain the
same training objective as the serialized baseline.

## Quick start

Python 3.10+ and an NVIDIA GPU are recommended. The baseline targets an H100 and
uses bf16. Qwen3-1.7B is downloaded from Hugging Face.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=your_token_if_needed
python scripts/smoke_test.py --config configs/baseline.yaml
```

The smoke test loads one real WTQ example and verifies tokenization, table encoding,
loss, backward propagation, finite non-zero gradients in every new component, and
greedy generation. Its answer is not expected to be correct before training.

Train the baseline:

```bash
python scripts/run_experiment.py --config configs/baseline.yaml
```

Train the continuous-prefix model:

```bash
python scripts/smoke_test.py --config configs/continuous_prefix.yaml
python scripts/run_experiment.py --config configs/continuous_prefix.yaml
```

Train the recommended LoRA version:

```bash
python scripts/smoke_test.py --config configs/continuous_prefix_lora.yaml
python scripts/run_experiment.py --config configs/continuous_prefix_lora.yaml
```

Run the new controlled structure-aware experiments:

```bash
python scripts/smoke_test.py --config configs/serialized_table_lora.yaml
python scripts/run_experiment.py --config configs/serialized_table_lora.yaml

python scripts/smoke_test.py --config configs/structured_2d_lora.yaml
python scripts/run_experiment.py --config configs/structured_2d_lora.yaml
```

Run the first serialized-versus-CNN-residual comparison:

```bash
python scripts/smoke_test.py --config configs/cnn_residual_mean_middle.yaml
python scripts/run_experiment.py --config configs/cnn_residual_mean_middle.yaml
```

Evaluate a saved checkpoint on validation:

```bash
python scripts/evaluate.py \
  --config outputs/baseline/config.yaml \
  --checkpoint outputs/baseline/checkpoint_last.pt \
  --split validation
```

Use `--split test` only for final reporting. Add `--max-examples N` for a quick
evaluation check.

## Architecture

For each batch, all real cells across all tables are collected and tokenized in one
operation. Qwen's own input embedding layer produces token vectors. Mean, max, or a
small learned attention scorer reduces the tokens in each cell to one vector. The
cell MLP maps Qwen's hidden size to 128, 256, or 512 dimensions.

Cell vectors are restored to `[B, R, C, D]`, changed to `[B, D, R, C]`, and passed
through a masked residual 2D CNN. The result is flattened and projected back to
Qwen's hidden size. Padded cells are masked in both the CNN and cross-attention.

Cross-attention is installed as a forward hook on the selected Qwen decoder layer.
This retains Hugging Face's native causal-LM loss, KV caching, and `generate()`
implementation. The residual is:

```text
H' = H + tanh(alpha) * CrossAttention(H, table_memory)
```

The Qwen backbone is frozen for the CNN baseline. The shared token embedding matrix
is consequently frozen too, while the cell MLP, CNN, projector, cross-attention, and
gate are trained.

### Continuous-prefix fusion

`configs/continuous_prefix.yaml` uses the same batched cell encoder and masked 2D
CNN, with the 128-dimensional cell representation that showed the strongest table
dependence in the initial sweep. Real cells are packed in row-major order and
projected to Qwen's hidden size. These continuous vectors are prepended to the
question's normal token embeddings and sent through Qwen with a standard causal
attention mask.

There is no decoder hook and no cross-attention block. The frozen Qwen model is the
only decoder; the cell MLP, CNN, and table projector learn a soft table prefix that
conditions it. Qwen thinking is disabled consistently for this experiment during
both training and evaluation, which keeps the short-answer prompt format aligned.

`configs/continuous_prefix_lora.yaml` additionally trains PEFT LoRA adapters on all
Qwen attention projections with rank 16, alpha 32, and dropout 0.05. The original
Qwen parameters remain frozen. LoRA is configured through the top-level `lora`
section; set `enabled: false` for the frozen-decoder ablation. The integration uses
Hugging Face's [PEFT LoRA API](https://huggingface.co/docs/peft/package_reference/lora).

### Structure-aware lexical-token model

`structured_2d` first applies deterministic question-conditioned selection. Columns
are ranked using question overlap with headers and cell values. Rows are ranked by
question/value overlap, and neighboring rows are retained so that `before`, `after`,
and adjacent-row questions keep their local context. Selected rows and columns remain
in original table order.

The selected cells are tokenized without pooling. Qwen receives its own pretrained
token embeddings plus learned row, column, and header/data-type embeddings. A learned
scale starts the structural contribution small, while Qwen LoRA and the structure
embeddings train jointly. The table prefix is capped by `max_table_tokens`; question
and answer labels follow it through the same causal decoder, and all table/question
positions remain masked from the language-model loss.

## Data behavior

The dataset is loaded exactly as:

```python
load_dataset(
    "stanfordnlp/wikitablequestions",
    revision="refs/pr/4",
)
```

`max_rows` is the total grid height including the header. In `leading` mode, a
`32 × 8` config keeps the header, first 31 data rows, and first 8 columns. In
`question_relevance` mode, it keeps up to 31 ranked rows plus their neighbors and up
to 8 ranked columns, restoring original order after selection.

Legacy configs continue to train on the first accepted answer and legacy Exact Match.
The new configs use `answer_mode: all`, join answer items with ` | `, and download the
official WTQ 1.0.2 tagged targets once. Validation, early stopping, and best-checkpoint
selection then use complete-denotation accuracy: predicted and target sets must have
equal sizes and every target item must match.

## Configuration

`configs/structured_2d_lora.yaml` is the proposed experiment,
`configs/serialized_table_lora.yaml` is its standard baseline, and
`configs/serialized_retrieval_lora.yaml` attributes gains from selection separately
from 2D embeddings. `configs/continuous_prefix_lora.yaml` remains the diagnosed CNN
ablation and `configs/baseline.yaml` is the original cross-attention CNN. Included
one-variable variants cover:

- pooling: mean, max, attention
- cell width: 128, 256, 512
- CNN depth: 2 and 4 (depth 6 is also supported)
- table grids: 16×8, 32×8, 64×8, 64×16
- serialized-table input

Variant files rely on the defaults in `src/config.py`; every run writes its fully
resolved config into its output directory. Other supported CNN choices are depths
2, 4, or 6 and kernel sizes 3 or 5.

For quick development runs, add these fields under `training`:

```yaml
max_train_examples: 100
max_validation_examples: 50
epochs: 1
```

Disconnect-safe training controls are also configurable:

```yaml
training:
  checkpoint_every_steps: 100
  early_stopping_patience: 3
  early_stopping_min_delta: 0.0
  auto_resume: true
```

Loss is shown in the live training progress bar and every optimizer-step loss is
stored in `history.json`. Set `early_stopping_patience: null` to disable early
stopping, or `checkpoint_every_steps: 0` to disable mid-epoch checkpoints.

## Outputs and reproducibility

Each training run saves:

```text
outputs/<run>/
├── config.yaml
├── history.json
├── checkpoint_best.pt
├── checkpoint_last.pt
└── validation_epoch_<N>/
    ├── metrics.json
    └── predictions.json
```

Both checkpoints are fully resumable. They contain the trainable model parameters,
optimizer state, epoch and next-batch position, global step, loss history, best
validation metric, early-stopping counters, resolved config, and Python/NumPy/Torch/
CUDA random states. The frozen Qwen backbone is reloaded by name rather than copied
into the CNN checkpoint. Serialized baseline checkpoints contain the fine-tuned
backbone and are correspondingly large.

To retain a second copy on persistent storage, pass a mirror directory:

```bash
python scripts/run_experiment.py \
  --config configs/baseline.yaml \
  --mirror-output-dir /content/drive/MyDrive/cnn_qwen_table_mcr/outputs/baseline
```

The local run remains under `outputs/baseline`. At every configured checkpoint and
after every epoch, all run artifacts are copied to the mirror. On a fresh Colab
runtime, the mirror is automatically restored before model training starts. Resume
uses the last completed optimizer step; it never restores a partial accumulation.
The resolved `config.yaml` records the mirror path.

To use Google Drive as the primary checkpoint location rather than as a mirror, use
the output override:

```bash
python scripts/run_experiment.py \
  --config configs/cnn_residual_mean_middle.yaml \
  --output-dir /content/drive/MyDrive/cnn_qwen_table_mcr/outputs/cnn_residual_mean_middle
```

In this mode, checkpoints and validation artifacts are written directly to Drive and
automatic resume reads `checkpoint_last.pt` from the same directory.

For the LoRA model in Colab, use its distinct Drive directory so it cannot resume an
incompatible frozen-prefix or cross-attention checkpoint:

```bash
python scripts/run_experiment.py \
  --config configs/continuous_prefix_lora.yaml \
  --mirror-output-dir /content/drive/MyDrive/cnn_qwen_table_mcr/outputs/continuous_prefix_lora
```

Automatic resume is the default. It can also be controlled explicitly:

```bash
# Restore a particular checkpoint
python scripts/train.py --config configs/baseline.yaml \
  --resume-from /path/to/checkpoint_last.pt

# Deliberately start from scratch
python scripts/train.py --config configs/baseline.yaml --no-resume
```

Checkpoint architecture is checked against the active config before loading, which
prevents accidentally loading one experiment's weights into another architecture.

Seeds are set for Python, NumPy, PyTorch, and CUDA. Evaluation uses greedy generation.

## Colab

For the CNN residual study, open `notebooks/cnn_residual_colab.ipynb`. Enter one
configuration name per line, or enter `all`. The notebook defaults to all six
configurations, streams training output through Colab's native progress renderer,
and writes every run directly under Google Drive rather than `/content`.

Open `notebooks/colab_runner.ipynb` and run the cells. The repository URL is already
configured. The setup cell safely pulls an existing clone or creates it, installs
dependencies, reads `HF_TOKEN` from Colab userdata, and mounts Google Drive. Training
outputs are kept both locally and under:

```text
/content/drive/MyDrive/cnn_qwen_table_mcr/outputs/baseline
```

Model logic stays in the package rather than notebook cells.

Both Colab notebooks now run the recommended continuous-prefix LoRA config. For the
focused workflow, open `notebooks/continuous_prefix_colab.ipynb`; it includes setup,
the LoRA smoke test, disconnect-safe training, history inspection, and a
correct-table versus shuffled-table diagnostic. Commands use unbuffered Python so
training progress appears live in Colab.

The notebook also includes a nine-config ordered sweep. Its persistent state is:

```text
/content/drive/MyDrive/cnn_qwen_table_mcr/outputs/sweep_state.json
```

Rerunning the sweep cell skips configs marked `completed`. If the runtime stopped
during a config, the sweep invokes that config again and training restores its latest
full checkpoint from the corresponding Drive directory. If training finished but
the runtime stopped before the sweep state was updated, the completed checkpoint is
detected and the training command exits without repeating epochs.

The same mechanism can run any ordered config list:

```bash
python scripts/run_sweep.py \
  --configs configs/baseline.yaml configs/pooling_max.yaml configs/cell_dim_128.yaml \
  --mirror-root /content/drive/MyDrive/cnn_qwen_table_mcr/outputs
```

## Diagnose whether saved models use their tables

The first six sweep checkpoints can be tested directly from Drive without retraining:

```bash
python scripts/diagnose_saved_runs.py \
  --mirror-root /content/drive/MyDrive/cnn_qwen_table_mcr/outputs \
  --max-examples 200
```

For each checkpoint, this evaluates the same validation questions four ways:

- original Qwen3 prompt with correct tables;
- original Qwen3 prompt with shuffled tables;
- `enable_thinking=False` with correct tables;
- `enable_thinking=False` with shuffled tables.

It reports correct-table Exact Match, shuffled-table Exact Match, their difference,
and the fraction of predictions that change after table replacement. It also prints
sample predictions and saves all records under:

```text
/content/drive/MyDrive/cnn_qwen_table_mcr/outputs/diagnostics/table_dependence/
```

A meaningful Exact Match drop or prediction-change rate indicates that the model is
using table information. If both are nearly zero, it is likely relying mostly on
question/answer priors. The `no_thinking` comparison shows whether Qwen3's default
thinking prompt is interfering with short-answer evaluation.

Because these checkpoints were trained with the original prompt, `no_thinking` is a
diagnostic prompt-mismatch test rather than a fair retrained result. A clean
non-thinking experiment still requires training a new checkpoint with
`enable_thinking=False` used consistently for training and evaluation.

## Audit the best continuous-prefix LoRA checkpoint

Open `notebooks/checkpoint_audit_colab.ipynb` to audit the saved LoRA model without
retraining. It loads `checkpoint_best.pt` from Drive (the epoch-3 checkpoint for the
reported run), evaluates correct and shuffled tables, prints representative
predictions, computes complete-denotation accuracy with the official WTQ 1.0.2
normalization rules, and measures whether the configured 32×8 crop removes answers
that were directly present in the full table.

The same audit can be launched directly in Colab:

```bash
python scripts/audit_saved_checkpoint.py \
  --config configs/continuous_prefix_lora.yaml \
  --run-dir /content/drive/MyDrive/cnn_qwen_table_mcr/outputs/continuous_prefix_lora \
  --checkpoint best \
  --max-examples 200
```

Generation defaults to 200 validation examples while direct truncation coverage is
computed over the full validation split. Use `--max-examples 2831` for full-split
generation. The official tagged answer metadata is downloaded once and cached under
the Drive diagnostics directory. Audit artifacts are saved under
`continuous_prefix_lora/diagnostics/checkpoint_audit/`.

The checkpoint was trained only against `answers[0]`, so this audit diagnoses the
existing model; it does not retroactively fix its target objective. Retraining with
all denotation items should be done only after reviewing these results.

## Structure-aware Colab workflow

Open `notebooks/structured_2d_colab.ipynb`. It sets up the repository and Drive,
lets you choose one of the three controlled configs, runs the appropriate gradient
smoke test, trains or resumes from a distinct persistent checkpoint, compares
completed histories, and audits the best checkpoint with correct and shuffled tables.
Run `serialized_table_lora` first and `structured_2d_lora` second; the retrieval-only
configuration is the follow-up ablation.

## Local tests

The component suite requires no model or dataset download:

```bash
python -m unittest discover -s tests -v
```

It checks configuration loading, masking and selection, complete-answer targets,
decoder injection, continuous-prefix and structured-2D construction, answer-only loss
masking, structural/LoRA gradient flow, denotation scoring, atomic full-checkpoint
restoration, output mirroring, early stopping, and completed-run auto-resume.

## Repository layout

```text
configs/                 experiment YAML files
src/config.py            strict dataclass configuration
src/data.py              WTQ loading, prompts, selection, answer targets, batching
src/pooling.py           mean/max/attention token pooling
src/cell_encoder.py      batched cell encoding and MLP
src/table_cnn.py         masked 2D CNN and projector
src/cross_attention.py   gated multi-head cross-attention
src/checkpointing.py     atomic full-state save and resume
src/model.py             CNN, continuous-prefix, serialized, and structured-2D models
src/train.py             training loop and run artifacts
src/evaluate.py          deterministic Exact Match and denotation evaluation
src/wtq_evaluation.py    official WTQ value matching and coverage audits
scripts/                 command-line entry points
tests/                   download-free integration tests
notebooks/               thin Colab runner
```
