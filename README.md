# Table-CNN MRC

A config-driven research implementation for testing whether a 2D CNN over semantic
table-cell embeddings can improve table question answering on WikiTableQuestions.

The proposed model keeps the two modalities separate:

```text
question ───────────────────────────────→ Qwen3-1.7B decoder
                                                ↑
table → Qwen embeddings → cell pooling → MLP → 2D CNN → projector
                                                │
                                      gated cross-attention
```

The decoder prompt contains the question only. The table becomes cross-attention
memory after decoder layer 13 by default. A separate serialized-table baseline is
included in `configs/serialized_table.yaml`.

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

## Data behavior

The dataset is loaded exactly as:

```python
load_dataset(
    "stanfordnlp/wikitablequestions",
    revision="refs/pr/4",
)
```

`max_rows` is the total grid height including the header. A `32 × 8` config therefore
keeps the header, the first 31 data rows, and the first 8 columns. This convention
ensures the memory length is at most `max_rows * max_cols`. Truncation is isolated in
`src/data.py` so it can later be replaced by learned or heuristic selection.

The first accepted answer is used for training. During evaluation, a prediction is
correct if it matches any accepted answer after stripping, lowercasing, and collapsing
repeated whitespace.

## Configuration

`configs/baseline.yaml` is the canonical CNN experiment. Included one-variable
variants cover:

- pooling: mean, max, attention
- cell width: 128, 256, 512
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

## Outputs and reproducibility

Each training run saves:

```text
outputs/<run>/
├── config.yaml
├── history.json
├── checkpoint_last.pt
└── validation_epoch_<N>/
    ├── metrics.json
    └── predictions.json
```

Checkpoints contain trainable parameters only. For the frozen-backbone CNN model,
this keeps checkpoints compact; evaluation reloads the named Qwen base model before
applying them. Serialized baseline checkpoints contain the fine-tuned backbone and
are correspondingly large.

To retain a second copy on persistent storage, pass a mirror directory:

```bash
python scripts/run_experiment.py \
  --config configs/baseline.yaml \
  --mirror-output-dir /content/drive/MyDrive/cnn_qwen_table_mcr/outputs/baseline
```

The local run remains under `outputs/baseline`. After every epoch, all run artifacts
are also copied to the mirror. The resolved `config.yaml` records the mirror path.

Seeds are set for Python, NumPy, PyTorch, and CUDA. Evaluation uses greedy generation.

## Colab

Open `notebooks/colab_runner.ipynb`, set `REPO_URL`, and run the cells. The notebook
clones the repository, installs dependencies, reads `HF_TOKEN` from Colab userdata,
and mounts Google Drive. Training outputs are kept both locally and under:

```text
/content/drive/MyDrive/cnn_qwen_table_mcr/outputs/baseline
```

Model logic stays in the package rather than notebook cells.

## Local tests

The component suite requires no model or dataset download:

```bash
python -m unittest discover -s tests -v
```

It checks configuration loading, masking and truncation, the decoder injection, loss,
and gradient flow through every new model component.

## Repository layout

```text
configs/                 experiment YAML files
src/config.py            strict dataclass configuration
src/data.py              WTQ loading, prompts, batching, truncation
src/pooling.py           mean/max/attention token pooling
src/cell_encoder.py      batched cell encoding and MLP
src/table_cnn.py         masked 2D CNN and projector
src/cross_attention.py   gated multi-head cross-attention
src/model.py             Qwen wrappers and checkpoint I/O
src/train.py             training loop and run artifacts
src/evaluate.py          deterministic Exact Match evaluation
scripts/                 command-line entry points
tests/                   download-free integration tests
notebooks/               thin Colab runner
```
