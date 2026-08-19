# Handoff: Table-CNN MRC Research Project

## Goal

Build a clean, reproducible research codebase for a Table MRC experiment.

Main idea:

```text
Table
→ Qwen token embeddings per cell
→ pool tokens inside each cell
→ MLP
→ restore 2D table grid
→ 2D CNN
→ projector
→ cross-attention into Qwen3-1.7B
→ answer generation
```

The decoder receives the **question only**. The table is handled separately by the CNN encoder.

The project should be easy to run from Google Colab by cloning the repository and launching config-driven experiments.

---

## Base Model

Use:

```text
Qwen/Qwen3-1.7B
```

Reuse Qwen's own token embedding matrix for table-cell text.

For the first experiments, freeze the pretrained Qwen backbone unless a config explicitly enables fine-tuning.

---

## Dataset

Use WikiTableQuestions:

```text
stanfordnlp/wikitablequestions
```

Load the Parquet-converted revision:

```python
dataset = load_dataset(
    "stanfordnlp/wikitablequestions",
    revision="refs/pr/4",
)
```

Splits:

```text
train
validation
test
```

Each example has:

```text
one table
one question
one or more accepted answers
```

Multiple examples may reuse the same table with different questions.

Observed statistics:

```text
TRAIN
examples: 11,321
unique tables: 1,332
questions/table: min 1, max 16, median 9, avg 8.50
rows/table: min 4, max 562, median 14, avg 25.19
cols/table: min 3, max 25, median 6, avg 6.36

VALIDATION
examples: 2,831
unique tables: 346
questions/table: min 1, max 15, median 9, avg 8.18
rows/table: min 4, max 753, median 14, avg 26.12
cols/table: min 3, max 21, median 6, avg 6.38

TEST
examples: 4,344
unique tables: 421
questions/table: min 1, max 17, median 11, avg 10.32
rows/table: min 5, max 517, median 15, avg 25.40
cols/table: min 3, max 21, median 6, avg 6.29
```

---

## Architecture

```text
                           QUESTION
                              |
                              v
                    Qwen3-1.7B decoder
                              |
                     decoder hidden state
                              |
                              Q
                              |
                       Cross-Attention
                      /               \
                     /                 \
                    Q                  K,V
                                       ^
                                       |
TABLE → cell encoding → 2D CNN → projector
                                       |
                                       v
                              table memory
```

The decoder then continues through the remaining Qwen layers and generates the answer autoregressively.

---

## Cell Encoding

Each table cell may contain one or many tokens.

Example:

```text
"Samsung Electronics"
```

Pipeline:

```text
cell string
→ Qwen tokenizer
→ Qwen token embedding layer
→ token embeddings
→ pooling
→ MLP
→ one fixed-size cell vector
```

Use the SAME Qwen embedding matrix used by the decoder.

Include the table header as part of the grid.

---

## Pooling Methods

Support:

```text
mean
max
attention
```

### Mean
Arithmetic mean over token embeddings.

### Max
Element-wise max over token embeddings.

### Attention
A lightweight learned attention scorer over tokens inside each cell.

Do not add another full Transformer per cell.

---

## Cell MLP

Qwen3-1.7B hidden size is 2048.

Support:

```text
2048 → 128
2048 → 256
2048 → 512
```

Also support an optional deeper version:

```text
2048 → 512 → 256
```

Default:

```text
mean pooling
2048 → 256
GELU
```

---

## 2D Grid Representation

After cell encoding, restore the original table layout.

Tensor:

```text
[B, R, C, D]
```

Before Conv2D:

```text
[B, D, R, C]
```

The feature dimension D is the CNN channel dimension.

Do NOT use a 3D CNN.

---

## Table Size Configurations

Support configurable maximum sizes:

```text
16 × 8
32 × 8
64 × 8
64 × 16
```

Default:

```text
32 rows × 8 columns
```

For oversized tables, initially use deterministic truncation:

```text
keep header
keep first max_rows data rows
keep first max_cols columns
```

Keep this logic isolated so smarter row/column selection can be added later.

For batching, pad smaller tables and maintain masks.

---

## CNN Encoder

Default CNN:

```text
cell_dim = 256

Conv2D 256 → 256
kernel 3×3
padding 1
GELU

Conv2D 256 → 256
kernel 3×3
padding 1
GELU

residual connection
```

Support experiments over:

```text
depth: 2, 4, 6
kernel: 3, 5
cell_dim: 128, 256, 512
```

---

## CNN Output

After CNN:

```text
[B, D, R, C]
```

Convert to:

```text
[B, R, C, D]
```

Then flatten:

```text
[B, R*C, D]
```

Project to Qwen hidden size:

```text
D → 2048
```

Final table memory:

```text
[B, num_cells, 2048]
```

This becomes K/V for decoder cross-attention.

---

## Cross-Attention Injection

Decoder input is the question only.

Initial default:

```text
inject after Qwen decoder layer 13
```

Qwen3-1.7B currently has 28 decoder layers.

Cross-attention:

```text
Q = decoder hidden states
K = table memory
V = table memory
```

Use a learnable gated residual:

```text
H' = H + tanh(alpha) * CrossAttention(H, table_memory)
```

Make `alpha` initialization configurable.

Baseline can use:

```text
gate_init = 0.1
```

Support later experiments with different insertion layers and multiple injection points, but baseline uses one.

---

## Already-Validated Smoke Test

A one-example prototype already succeeded.

Example:

```text
Question:
what is the total number of international goals that holosko has scored?

Gold:
13
```

Observed shapes:

```text
After cell MLP:
(1, 8, 7, 256)

After 2D CNN:
(1, 256, 8, 7)

Flattened CNN memory:
(1, 56, 256)

Projected table memory:
(1, 56, 2048)
```

Cross-attention inserted after decoder layer 13.

Trainable parameters in prototype:

```text
table encoder: 2,296,832
cross-attention: 16,785,409
```

Backward pass succeeded:

```text
Loss: 2.1959381103515625
Table encoder grad norm: 0.07113698869943619
Cross-attention grad norm: 0.2661801278591156
```

Generation also worked mechanically. It predicted `12` instead of `13`, which is expected because the new modules were untrained.

Conclusion:

```text
real WTQ table
→ Qwen embeddings
→ pooling
→ MLP
→ CNN
→ projector
→ cross-attention
→ Qwen decoder
→ LM loss
→ backward
→ generation
```

already works end-to-end.

---

## Training Objective

Use ordinary causal language-model loss over answer tokens.

Prompt concept:

```text
System:
Answer the user's question using the table representation provided separately.
Return only the final answer.

User:
<question>
```

The table is NOT serialized into the prompt for the CNN model.

Mask prompt tokens:

```text
labels[prompt_tokens] = -100
```

Train against answer tokens only.

For examples with multiple accepted answers, use the first answer for training initially. Evaluation should accept all listed gold answers.

---

## Baselines

### Baseline A: Serialized Table

```text
question + serialized table
→ Qwen3-1.7B
→ answer
```

No CNN.

This is essential.

### Baseline B: Proposed CNN Table Encoder

```text
question → Qwen
table → Qwen embeddings → pooling → MLP → CNN
CNN output → cross-attention into Qwen
→ answer
```

Optional later ablations:

```text
cell embeddings with no CNN
MLP-only table encoder
shuffled row/column grid
```

---

## Main Research Questions

1. Does a simple 2D CNN over semantic cell embeddings outperform serialized-table input?
2. Which pooling works best: mean, max, or attention?
3. Which cell dimension works best: 128, 256, or 512?
4. How much CNN capacity is needed?
5. Where should cross-attention be inserted in the decoder?

Avoid blindly running the full Cartesian product. Change one major variable at a time around a strong baseline.

---

## Baseline Config

Create:

```text
configs/baseline.yaml
```

Example:

```yaml
model:
  name: Qwen/Qwen3-1.7B
  freeze_backbone: true

data:
  dataset: stanfordnlp/wikitablequestions
  revision: refs/pr/4
  max_rows: 32
  max_cols: 8

cell_encoder:
  pooling: mean
  cell_dim: 256
  mlp_type: single

cnn:
  channels: 256
  depth: 2
  kernel_size: 3
  residual: true

cross_attention:
  insertion_layer: 13
  num_heads: 8
  gate_init: 0.1

training:
  bf16: true
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  epochs: 3
  seed: 42

generation:
  max_new_tokens: 32
```

Exact optimization hyperparameters may be adjusted if needed.

---

## Experiment Configs

Create examples such as:

```text
configs/
├── baseline.yaml
├── pooling_mean.yaml
├── pooling_max.yaml
├── pooling_attention.yaml
├── cell_dim_128.yaml
├── cell_dim_256.yaml
├── cell_dim_512.yaml
├── grid_16x8.yaml
├── grid_32x8.yaml
├── grid_64x8.yaml
└── grid_64x16.yaml
```

Keep config tooling simple.

---

## Evaluation

At minimum report:

```text
Exact Match
number evaluated
number correct
accuracy
```

Normalize prediction/gold with:

```text
strip whitespace
lowercase
normalize repeated spaces
```

Save per-example predictions:

```text
question
prediction
gold_answers
correct
table_rows
table_cols
```

Use validation for model/config selection. Test only for final reporting.

---

## Reproducibility

Save for every run:

```text
config
seed
model name
dataset revision
training loss history
validation metrics
checkpoint
predictions
test metrics if explicitly run
```

Set seeds for Python, NumPy, PyTorch, and CUDA.

---

## Repository Structure

Use approximately:

```text
table-cnn-mrc/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── baseline.yaml
│   ├── pooling_mean.yaml
│   ├── pooling_max.yaml
│   ├── pooling_attention.yaml
│   ├── cell_dim_128.yaml
│   ├── cell_dim_256.yaml
│   ├── cell_dim_512.yaml
│   ├── grid_16x8.yaml
│   ├── grid_32x8.yaml
│   ├── grid_64x8.yaml
│   └── grid_64x16.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── pooling.py
│   ├── cell_encoder.py
│   ├── table_cnn.py
│   ├── cross_attention.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── scripts/
│   ├── smoke_test.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_experiment.py
│
├── notebooks/
│   └── colab_runner.ipynb
│
└── outputs/
    └── .gitkeep
```

---

## Colab Runner

Keep the notebook thin.

It should mainly do:

```python
!git clone <repo>
%cd table-cnn-mrc
!pip install -r requirements.txt
```

Then:

```python
from google.colab import userdata
import os

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Then:

```python
!python scripts/smoke_test.py
```

or:

```python
!python scripts/run_experiment.py --config configs/baseline.yaml
```

Research/model code belongs in the repo, not notebook cells.

---

## Smoke Test Script

Create:

```text
scripts/smoke_test.py
```

It must:

1. Load one real WTQ example.
2. Print question, answer, and table shape.
3. Tokenize cells with Qwen tokenizer.
4. Use Qwen's embedding matrix.
5. Pool tokens inside each cell.
6. Run cell MLP.
7. Restore 2D table grid.
8. Run CNN.
9. Flatten/project to Qwen hidden size.
10. Inject through cross-attention.
11. Compute one LM loss.
12. Run `loss.backward()`.
13. Assert finite, non-zero gradients reach:
   - cell encoder
   - CNN
   - projector
   - cross-attention
14. Run one autoregressive generation.
15. Print all major tensor shapes.

Generation accuracy is irrelevant for the smoke test.

---

## Efficiency

Target hardware:

```text
NVIDIA H100
Google Colab / remote H100
```

Use bf16.

The first prototype encoded cells one-by-one in Python loops. That is okay only for smoke testing.

For actual training, batch cell processing:

```text
collect cells across batch
→ tokenize together
→ Qwen embedding lookup
→ masked pooling
→ reshape into table grids
```

Avoid hundreds of tiny GPU calls per table.

---

## Padding and Masks

Tables have different sizes.

Implement:

```text
row padding
column padding
cell mask
```

Padded cells should not become meaningful table memory.

When flattening CNN output for cross-attention, use a key-padding mask if practical.

---

## Important Architectural Rule

For the proposed CNN model:

```text
Decoder input:
question + generated answer tokens

Encoder side:
table only
```

Do NOT also serialize the table into the decoder prompt.

Serialized-table input is only for the baseline.

---

## Do Not Add Yet

Do NOT add:

```text
3D CNN
full Transformer inside each cell
GNN
2D-TPE
TaMo
GraphOTTER
OHD
RL
large LoRA sweeps
multi-GPU training
complex experiment-tracking frameworks
```

Keep the first version focused on the CNN-table-encoder question.

---

## Implementation Order

1. Repository skeleton
2. Dataset loader
3. Config system
4. Batched cell tokenizer/pooling
5. Cell MLP
6. 2D CNN
7. Table projector
8. Qwen wrapper + cross-attention injection
9. Smoke-test script
10. Verify forward/backward/generation
11. Training loop
12. Validation evaluation
13. Serialized-table baseline
14. Experiment configs
15. Colab runner
16. README

Do not build the entire experiment framework before the smoke test passes.

---

## Definition of Done

This should work:

```bash
git clone <repo>
cd table-cnn-mrc
pip install -r requirements.txt
export HF_TOKEN=...
python scripts/smoke_test.py
```

Then:

```bash
python scripts/run_experiment.py --config configs/baseline.yaml
```

should train the CNN/cross-attention system and save metrics/checkpoints.

And:

```bash
python scripts/evaluate.py ...
```

should generate deterministic predictions and Exact Match results.

The Colab notebook should only need to clone the repo, install dependencies, load `HF_TOKEN` from Colab userdata, and call the scripts.
