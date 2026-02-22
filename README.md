# CausalFlip

CausalFlip is an LLM causal-judgment benchmark featuring **semantically similar, label-flipped question pairs** built from event triples \((X, Y, Z)\). It is designed to evaluate whether models make causal judgments grounded in **causal structure**, rather than relying on **spurious semantic correlations**.

This repo currently contains the dataset CSVs and the corresponding train/test splits.

---

## Dataset files

### Full paired datasets
Each row is a **paired instance** containing two causal questions with **opposite** ground-truth labels:

- `Causal_Pairs_Confounder.csv`
- `Causal_Pairs_Chain.csv`
- `Causal_Pairs_Collider.csv`

**Columns (paired files):**
- `Pair`: integer pair id (1..1000)
- `Causal_Relation_1`: question 1 text (Q1)
- `Conclusion_1`: label for Q1 (`Yes` / `No`)
- `Causal_Relation_2`: question 2 text (Q2)
- `Conclusion_2`: label for Q2 (`Yes` / `No`)
- `Classification`: one of `B_D`, `B_A`, `O_D`, `O_A`
- `X`, `Y`, `Z`: natural-language event variables

### Train/test splits (1000 questions each split, per structure)
These files contain **single questions** (one per pair) for training and testing:

- Confounder: `train_confounder.csv`, `test_confounder.csv`
- Chain: `train_chain.csv`, `test_chain.csv`
- Collider: `train_collider.csv`, `test_collider.csv`

**Columns (split files):**
- `data`: question text
- `label`: ground-truth label (`Yes` / `No`)
- `X`, `Y`, `Z`
- `Classification`

---

## Classification codes

Each pair is tagged with a `Classification`:

- **`B_*` vs `O_*` (causal structure):**
  - **Base (B)** uses the *canonical causal graph* for the dataset type (confounder / chain / collider).
  - **Opposite (O)** uses an *alternative causal graph*, constructed so that the **same paired query types** receive the **opposite labels** compared to Base. This Base/Opposite flip prevents a model from mapping a question form (template) to a fixed label, thereby forcing answers grounded in causal structure.

- **`*_D` vs `*_A` (question template / phrasing):**
  - **Default (D)** uses a direct binary question phrasing (e.g., “Will an increase in X cause Y ...?”).
  - **Alternative (A)** uses a different phrasing for the same causal query type (e.g., a declarative statement the model judges as Yes/No).

Combining **Base/Opposite** with **Default/Alternative** yields four categories: **BD, BA, OD, OA** (recorded as `B_D, B_A, O_D, O_A`).
All three datasets include **250 pairs** per classification, for a total of 1000 pairs per structure.

---

## Pairwise train/test split (how the split is constructed)

For each paired dataset (the `Causal_Pairs_*.csv` files), the split is deterministic and alternates by the pair id:

- If `Pair` is **odd**:  
  - `Causal_Relation_1` → **train**, `Causal_Relation_2` → **test**
- If `Pair` is **even**:  
  - `Causal_Relation_2` → **train**, `Causal_Relation_1` → **test**

ecause each dataset is **ordered by category blocks** (BD, BA, then OD, OA) and **each block contains an equal even number of pairs**, alternating assignment by even/odd pair index keeps the number of **category instances balanced** across the training and test splits (avoiding split-by-order bias).

This split ensures that every test question has a **semantically similar counterpart in training** that reuses the same (X, Y, Z) but carries the **opposite label**, so methods that rely on **spurious semantic correlations** are systematically penalized.

---

# Training Strategies Scripts

This repository has three training/evaluation entry points for causal-judgment fine-tuning on the CausalFlip splits:

- `noCoT.py`: **naive pretraining model eval** + **no chain-of-thought fine-tune** (target is only `<FINAL_ANSWER> Yes/No`).
- `explicitCoT.py`: **explicit chain-of-thought fine-tune** trains with **explicit CoT text + final answer**.
- `implicitCR.py`: **implicit causal reasoning** starts with CoT supervision, then **gradually removes CoT tokens during training** (implicit causal reasoning schedule).

All three scripts:

- read paired CSV data (default: confounder),
- build deterministic train/test splits,
- fine-tune `meta-llama/Llama-3.2-3B-Instruct` with LoRA,
- evaluate by generation and parsing the final answer.

## Prerequisites

- Python environment with: `torch`, `transformers`, `datasets`, `trl`, `peft`, `accelerate`, `numpy`, `pandas`, `tqdm`, and `bitsandbytes` (used by `optim="paged_adamw_8bit"`).
- Access to `meta-llama/Llama-3.2-3B-Instruct` on Hugging Face.
- GPU setup (scripts default to `CUDA_VISIBLE_DEVICES=1,2,3,4,5`; override with `--cuda-visible-devices`).

## Script Purposes

### `noCoT.py`

Purpose:

- Naive pretrain model eval.
- NoCoT finetuning.

Default pipeline:

- split -> pre-eval -> train -> post-eval

### `explicitCoT.py`

Purpose:

- Supervised explicit rationale training (CoT text is part of training target).
- Optional `--noisy-prefix` adds an association-related distractor prefix before CoT.

Default pipeline:

- split -> train -> post-eval

### `implicitCR.py`

Purpose:

- Implicit causal reasoning via progressive CoT removal.

Default pipeline:

- split -> train -> post-eval

## How To Run (default on confounder dataset)

### 1) NoCoT baseline

```bash
python noCoT.py
```

Run on different datasets
```bash
python noCoT.py --source-csv Causal_Pairs_Confounder.csv --train-split-csv train_confounder.csv --test-split-csv test_confounder.csv
python noCoT.py --source-csv Causal_Pairs_Chain.csv --train-split-csv train_chain.csv --test-split-csv test_chain.csv
python noCoT.py --source-csv Causal_Pairs_Collider.csv --train-split-csv train_collider.csv --test-split-csv test_collider.csv
```

### 2) Explicit CoT

```bash
python explicitCoT.py
```

Run on different datasets
```bash
python explicitCoT.py --source-csv Causal_Pairs_Confounder.csv --train-split-csv train_confounder.csv --test-split-csv test_confounder.csv
python explicitCoT.py --source-csv Causal_Pairs_Chain.csv --train-split-csv train_chain.csv --test-split-csv test_chain.csv
python explicitCoT.py --source-csv Causal_Pairs_Collider.csv --train-split-csv train_collider.csv --test-split-csv test_collider.csv
```

With noisy prefix (default & on different datasets):

```bash
python explicitCoT.py --noisy-prefix
python explicitCoT.py --source-csv Causal_Pairs_Confounder.csv --train-split-csv train_confounder.csv --test-split-csv test_confounder.csv --noisy-prefix
python explicitCoT.py --source-csv Causal_Pairs_Chain.csv --train-split-csv train_chain.csv --test-split-csv test_chain.csv --noisy-prefix
python explicitCoT.py --source-csv Causal_Pairs_Collider.csv --train-split-csv train_collider.csv --test-split-csv test_collider.csv --noisy-prefix
```

### 3) Implicit Causal Reasoning

```bash
python implicitCR.py
```

Run on different datasets
```bash
python implicitCR.py --source-csv Causal_Pairs_Confounder.csv --train-split-csv train_confounder.csv --test-split-csv test_confounder.csv
python implicitCR.py --source-csv Causal_Pairs_Chain.csv --train-split-csv train_chain.csv --test-split-csv test_chain.csv
python implicitCR.py --source-csv Causal_Pairs_Collider.csv --train-split-csv train_collider.csv --test-split-csv test_collider.csv
```

With noisy prefix (default & on different datasets):

```bash
python implicitCR.py --noisy-prefix
python implicitCR.py --source-csv Causal_Pairs_Confounder.csv --train-split-csv train_confounder.csv --test-split-csv test_confounder.csv --noisy-prefix
python implicitCR.py --source-csv Causal_Pairs_Chain.csv --train-split-csv train_chain.csv --test-split-csv test_chain.csv --noisy-prefix
python implicitCR.py --source-csv Causal_Pairs_Collider.csv --train-split-csv train_collider.csv --test-split-csv test_collider.csv --noisy-prefix
```

## Show full options:

```bash
python noCoT.py --help
python explicitCoT.py --help
python implicitCR.py --help
```
---
## Note: How CoT is formed in code

CoT is constructed from each training instance/causal question, not manually written per question.
  - `explicitCoT.py`, `implicitCR.py`
  - It takes each training instance's event variables `X`, `Y`, `Z` plus its `Classification` (`B_D`, `B_A`, `O_D`, `O_A`).
  - It also infers the dataset type (`confounder`, `chain`, `collider`) from the selected dataset file names.
  - Based on its `Classification` (Base (`B_*`) vs Opposite (`O_*`)) and dataset type, it selects the corresponding CoT template and fills in `X/Y/Z`.
    
For example:
  - For Base instance with event triple (X,Y,Z) in Confounder dataset, the CoT is: `No directed causal path from {X} to {Y} AND Adjusting for {Z} closes the backdoor between {X} and {Y}, therefore `
  - For Opposite instance with event triple (X,Y,Z) in Confounder dataset, the CoT is: `Directed causal path from {X} to {Y} exists AND No relevant backdoor path via {Z} and {Z} is non-causal, therefore `
---
