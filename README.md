# CausalFlip

CausalFlip is an LLM causal-judgment benchmark featuring **semantically similar, label-flipped question pairs** built from event triples \((X, Y, Z)\). It is designed to evaluate whether models make causal judgments grounded in **causal structure**, rather than relying on **spurious semantic correlations**.

This repo currently contains the dataset CSVs and the corresponding train/test splits.

---

## Dataset files

### Full paired datasets (1000 pairs each)
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
  - **Opposite (O)** uses an *alternative causal graph over the same (X, Y, Z)*, constructed so that the **same paired query types** receive the **opposite labels** compared to Base. This Base/Opposite flip prevents a model from mapping a question form (template) to a fixed label, thereby forcing answers grounded in causal structure.

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
