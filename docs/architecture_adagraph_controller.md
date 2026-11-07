# AdaGraph Controller for GRL-RAG

## Overview

The AdaGraph Controller is an **adaptive budgeted path selection module** that sits ON TOP of the GNN-RAG heuristic. It does not replace the GNN — instead, it controls **which** reasoning paths from GNN-RAG to keep, **how many** to select, and **when** to stop exploring.

```
┌────────────────────────────────────────────────────────────────────┐
│                     GNN-RAG + AdaGraph Controller                  │
│                                                                    │
│  Phase 2 Output (frozen):                                          │
│    GNN-RAG → entity scores + subgraph + candidate paths            │
│                                                                    │
│  AdaGraph Controller (learned):                                    │
│    ┌───────────────────────────────────────────────────────┐       │
│    │  Round 1:                                             │       │
│    │    Marginal utility Δ(path|S) per candidate path      │       │
│    │    → Select top-k paths → add to evidence set S       │       │
│    │    → Update evidence (GRU) → Check confidence (SPRT)  │       │
│    │                                                       │       │
│    │  Round 2: (if confidence < β)                         │       │
│    │    Recompute Δ(path|S) with updated evidence          │       │
│    │    → Select next top-k → evidence update → SPRT check │       │
│    │                                                       │       │
│    │  ... until: confidence ≥ β  OR  budget exhausted      │       │
│    └───────────────────────────────────────────────────────┘       │
│                                                                    │
│  Output: selected paths → verbalize → LLM → answer                │
└────────────────────────────────────────────────────────────────────┘
```

## Why This Architecture?

### The Problem

GNN-RAG (ACL 2025) extracts ALL shortest paths between query entities and top-K candidate entities. This creates a problem:

- Too many paths → LLM context overflow, confusion, token waste
- Too few paths → missing evidence, wrong answer
- Fixed top-K → no adaptation to query difficulty

### The Solution

AdaGraph Controller adds three capabilities:

1. **Marginal Utility Ranking** — which paths add the most NEW information?
2. **Budget-Aware Selection** — how many paths can we afford?
3. **SPRT Confidence Stopping** — when do we have ENOUGH evidence?

### Theoretical Backing

**Theorem 1 (Adaptive Submodularity):** Path utility $f(S) = |\text{entities}(S) \cap \text{gold}| / |\text{gold}|$ is monotone submodular → greedy selection achieves $(1-1/e) \cdot OPT$.

**Theorem 2 (SPRT Optimality):** The confidence-based stopping rule minimizes expected rounds: $\mathbb{E}[\tau] \leq \log(1/\delta) / D_{KL}$.

## Components

### 1. MarginalUtilityHead

Estimates $\Delta(\text{path} \mid S_t)$ — how much NEW information a candidate path adds to the current evidence set.

```
Input per path (emb_dim*3 + 4 = 1156 dims):
  ┌─────────────┬─────────────────┬───────────┬──────────────────────┐
  │ path_emb    │ evidence_state  │ query_emb │ [gnn_score, len,     │
  │ (384)       │ (384)           │ (384)     │  redundancy, step]   │
  └─────────────┴─────────────────┴───────────┴──────────────────────┘
                              │
                  MLP: 1156 → 256 → 128 → 1
                              │
                        Δ(path|S_t) = scalar
```

**Key feature: redundancy input.** The marginal head receives an explicit redundancy score from the `RedundancyHead`, preventing selection of paths that cover the same entities as already-selected paths.

### 2. ConfidenceHead (SPRT)

Estimates $P(\text{answer correct} \mid \text{evidence}_{1:t})$.

```
Input (emb_dim*2 + 3 = 771 dims):
  ┌─────────────────┬───────────┬───────────────────────────────────────┐
  │ evidence_state  │ query_emb │ [n_selected/budget, remaining, step] │
  │ (384)           │ (384)     │ (3)                                   │
  └─────────────────┴───────────┴───────────────────────────────────────┘
                              │
                  MLP: 771 → 128 → 1 → sigmoid
                              │
                     confidence ∈ (0, 1)
```

**SPRT rule:** Stop when `confidence > β` (default β=0.90).

### 3. EvidenceAccumulator (GRU)

Maintains running evidence summary across selection rounds.

```
evidence_0 = Linear(query_emb)         # initialize from query
evidence_t = GRU(path_emb, evidence_{t-1})  # update with each selected path
```

**Why GRU?** Unlike mean pooling, GRU has gating — it can:
- Forget irrelevant early evidence
- Emphasize important new evidence
- Handle variable-length selection sequences

### 4. RedundancyHead

Detects how similar each candidate path is to already-selected evidence.

```
Input: [path_emb, evidence_state] → MLP → sigmoid → redundancy ∈ (0,1)
```

High redundancy → marginal utility decreases → path less likely to be selected. This directly implements the diminishing returns property of submodular functions.

### 5. AdaGraphController (Orchestrator)

Ties all components together in a multi-round selection loop:

```python
for round in range(max_rounds):
    # 1. Check confidence → maybe stop (SPRT)
    conf = confidence_head(evidence, query, budget_features)
    if conf > β: break

    # 2. Compute redundancy of remaining paths
    redundancy = redundancy_head(paths, evidence)

    # 3. Estimate marginal utility per path
    marginals = marginal_head(paths, evidence, query, [gnn_score, len, redundancy, step])
    marginals[already_selected] = -inf

    # 4. Select top-k by marginal utility
    topk = marginals.topk(k)
    selected[topk] = True

    # 5. Update evidence with each selected path
    for path in topk:
        evidence = gru(path_emb, evidence)
```

## Integration with GRL-RAG Pipeline

### Before (Phase 4a: PathSelectionRetriever)

```python
# Old: score all paths, take top-K
selection_probs = policy(query_emb, path_embs, gnn_scores, lengths)
if greedy:
    selected = selection_probs.topk(max_paths)
else:
    selected = bernoulli(selection_probs)
```

**Problem:** One-shot scoring, no iterative refinement, no budget awareness.

### After (AdaGraph Controller)

```python
# New: iterative selection with evidence accumulation
retriever = AdaGraphPathRetriever(gnn, controller, embedder)
result = retriever.retrieve(qid)
# result['selected_paths']: adaptively chosen subset
# result['context']: verbalized for LLM
```

**Key improvements:**
- Multi-round selection with GRU evidence tracking
- Marginal utility prevents redundant paths
- SPRT stops early for easy questions
- Budget constraint prevents context overflow

## Training

### Stage 1: Supervised Pretraining

```python
loss_fn = AdaGraphControllerLoss(
    w_marginal=0.4,      # predict which paths lead to gold entities
    w_confidence=0.3,    # calibrate confidence to actual recall
    w_budget=0.3,        # penalize excessive selection
)
```

**L_marginal supervision:** For each path, actual marginal = 1 if path contains gold entity, 0 otherwise.

### Stage 2: SCST Fine-tuning

```python
trainer = SCSTTrainer(
    controller, gnn, embedder,
    alpha=0.6,   # answer coverage (main reward)
    beta=0.2,    # efficiency (fewer paths = better)
    gamma=0.2,   # diversity (penalize redundant paths)
    n_samples=8,
)

for qid in train_qids:
    result = trainer.train_step(qid)
```

**SCST reward:** `R = 0.6 * coverage + 0.2 * efficiency + 0.2 * diversity`

**Baseline:** Greedy controller output reward. Advantage = R_sample - R_baseline.

## Hyperparameters

```python
# Controller
emb_dim       = 384          # SBERT embedding dimension (all-MiniLM-L6-v2)
hidden_dim    = 256          # MLP hidden dimension
max_rounds    = 5            # maximum selection rounds
select_k      = 3            # paths selected per round
max_paths     = 15           # total budget (max selected paths)
sprt_beta     = 0.90         # SPRT confidence threshold

# Training
s1_lr         = 3e-4         # Stage 1 learning rate
s2_lr         = 1e-5         # Stage 2 (SCST) learning rate
s2_samples    = 8            # SCST samples per question
```

## Expected Experiments

### Comparison Table

| Method | Answer Coverage | Avg Paths | LLM Tokens | EM | F1 |
|--------|:--------------:|:---------:|:----------:|:--:|:--:|
| GNN-RAG (all paths) | High | ~30-50 | Large | X | X |
| GNN-RAG (top-10) | Medium | 10 | Medium | X | X |
| PathSelectionRetriever (Phase 4a) | Medium | ~10 | Medium | X | X |
| **AdaGraph Controller** | **High** | **~5-10 adaptive** | **Small** | **X+?** | **X+?** |

### Key Experiments

1. **Coverage vs Cost Pareto:** Vary max_paths and sprt_beta, plot coverage vs avg paths selected
2. **SPRT Tradeoff:** β sweep showing easy questions need ~3 paths, hard need ~12
3. **Ablation:** Remove marginal head / confidence head / GRU → show each component matters
4. **vs Baselines:** GNN-RAG (all), GNN-RAG (top-K), GreedyGNN, TraversalRL, PathSelection
5. **LLM End-to-End:** EM/F1 with Llama-2/ChatGPT using selected paths vs all paths

## File Structure

```
colab/graph_rag/
├── grl_rag.py                          # Phase 1-4 shared classes (existing)
├── adagraph_controller.py              # AdaGraph Controller (NEW)
├── docs/
│   └── architecture_adagraph_controller.md  # This document
├── P0-P6 notebooks                     # Existing pipeline
└── P7_AdaGraph_Controller.ipynb        # NEW notebook (to be created)
```

## Relationship to Other Versions

```
AdaGraph v1 (original):     RL policy on custom GNN → no LLM, no paths
AdaGraph v2 (PonderNet):    Differentiable halting on custom GNN → no LLM
AdaGraph v3 (Budgeted):     Budgeted selection on custom GNN → no LLM
OptStop-GR:                 SPRT optimal stopping on custom GNN → no LLM

  ─── PARADIGM SHIFT ───

AdaGraph Controller:        Budgeted selection + SPRT stopping
                            ON TOP OF GNN-RAG (existing, proven GNN)
                            WITH LLM evaluation (end-to-end EM/F1)
                            = "learned adaptive selector over strong graph heuristic"
```

This is the version that addresses ALL expert review concerns:
- Uses a strong GNN baseline (GNN-RAG, ACL 2025)
- Has end-to-end LLM evaluation (EM/F1)
- Clean narrative: "budgeted control over graph heuristic"
- Principled stopping (SPRT)
- Proper comparison with SOTA
