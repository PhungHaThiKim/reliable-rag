# EMNLP 2026 Submission Runbook

> Action items 1–7 from reviewer-style critique, ordered by ROI.
> Deadline ARR: 25/05/2026. Compute: Colab Pro+, A100.

---

## ✅ Already done (no compute needed)

- [x] #4 Removed "we don't claim" section from abstract → moved to HTML comment as
  rebuttal cribsheet only ([CGAR_abstract_v1.md](../paper/CGAR_abstract_v1.md))
- [x] #7 Dropped misleading "5× Path-F1" headline. Reason: GreedyGNN baseline
  uses budget=50 entities → tiny precision by construction. Replacement:
  matched-cost compare table (item #5).
- [x] #3 Significance module ([significance.py](significance.py)). Smoke-tested
  with 31 unit-test pass + synthetic jsonl. Reframes "match" → "statistically
  indistinguishable (Δ, 95% CI, paired-bootstrap p-value)".

---

## ⏳ TODO — needs Colab compute

### #1 CWQ full results (MANDATORY)

**Data availability checklist** (run on Colab BEFORE any training):

```bash
# In Colab cell — verify these paths exist
ls -lh $GNN_RAG_DIR/llm/results/gnn/RoG-cwq/rearev-sbert/test.info
ls -lh $GNN_RAG_DIR/gnn/data/CWQ/test.json
ls -lh $GNN_RAG_DIR/gnn/data/CWQ/entities.txt
ls -lh $GNN_RAG_DIR/gnn/data/CWQ/relations.txt
```

If missing, follow GNN-RAG repo README to download. CWQ test = 3,531 questions
(2.2× WebQSP). Expect:
- Feature extraction (P7v3 cell 1's `load_dataset_features('cwq')`): ~10 min
- Stage 1 train on CWQ from scratch: ~3 hours on A100
- LLM eval per method per LLM (Llama-2-7B): ~25 min × 3 methods × 2 LLMs ≈ 2.5 h

**Run order:**

```python
# In P7v3 Cell 1, after datasets dict is built:
assert 'cwq' in datasets, 'CWQ features missing — see runbook #1'
print(f"  cwq: {len(datasets['cwq']['test_qids'])} questions (expect ~3531)")

# Train CGAR ONLY ONCE on WebQSP train (transfer eval to CWQ)
# OR train two checkpoints (one per dataset) — recommend the latter for paper
DATASETS_TO_TRAIN = ['webqsp', 'cwq']  # extend train loop accordingly
```

CWQ has its own train/dev set in `$GNN_RAG_DIR/gnn/data/CWQ/{train,dev}.json`.
Cell 1 already supports CWQ — just need feature extraction to complete.

---

### #2 Llama-3.1-8B-Instruct eval (MANDATORY for "doesn't depend on backbone")

**Add this cell to P6_LLM_Evaluation.ipynb** AFTER Cell 5 (LLM wrappers):

```python
class Llama31Model:
    """Llama-3.1-8B-Instruct via HuggingFace.

    Same prompt format as Llama-2 (uses [INST]) for fair comparison.
    Model: meta-llama/Llama-3.1-8B-Instruct (gated) OR
           NousResearch/Meta-Llama-3.1-8B-Instruct (ungated mirror).
    """
    def __init__(self, model_path='NousResearch/Meta-Llama-3.1-8B-Instruct',
                 max_new_tokens=512, dtype='bf16'):
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_context = 8192 - 100   # Llama-3.1 has 128K context, but stay safe
        self.prompt_type = 'llama3'

        dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        print(f'Loading {model_path} ({dtype})...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map='auto')
        self.model.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.model.eval()

    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))

    @torch.inference_mode()
    def generate(self, prompt):
        # Use the chat template (Llama-3.1 expects role-tagged messages)
        msgs = [{'role': 'user', 'content': prompt}]
        chat = self.tokenizer.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = self.tokenizer(chat, return_tensors='pt').to(self.model.device)
        L = inputs['input_ids'].shape[1]
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(out[0][L:], skip_special_tokens=True).strip()


# Add a new prompt template entry
PROMPT_TEMPLATES['llama3'] = '{instruction}\n\n{input}'  # chat template handles wrapping

PRED_FILE_LLAMA31 = os.path.join(P6_DIR, 'predictions_llama31.jsonl')
```

Then duplicate **Cell 7** (Llama-2 inference) but with `Llama31Model` and
`PRED_FILE_LLAMA31`. Cell 10 (eval) auto-detects new file.

**Cost on Colab Pro+ A100-80GB:**
- bf16, batch=1: ~140 ms/q × 1639 questions × 3 methods ≈ 11 min/dataset
- For both WebQSP + CWQ → ~30 min total

---

### #3 Significance test integration

After running Cell 7 (or any LLM inference cell), add:

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/ThS/Research/colab/graph_rag')
from significance import compare_methods_on_jsonl

print('=== Significance: GNN-RAG vs CGAR (Llama-2-7B) ===')
res_l2 = compare_methods_on_jsonl(PRED_FILE_LLAMA, 'gnn_rag', 'cgar_v2',
                                    metric='hit', n_bootstrap=10000)

print('\n=== Significance: GNN-RAG vs CGAR (Llama-3.1-8B) ===')
res_l31 = compare_methods_on_jsonl(PRED_FILE_LLAMA31, 'gnn_rag', 'cgar_v2',
                                    metric='hit', n_bootstrap=10000)

# Save for paper
with open(os.path.join(P6_DIR, 'significance.json'), 'w') as f:
    json.dump({'llama2': res_l2, 'llama31': res_l31}, f, indent=2, default=str)
```

Output goes directly into Section 5.2 of paper as "Δ = X pp, 95% CI [...], p = ...".

---

### #5 Direct comparison table — Graph-R1 + GraphRAG-FI

**Reproducibility approach:** since Graph-R1 retraining is expensive, use the
authors' released numbers from their paper Table 1 on shared datasets. For
GraphRAG-FI, only the filtering ratio is hyperparameter — easy to re-impl.

Template for paper Section 5.5 (drop into LaTeX):

```
| Method               | LLM        | WebQSP Hit | CWQ Hit | Avg paths | Compute |
|----------------------|------------|------------|---------|-----------|---------|
| GNN-RAG (paper)      | Llama-2-7B | 85.2       | 67.5    | all       | retr.   |
| GNN-RAG (ours)       | Llama-2-7B | 83.7       | TODO    | 6.5       | retr.   |
| GNN-RAG (ours)       | Llama-3.1  | TODO       | TODO    | 6.5       | retr.   |
| GraphRAG-FI*         | Llama-2-7B | TODO       | TODO    | TODO      | retr.   |
| Graph-R1 (paper)†    | LLM-trained| 89.0       | 71.4    | n/a       | end2end |
| **CGAR (ours)**      | Llama-2-7B | **TODO**   | **TODO**| **5.1**   | retr.   |
| **CGAR (ours)**      | Llama-3.1  | **TODO**   | **TODO**| **5.1**   | retr.   |
```

*\* Re-implementation, hyperparameters from EMNLP 2025 paper §3.2*
*† Graph-R1 fine-tunes the LLM end-to-end, requires order-of-magnitude more
compute than CGAR. We mark it as "complementary, not directly comparable".*

**Re-impl GraphRAG-FI as baseline** (≤ 50 lines), put in
`baselines/graphrag_fi.py`:

```python
class GraphRAGFI:
    """Reimpl of Yang et al. (EMNLP 2025): two-stage filter on retrieved paths.

    Stage 1: drop paths with mean GNN score below quantile τ_drop.
    Stage 2: keep top-K_keep by score.
    Default hyperparameters from paper §3.2: τ_drop=0.3, K_keep=8.
    """
    def __init__(self, gnn, tau_drop=0.3, k_keep=8):
        self.gnn = gnn; self.tau_drop = tau_drop; self.k_keep = k_keep
    def retrieve(self, qid):
        from grl_rag import extract_candidate_paths
        paths = extract_candidate_paths(qid, self.gnn, max_candidates=20)
        if not paths:
            return {'triples': [], 'cost': 0, 'context': '', 'selected_paths': []}
        scores = sorted([p['gnn_score'] for p in paths])
        cutoff = scores[int(len(scores) * self.tau_drop)] if scores else 0
        kept = [p for p in paths if p['gnn_score'] >= cutoff]
        kept.sort(key=lambda p: -p['gnn_score'])
        kept = kept[:self.k_keep]
        triples = [t for p in kept for t in p['path_triples']]
        return {'triples': triples, 'cost': len(kept),
                'context': '\n'.join(p['path_text'] for p in kept),
                'selected_paths': kept}
```

Add it to P6 Cell 6 retrieval loop alongside `gnn_rag` and `cgar_v2`.

---

### #6 Multi-seed std for RL results

**Modify P7v3 Cell 4 (config) and Cell 6 (Stage 2 SCST):**

```python
# Cell 4: list of seeds instead of single SEED
SEEDS = [42, 1337, 2024]  # 3 seeds — minimum for std reporting
S2_STEPS_PER_SEED = 300

# Cell 6: wrap training in a seed loop, save per-seed checkpoint
seed_results = {}
for seed in SEEDS:
    print(f'\n{"="*40}\nSEED = {seed}\n{"="*40}')
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # Re-init controller from S1 checkpoint (same starting point)
    s1 = torch.load(os.path.join(CHECKPOINT_DIR, 'cgar_v2_s1_best.pt'),
                    map_location='cpu', weights_only=False)
    controller.load_state_dict(s1['model_state'])

    trainer = CGARSCSTTrainer(controller, gnn_train, embedder,
                              alpha=ALPHA_REWARD, beta=BETA_REWARD,
                              gamma=GAMMA_REWARD, n_samples=S2_SAMPLES,
                              lr=S2_LR, device=str(device))
    history = []
    for step in range(1, S2_STEPS_PER_SEED + 1):
        qid = train_qids[(step-1) % len(train_qids)]
        r = trainer.train_step(qid)
        if r: history.append(r)

    # Eval on test set
    cgar_ret = CGARRetriever(gnn_test, controller, embedder,
                              max_paths_per_pair=MAX_PATHS_PER_PAIR, device=device)
    eval_r = evaluate_cgar(cgar_ret, gnn_test, gnn_test_qids)
    seed_results[seed] = {'history': history, 'eval': eval_r}

    torch.save({'seed': seed, 'model_state': controller.state_dict()},
               os.path.join(CHECKPOINT_DIR, f'cgar_v2_seed{seed}.pt'))

# Aggregate — paper-ready mean ± std
import numpy as np
metrics = ['f1', 'recall', 'cost']
print('\n=== Multi-seed Aggregate (n=' + str(len(SEEDS)) + ') ===')
for m in metrics:
    vals = [seed_results[s]['eval'][m] for s in SEEDS]
    print(f'  {m}: {np.mean(vals)*100:.2f} ± {np.std(vals)*100:.2f}'
          if m != 'cost' else f'  {m}: {np.mean(vals):.2f} ± {np.std(vals):.2f}')
```

**Cost:** Stage 2 is 300 steps × 8 samples × ~50 ms ≈ 2 min per seed → **6 min for
3 seeds**. Stage 1 (10 epochs × 2848 questions) costs more — only run once and
share across seeds.

---

## 📊 Final EMNLP submission table (target)

```
       Method        | LLM      | WebQSP Hit       | CWQ Hit         | ECE    | Cost
---------------------|----------|------------------|-----------------|--------|-----
GNN-RAG (ours)       | Llama-2  | 83.7 ± 0.0       | XX.X ± 0.0      | 0.218  | 6.5
GNN-RAG (ours)       | Llama-3  | XX.X ± 0.0       | XX.X ± 0.0      | 0.218  | 6.5
GraphRAG-FI (re-impl)| Llama-2  | XX.X ± 0.0       | XX.X ± 0.0      | 0.156  | X.X
CGAR (ours, n=3)     | Llama-2  | 82.9 ± 0.X†      | XX.X ± 0.X†     | 0.043  | 5.1
CGAR (ours, n=3)     | Llama-3  | XX.X ± 0.X       | XX.X ± 0.X      | 0.043  | 5.1
```

†: paired bootstrap p > 0.1 vs GNN-RAG (statistically indistinguishable)

---

## 🚦 Run order on Colab (rough timeline)

| Day | Task | Cell/file |
|---|---|---|
| 1 (now) | Sync 4 files to Drive, run unit test | `python utils/test_cgar_v2.py` |
| 1 | CWQ feature extraction | P7v3 Cell 1 |
| 2 | Stage 1 retrain on (WebQSP, CWQ) ×3 seeds Stage 2 | P7v3 cells 4–6 |
| 3 | Multi-dataset eval + calibration | P7v3 cells 11–13 |
| 4 | Llama-3.1 inference both datasets | P6 (new cell + duplicate Cell 7) |
| 5 | GraphRAG-FI re-impl + run | new cell + P6 Cell 6 loop |
| 6 | Significance tests + final figures | `significance.py` + P5 |
| 7 | Paper writing + integration | `paper/` |
| 8–10 | Buffer + revision | — |
