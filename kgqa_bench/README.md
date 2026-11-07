# kgqa_bench

One harness, one metric suite, any KGQA dataset — for CGAR, PRISM and the
baselines (GNN-RAG all/top-K, GraphRAG-FI, Random-K, no-retrieval reader).

```
python tests/test_kgqa_bench.py                      # CPU smoke tests (no SBERT / HF needed)
python -m kgqa_bench.cli run --help
python -m kgqa_bench.cli run \
    --train rog:rmanluo/RoG-webqsp:train \
    --eval  rog:rmanluo/RoG-webqsp:test rog:rmanluo/RoG-cwq:test \
    --methods gnnrag_all gnnrag_top10 sbert_top10 graphrag_fi cgar prism \
    --seeds 42 1337 2024 --sweep --reader hf:Qwen/Qwen2.5-7B-Instruct \
    --out results/bench_v1
python -m kgqa_bench.cli report --results results/bench_v1
```

Dataset specs: `rog:<hf_name>:<split>` · `gnnrag:<pkl>:<name>:<split>` ·
`jsonl:<path>:<name>:<split>` · `synthetic:<n>:<seed>[:split]` · `pkl:<saved BenchDataset>`.

Modules: `data` (records + loaders) → `adapters` (CGAR features / PRISM samples from the
same candidates & paths) → `methods` (runners → per-question records) → `metrics`
(retrieval, cost, calibration, selective, answer-level, CI, paired tests, strata, seeds)
→ `llm_reader` (HF readers, GNN-RAG prompt + scoring) → `report` (markdown / LaTeX / Pareto /
reliability). Training: `cgar_train.py` (S1 + SCST), `methods.PRISMMethod.fit`
(distill + RL with eval-based checkpointing).

See `../../paper/CGAR_PRISM_code_review_and_metrics_plan.md` for the review findings, the
metric definitions and the experiment plan.

## Fair baselines (Nov 2026)

* `--baseline-candidates N` (default 20): every path-dict baseline (`gnnrag_*`, `random_k*`,
  `graphrag_fi*`, `sbert_top*`) draws from the same N-candidate pool CGAR/PRISM can reach
  (datasets carry 20 ranked candidates). Use `10` to reproduce the Seminar-1 "GNN-RAG-all" row.
* `sbert_topK`: non-learned baseline ranking paths by cosine(question, path) with the run's
  embedder. CGAR must beat it clearly at equal cost, otherwise its gain is the embedding, not the controller.
* `report.md` now states per dataset whether the pool is GNN-RAG-scored or heuristic
  (heuristic `gnnrag_topK` rows must be read as *heuristic-topK*; their ECE is meaningless) and
  adds an **Adaptivity** section (cost / chosen_k histogram) for cgar/prism.
