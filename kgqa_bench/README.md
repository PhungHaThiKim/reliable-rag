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

## Hallucination / groundedness evaluation (Nov 2026) — `kgqa_bench/hallucination.py`

`llm_unsupported_rate` only checked whether an answered entity *appears somewhere* in the
context, which (a) calls "Alice works at X, X is in London → born in London" supported and
(b) flags aliases as hallucinated. The new module separates **faithfulness** (is each asserted
entity entailed by the shown graph evidence?) from **correctness** (is it gold?) and runs offline
on the `records_<dataset>_<method>.jsonl` of any `--reader` run:

```bash
python -m kgqa_bench.hallucination \
    --results results_gnn/b08_reader_Qwen2.5-7B-Instruct \
    --eval gnnrag:ckpt/gnn_features_webqsp_test.pkl:webqsp:test \
    --methods gnnrag_all gnnrag_top5 sbert_top5 cgar --ref gnnrag_all \
    --review-sample 200            # + optional: --judge hf:Qwen/Qwen2.5-7B-Instruct, --strict
```

Per asserted entity (claim), tier 1 is deterministic: `path_endpoint` (terminates a selected
topic→entity path), `in_context_no_proof` (intermediate node only → UNSUPPORTED),
`in_subgraph_only` (in the KG but not retrieved → UNSUPPORTED), `absent` (→ UNSUPPORTED).
Tier 2 compares the path's *directed* relation sequence with the question's reference proofs
(topic→gold relation sequences in the full subgraph; evaluation-only) → SUPPORTED, else
UNCERTAIN (open world); an optional LLM judge can resolve UNCERTAIN into SUPPORTED/CONTRADICTED.

Outputs in `<results>/hallucination/`: `hallucination.json` (metric groups A faithfulness —
claim/question hallucination rate, proof-support, contradiction, uncertain; B correctness —
grounded-correct rate, grounded answer F1, correct-but-ungrounded, abstention; C risk —
hallucination and incorrect-answer risk–coverage ranked by retriever confidence, AURC), paired
bootstrap + McNemar vs `--ref` on question-level labels, an error taxonomy per question
(missing_kg_evidence / candidate_omission / path_selection_failure / wrong_relation_evidence /
reader_unsupported_generation / reader_reasoning_failure / correct_ungrounded / abstained),
`claims_*.jsonl` (per-claim detail), `report_*.md`, `risk_coverage_*.png` and
`review_sample_*.csv` (disagreements first, then UNCERTAIN, then stratified) for human labelling.

Reader prompt variant for the grounded-generation baseline: `--reader hf:<model>:grounded`
(answer only from the paths, copy names verbatim, reply `unknown` when evidence is insufficient);
`is_abstention()` recognises those answers so abstentions are not scored as hallucinations.
