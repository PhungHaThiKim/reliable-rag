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

## Confidence-gated escalation (Nov 2026) — `kgqa_bench/escalation.py`

GNN-RAG+RA gains by *adding* evidence; CGAR only *removes* it (−2 pp F1 vs all at 41 % cost) but has
the best AURC. `escalation.py` evaluates the two-tier policy "CGAR selection if confidence ≥ τ,
else the fallback's paths" **offline** from two existing records files on the same qids (the
fallback is a superset, so an escalated question is exactly the fallback's record). It sweeps τ over
confidence quantiles, adds the `oracle` gate (upper bound) and same-qid top-K reference points,
and reports operating points (match-fallback-quality, best-at-60 %-cost, knee). With reader
records it also gives llm_hit / llm_f1 / unsupported / real prompt tokens per τ.

```bash
python -m kgqa_bench.escalation --results results_gnn/b06_main --dataset webqsp:test --primary cgar --fallback gnnrag_all --plot
python -m kgqa_bench.escalation --results results_gnn/b08_reader_Qwen2.5-7B-Instruct --dataset webqsp:test --quality llm_hit
```

## CGAR v3 options and offline analyses (Nov 2026)

All default OFF, so `cgar` without flags is still the v2.1 controller:

| flag | item | what changes |
|---|---|---|
| `--cgar-learned-stop [--cgar-stop-threshold 0.5 --cgar-w-stop 0.3]` | B1 | a `stop_head` predicts, before each MMR step, P(the best remaining path adds no new answer entity); eval stops when it exceeds the threshold. Replaces the K-level SPRT (never fires on a ≤6-candidate pool) as the adaptive-cost mechanism. Trained in Stage 1 (teacher forcing: full budget in train mode). |
| `--cgar-rel-label proof` | B2 | relevance label = the path's *directed* relation sequence equals a reference proof (shortest topic→gold relation sequence in the subgraph) instead of "path touches a gold entity"; falls back to the entity label when a question has no proof. |
| `--cgar-conf-target hit` | B3 | confidence head trained with BCE against hit (1[recall>0]) instead of MSE against recall, i.e. against what ECE/AURC measure. |
| `--cgar-s2-steps 0` | B4 | S1 is the main method (S2 REINFORCE: +0.19 pp, p=0.033, 7.8 % non-zero advantages). |

Checkpoints: v3 checkpoints contain `stop_head.*`; pre-v3 checkpoints load with `learned_stop=False`
and are refused with `learned_stop=True` (`CGARMethod.load`, strict on everything but the stop head).

`python -m kgqa_bench.analysis` (A2–A4): adaptivity split per question (`controller_stopped_early`
vs `pool_limited` vs `full_budget`, using the exact number of candidate paths), F1/Hit-ctx by
#candidates bin, paired bootstrap + McNemar for arbitrary `A:B` pairs, selective-answering table
and figure (`sel_hit@c`, AURC) for every method with a confidence.

Reader parsing (D2): `extract_answer_entities` now drops chat openers ("Here are the possible
answers:", "The answer is:" headers), markdown bold/italics and quotes before matching, which
fixes `llm_hit1 ≈ 0.07` on Llama-3.1 outputs. `llm_hit` (substring on the raw answer) is unchanged.

## P9 notebook - detailed hallucination evaluation

`P9_Hallucination_Eval_Colab.ipynb` runs the groundedness evaluation offline on every
`results_gnn/b08*_reader_*` directory produced by P8 cell 08: metric groups A/B/C per reader and
prompt, claim label x evidence-type cross-tabs against the legacy proxy, worked examples per label,
error taxonomy and per-qid CGAR-vs-all questions, risk-coverage curves, paired tests, chat vs
grounded prompt, confidence-gated escalation at the hallucination level, optional LLM judge,
human-review CSV loading with detector precision/recall and Cohen's kappa, and a final checklist.

## Rescore reader outputs offline (`kgqa_bench.rescore`, Nov 2026)

`records_*.jsonl` keep the raw `llm_pred`, so every `llm_*` metric can be recomputed without a GPU
after a parser or abstention-rule change:

```
python -m kgqa_bench.rescore --results results_gnn/b08_reader_Qwen2.5-7B-Instruct results_gnn/b08g_reader_Qwen2.5-7B-Instruct
```

* abstentions (`unknown`, `cannot determine`, ...) become `llm_abstained=1`, hit/hit1/f1 = 0 and
  `llm_unsupported_rate = 0` (the old proxy counted the literal string `unknown` as an unsupported entity,
  which inflated unsupported_rate under the grounded prompt);
* `llm_unsupported_rate` is now "fabricated entities among answered questions"; the old value is kept as
  `llm_unsupported_rate_legacy`; originals are backed up once to `records_*.jsonl.orig`;
* `results.json` summaries, paired comparisons (`llm_hit`, `llm_hit1`, `llm_f1` vs `--ref`) and `report.md` are rewritten.
  P8 cell 08 calls it after every reader run, including `[skip]`ped dirs, so the Llama chat `llm_hit1` (old parser) is fixed in place.
