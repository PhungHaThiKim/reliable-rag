"""End-to-end tests for kgqa_bench on synthetic data (CPU, no SBERT/HF).

Run:  cd colab/graph_rag && python -m pytest -q tests/test_kgqa_bench.py
  or: python tests/test_kgqa_bench.py
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from kgqa_bench.data import BenchDataset, load_spec  # noqa: E402
from kgqa_bench.adapters import PrismVocab, build_prism_samples, to_gnn_features  # noqa: E402
from kgqa_bench import metrics as M  # noqa: E402
from kgqa_bench.methods import (CGARMethod, GNNRAGMethod, GraphRAGFIMethod,  # noqa: E402
                                PRISMMethod, RandomKMethod)
from kgqa_bench.llm_reader import DummyReader, attach_llm_answers  # noqa: E402
from grl_rag import Embedder, extract_candidate_paths, extract_candidate_paths_by_k  # noqa: E402


def _emb():
    return Embedder(fallback="hash")


def test_synthetic_dataset_roundtrip():
    ds = BenchDataset.synthetic(12, seed=3).annotate_hops()
    assert len(ds) == 12
    st = ds.stats()
    assert st["answer_in_subgraph_rate"] == 1.0          # planted gold is always reachable
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.pkl"); ds.save(p)
        ds2 = BenchDataset.load(p)
        assert ds2[0].triples == ds[0].triples and ds2.key == ds.key
    assert len(load_spec("synthetic:5:1")) == 5


def test_paths_by_k_equals_per_k_extraction():
    ds = BenchDataset.synthetic(6, seed=5)
    gnn = to_gnn_features(ds)
    for q in ds:
        by_k = extract_candidate_paths_by_k(q.qid, gnn, k_options=(5, 10, 20), max_paths_per_pair=3)
        for K in (5, 10, 20):
            ref = extract_candidate_paths(q.qid, gnn, max_candidates=K, max_paths_per_pair=3)
            assert {p["path_text"] for p in ref} == {p["path_text"] for p in by_k[K]}, (q.qid, K)


def test_prism_samples_share_candidates_with_cgar():
    ds = BenchDataset.synthetic(8, seed=7)
    vocab = PrismVocab().fit(ds)
    samples = build_prism_samples(ds, vocab, _emb())
    assert samples, "no samples built"
    by_qid = {q.qid: q for q in ds}
    for s in samples:
        q = by_qid[s.qid]
        # candidate order == BenchQuestion ranking (prefix), sizes consistent
        assert s.candidate_names == [c for c in q.top_candidates if c in s.candidate_names][: len(s.candidate_names)]
        assert s.paths.size(0) == len(s.path_texts) == len(s.path_triples)
        assert abs(float(s.teacher_distribution.sum()) - 1) < 1e-4
        assert s.paths.max() < len(vocab)
    # eval mode must NOT force gold into the candidates
    from kgqa_bench.adapters import PrismBuildConfig
    s_forced = build_prism_samples(ds, vocab, _emb(), PrismBuildConfig(force_gold_in_candidates=True))
    assert len(s_forced) >= len(samples)


def test_baselines_and_metrics():
    ds = BenchDataset.synthetic(20, seed=11).annotate_hops()
    recs_all = GNNRAGMethod(None).run(ds, verbose=False)
    recs_top3 = GNNRAGMethod(3).run(ds, verbose=False)
    recs_fi = GraphRAGFIMethod(k_keep=4).run(ds, verbose=False)
    recs_rand = RandomKMethod(3).run(ds, verbose=False)
    for recs in (recs_all, recs_top3, recs_fi, recs_rand):
        assert len(recs) == 20
        for r in recs:
            assert 0 <= r["coverage"] <= 1 and 0 <= r["precision"] <= 1 and r["hit_ctx"] in (0, 1)
            assert r["hops"] is not None
    s_all, s_top = M.summarize_records(recs_all), M.summarize_records(recs_top3)
    assert s_top["cost"] <= s_all["cost"] and s_all["cov"] >= s_top["cov"]
    assert "f1_ci_low" in s_all and s_all["f1_ci_low"] <= s_all["f1"] <= s_all["f1_ci_high"]
    assert "ece" in s_all  # gnn score used as confidence
    strat = M.stratify_records(recs_all, "hops")
    assert sum(v["n"] for v in strat.values()) == 20
    cmp = M.compare_records(recs_all, recs_top3, "hit_ctx", n_bootstrap=200)
    assert cmp["n"] == 20 and "mcnemar" in cmp
    agg = M.aggregate_seeds({1: s_all, 2: s_top})
    assert agg["f1"]["n"] == 2


def test_sbert_topk_and_shared_pool():
    from kgqa_bench.methods import SBERTTopKMethod, baseline_from_name
    ds = BenchDataset.synthetic(12, seed=13).annotate_hops()
    emb = _emb()
    recs = SBERTTopKMethod(emb, k=4).run(ds, verbose=False)
    assert len(recs) == 12 and all(r["cost"] <= 4 for r in recs) and all(r["confidence"] is not None for r in recs)
    # shared pool: baselines default to 20 candidates (CGAR's reachable pool); 10 reproduces Seminar-1 "all"
    all20 = M.summarize_records(baseline_from_name("gnnrag_all").run(ds, verbose=False))
    all10 = M.summarize_records(baseline_from_name("gnnrag_all", max_candidates=10).run(ds, verbose=False))
    assert all20["cost"] >= all10["cost"] and all20["cov"] >= all10["cov"]
    m = baseline_from_name("sbert_top3", embedder=emb); assert m.name == "sbert_top3"
    try:
        baseline_from_name("sbert_top3"); assert False, "needs embedder"
    except ValueError:
        pass


def test_cgar_train_and_run():
    torch.manual_seed(0)
    tr = BenchDataset.synthetic(24, seed=21, split="train")
    ev = BenchDataset.synthetic(10, seed=22).annotate_hops()
    emb = _emb()
    with tempfile.TemporaryDirectory() as d:
        m = CGARMethod(emb, device="cpu", hidden_dim=32, rel_threshold=None)
        info = m.fit(tr, ev, s1_epochs=1, s2_steps=4, s2_samples=2, eval_max=10, ckpt_dir=d, verbose=False)
        assert info["stage1"]["history"] and "eval_f1" in info["stage1"]["history"][0]
        assert os.path.exists(os.path.join(d, "cgar_s1_seed42.pt"))
        recs = m.run(ev, verbose=False)
        assert len(recs) == 10 and all(r["confidence"] is not None for r in recs)
        s = M.summarize_records(recs)
        assert "ece" in s and "aurc" in s and s["cost"] > 0
        # reload
        m2 = CGARMethod(emb, device="cpu", hidden_dim=32).load(os.path.join(d, "cgar_s1_seed42.pt"))
        assert len(m2.run(ev, verbose=False)) == 10
        # adaptive path count variant selects <= legacy
        m3 = CGARMethod(emb, device="cpu", hidden_dim=32, rel_threshold=50.0, min_select=1)
        m3.controller.load_state_dict(m.controller.state_dict())
        assert M.summarize_records(m3.run(ev, verbose=False))["cost"] <= s["cost"]


def test_prism_train_and_run():
    torch.manual_seed(0)
    from prism.config import Config
    cfg = Config(device="cpu", d_model=32, n_layers=1, n_heads=2, distill_epochs=1, rl_steps=4,
                 rl_n_samples=2, rl_eval_every=2, max_paths=20, n_candidates=8, max_select_paths=4)
    tr = BenchDataset.synthetic(24, seed=31, split="train")
    ev = BenchDataset.synthetic(10, seed=32).annotate_hops()
    with tempfile.TemporaryDirectory() as d:
        cfg.out_dir = d
        m = PRISMMethod(_emb(), cfg=cfg, device="cpu")
        info = m.fit(tr, ev, verbose=False, ckpt_dir=d)
        assert info["teacher"] == "oracle_heuristic" and os.path.exists(info["distill_ckpt"])
        recs = m.run(ev, verbose=False)
        assert len(recs) == 10
        for r in recs:
            assert r["las_hit1"] in (0, 1) and r["confidence"] is not None and 0 < r["cost"] <= cfg.max_select_paths
            assert r["n_steps"] == r["cost"] and r["mean_step_kl"] >= 0
        s = M.summarize_records(recs)
        assert "las_hit1" in s and "ece" in s
        m.save(os.path.join(d, "prism.pt"))
        m2 = PRISMMethod(_emb(), cfg=cfg, device="cpu").load(os.path.join(d, "prism.pt"),
                                                            vocab_path=os.path.join(d, "prism.pt.vocab.json"))
        assert len(m2.run(ev, verbose=False)) == 10


def test_llm_reader_attach():
    ds = BenchDataset.synthetic(8, seed=41).annotate_hops()
    recs = GNNRAGMethod(None).run(ds, verbose=False)
    attach_llm_answers(recs, ds, DummyReader(), verbose=False)
    for r in recs:
        assert r["llm_hit"] in (0, 1) and 0 <= r["llm_f1"] <= 1 and "llm_unsupported_rate" in r
        # the oracle extractor can only answer what is in the context
        assert r["llm_hit"] <= r["hit_ctx"] or r["llm_hit"] == 0
    assert "llm_unsupported_rate" in M.summarize_records(recs) and "n_tokens_prompt" in M.summarize_records(recs)
    s = M.summarize_records(recs)
    assert "llm_hit" in s and "ece_llm" in s


def test_drift_dataset():
    from kgqa_bench.data import drift_dataset
    ds = BenchDataset.synthetic(10, seed=51).annotate_hops()
    d1 = drift_dataset(ds, 0.5, "relation_oov", seed=1)
    rels = {r for q in d1 for (_h, r, _t) in q.triples}
    assert "UNK_REL" in rels and len(d1) == len(ds) and d1[0].qid == ds[0].qid
    d2 = drift_dataset(ds, 0.5, "score_noise", seed=1)
    assert any(d2[i].top_candidates != ds[i].top_candidates for i in range(len(ds)))
    assert d2[0].triples == ds[0].triples
    # both methods run unchanged on the drifted copy; unseen relation -> <UNK> for PRISM
    vocab = PrismVocab().fit(ds)
    samples = build_prism_samples(d1, vocab, _emb())
    assert any((s.paths == vocab.UNK).any() for s in samples)
    assert len(GNNRAGMethod(3).run(d1, verbose=False)) == 10


def test_cli_smoke():
    from kgqa_bench.cli import main
    with tempfile.TemporaryDirectory() as d:
        main(["run", "--train", "synthetic:20:0:train", "--eval", "synthetic:8:1", "synthetic:6:2",
              "--methods", "gnnrag_all", "gnnrag_top3", "sbert_top3", "graphrag_fi_k4", "cgar", "prism",
              "--embedder", "hash", "--device", "cpu", "--dev-split", "0.2",
              "--cgar-s1-epochs", "1", "--cgar-s2-steps", "2", "--cgar-s2-samples", "2",
              "--prism-distill-epochs", "1", "--prism-rl-steps", "2", "--prism-d-model", "32",
              "--prism-layers", "1", "--prism-candidates", "8",
              "--reader", "dummy", "--no-retrieval-baseline", "--n-bootstrap", "100", "--quiet",
              "--sweep", "--sweep-max", "5", "--max-items", "7", "--subset-seed", "3",
              "--out", d])
        R = json.load(open(os.path.join(d, "results.json")))
        assert set(R["summaries"]) == {"synthetic:test", "synthetic:test"} or len(R["summaries"]) >= 1
        for ds, by_m in R["summaries"].items():
            assert {"gnnrag_all", "gnnrag_top3", "sbert_top3", "graphrag_fi_k4", "cgar", "prism", "no_retrieval"} <= set(by_m)
        rep = open(os.path.join(d, "report.md")).read()
        assert "Adaptivity" in rep and "heuristic" in rep and "cgar_cost_sweep" in rep and "prism_eps_sweep" in rep
        sw = R["train_logs"]["cgar_cost_sweep"]; assert {p["max_paths"] for p in sw} == {3, 5, 10, 20}
        sw = R["train_logs"]["prism_eps_sweep"]; assert any(p["eps"] == 0.0 and p["max_select"] == 3 for p in sw)
        assert os.path.exists(os.path.join(d, "report.md"))
        assert any(k.startswith("cgar_vs_gnnrag_all") for k in next(iter(R["comparisons"].values())))


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  ✓ {name}")
            except Exception:
                fails += 1; print(f"  ✗ {name}"); traceback.print_exc()
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
