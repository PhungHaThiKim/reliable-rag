"""Tests for the Nov-2026 improvements: CGAR v3 options (learned stop, proof labels,
hit confidence) and kgqa_bench.analysis. CPU, synthetic, hash embedder."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from kgqa_bench.data import BenchDataset  # noqa: E402
from kgqa_bench.methods import CGARMethod, GNNRAGMethod, SBERTTopKMethod  # noqa: E402
from kgqa_bench.cgar_train import proof_info  # noqa: E402
from kgqa_bench.adapters import to_gnn_features  # noqa: E402
from kgqa_bench import analysis as A  # noqa: E402
from grl_rag import Embedder  # noqa: E402
from adagraph_cgar_v2 import CGARLoss, directed_relation_sequence  # noqa: E402


def _emb():
    return Embedder(fallback="hash")


def test_proof_info_and_directed_sequence():
    ds = BenchDataset.synthetic(6, seed=31)
    gnn = to_gnn_features(ds)
    q = ds[0]
    tset, proofs = proof_info(gnn, q.qid)
    assert tset and proofs, "planted gold path must yield a reference proof"
    # a gold path taken from the triples is forward-directed; reversed traversal gets inv:
    h, r, t = next(iter(tset))
    assert directed_relation_sequence([(h, r, t)], tset) == (r,)
    assert directed_relation_sequence([(t, r, h)], tset) == ("inv:" + r,)
    assert directed_relation_sequence([("x", "nope", "y")], tset) == ("missing:nope",)


def test_cgar_v3_train_run_and_ckpt_compat():
    tr = BenchDataset.synthetic(16, seed=32, split="train")
    te = BenchDataset.synthetic(8, seed=33)
    emb = _emb()
    with tempfile.TemporaryDirectory() as d:
        m = CGARMethod(emb, device="cpu", max_paths=3, learned_stop=True, stop_threshold=0.5)
        info = m.fit(tr, te, s1_epochs=1, s2_steps=0, eval_max=8, ckpt_dir=d, verbose=False,
                     loss_kw=dict(conf_target="hit", rel_label="proof", w_stop=0.3))
        assert info["loss_kw"]["rel_label"] == "proof" and info["controller"]["learned_stop"] is True
        recs = m.run(te, verbose=False)
        assert len(recs) == 8 and all(0 <= r["confidence"] <= 1 for r in recs)
        assert all(r["cost"] <= 3 for r in recs)
        # the stop head is in the checkpoint; a legacy controller (learned_stop=False) still loads it
        ck = os.path.join(d, "cgar_s1_seed42.pt")
        assert os.path.exists(ck)
        legacy = CGARMethod(emb, device="cpu", max_paths=3).load(ck)
        assert len(legacy.run(te, verbose=False)) == 8
        # a pre-v3 checkpoint (no stop_head) loads when learned_stop=False, fails when True
        sd = {k: v for k, v in torch.load(ck, map_location="cpu", weights_only=False)["model_state"].items()
              if not k.startswith("stop_head.")}
        old = os.path.join(d, "old.pt"); torch.save({"model_state": sd}, old)
        CGARMethod(emb, device="cpu", max_paths=3).load(old)
        try:
            CGARMethod(emb, device="cpu", max_paths=3, learned_stop=True).load(old)
            assert False, "should refuse"
        except RuntimeError:
            pass


def test_loss_options_change_targets():
    tr = BenchDataset.synthetic(4, seed=34, split="train")
    emb = _emb()
    m = CGARMethod(emb, device="cpu", max_paths=3, learned_stop=True)
    gnn = to_gnn_features(tr)
    from grl_rag import extract_candidate_paths_by_k
    q = tr[0]
    by_k = extract_candidate_paths_by_k(q.qid, gnn, k_options=m.controller.K_OPTIONS, max_paths_per_pair=3)
    m.controller.train()
    sel, det = m.controller(emb.encode(q.question).clone().unsqueeze(0), by_k, emb, deterministic=True, return_details=True)
    assert det["stop_logits"] and len(det["stop_logits"]) == len(det["stop_candidates"]) == len(sel)
    gold = set(q.answers)
    tset, proofs = proof_info(gnn, q.qid)
    l_ent = CGARLoss(rel_label="entity", conf_target="recall")(det, gold, by_k, actual_recall=0.5)
    l_prf = CGARLoss(rel_label="proof", conf_target="hit", w_stop=0.3)(det, gold, by_k, actual_recall=0.5,
                                                                        gold_proofs=proofs, triple_set=tset)
    assert "stop" in l_prf and l_prf["stop"] > 0 and l_ent["stop"] == 0
    l_prf["total"].backward()   # gradient reaches stop_head
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.controller.stop_head.parameters())


def test_analysis_cli():
    ds = BenchDataset.synthetic(14, seed=35, name="synthetic35").annotate_hops()
    emb = _emb()
    recs = {"gnnrag_all": GNNRAGMethod(None).run(ds, verbose=False), "gnnrag_top5": GNNRAGMethod(5).run(ds, verbose=False),
            "sbert_top5": SBERTTopKMethod(emb, 5).run(ds, verbose=False)}
    m = CGARMethod(emb, device="cpu", max_paths=3, learned_stop=True); m.controller.eval()
    recs["cgar@seed42"] = m.run(ds, verbose=False)
    n_paths = A.paths_available(ds)
    ad = A.adaptivity_breakdown(recs["cgar@seed42"], 3, n_paths)
    assert "categories" in ad and sum(v["n"] for v in ad["categories"].values()) == 14
    with tempfile.TemporaryDirectory() as d:
        key = ds.key.replace(":", "_")
        for mname, rs in recs.items():
            with open(os.path.join(d, f"records_{key}_{mname}.jsonl"), "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
        A.main(["--results", d, "--dataset", ds.key, "--eval", "synthetic:14:35", "--budget", "3", "--primary", "cgar",
                "--pairs", "cgar:sbert_top5", "cgar:gnnrag_top5", "--n-bootstrap", "100", "--plot", "--quiet"])
        out = os.path.join(d, "analysis")
        R = json.load(open(os.path.join(out, f"analysis_{key}.json")))
        cats = R["adaptivity"]["categories"]; assert cats and sum(v["n"] for v in cats.values()) == 14
        assert set(cats) <= {"controller_stopped_early", "pool_limited", "full_budget", "unknown"} and "unknown" not in cats
        assert R["pairwise"]["cgar:sbert_top5"]["f1"]["n"] == 14
        assert "cgar" in R["selective"] and "sel_hit@50" in R["selective"]["cgar"]
        assert os.path.exists(os.path.join(out, f"selective_{key}.png")) and os.path.exists(os.path.join(out, f"analysis_{key}.md"))


def test_cli_v3_flags_smoke():
    from kgqa_bench.cli import main
    with tempfile.TemporaryDirectory() as d:
        main(["run", "--train", "synthetic:12:0:train", "--eval", "synthetic:6:1", "--methods", "cgar",
              "--embedder", "hash", "--device", "cpu", "--dev-split", "0.2", "--cgar-s1-epochs", "1", "--cgar-s2-steps", "0",
              "--cgar-max-paths", "3", "--cgar-learned-stop", "--cgar-rel-label", "proof", "--cgar-conf-target", "hit",
              "--n-bootstrap", "50", "--quiet", "--out", d])
        R = json.load(open(os.path.join(d, "results.json")))
        assert R["config"]["cgar_learned_stop"] is True and R["config"]["cgar_rel_label"] == "proof"
        assert R["train_logs"]["cgar@seed42"]["loss_kw"]["conf_target"] == "hit"


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("  ✓", name)
            except Exception:
                fails += 1; print("  ✗", name); traceback.print_exc()
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
