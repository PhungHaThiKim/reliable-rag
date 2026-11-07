"""Tests for kgqa_bench.escalation (CPU, synthetic)."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from kgqa_bench.data import BenchDataset  # noqa: E402
from kgqa_bench.methods import GNNRAGMethod  # noqa: E402
from kgqa_bench.llm_reader import DummyReader, attach_llm_answers  # noqa: E402
from kgqa_bench import escalation as E  # noqa: E402


def _recs(ds, k, conf_seed=0):
    import random
    rng = random.Random(conf_seed)
    recs = GNNRAGMethod(k).run(ds, verbose=False)
    for r in recs:                      # synthetic confidence correlated with f1 (+noise)
        r["confidence"] = min(1.0, max(0.0, r["f1"] + rng.gauss(0, 0.2)))
    return recs


def test_curve_endpoints_and_oracle():
    ds = BenchDataset.synthetic(30, seed=21).annotate_hops()
    P, F = _recs(ds, 2), GNNRAGMethod(None).run(ds, verbose=False)
    rows = E.escalation_curve(P, F)
    assert rows[0]["tau"] == "-inf" and rows[0]["escalated_frac"] == 0.0
    allfb = [r for r in rows if r["tau"] == "+inf"][0]
    assert allfb["escalated_frac"] == 1.0
    # endpoints reproduce the pure methods
    import numpy as np
    assert abs(rows[0]["f1"] - np.mean([r["f1"] for r in P])) < 1e-9
    assert abs(allfb["f1"] - np.mean([r["f1"] for r in F])) < 1e-9
    orc = [r for r in rows if r["tau"] == "oracle"][0]
    assert orc["f1"] >= max(rows[0]["f1"], allfb["f1"]) - 1e-9     # oracle gate is an upper bound
    # cost is monotone non-decreasing in escalated fraction (fallback = superset)
    real = [r for r in rows if r["tau"] != "oracle"]
    for a, b in zip(real, real[1:]):
        assert b["cost"] >= a["cost"] - 1e-9
    pts = E.pick_operating_points(rows, rows[0], allfb)
    assert "match_fallback_quality" in pts


def test_llm_columns_and_cli():
    ds = BenchDataset.synthetic(12, seed=22, name="synthetic22").annotate_hops()
    P, F = _recs(ds, 2), GNNRAGMethod(None).run(ds, verbose=False)
    attach_llm_answers(P, ds, DummyReader(), verbose=False); attach_llm_answers(F, ds, DummyReader(), verbose=False)
    rows = E.escalation_curve(P, F, quality="llm_hit")
    assert "llm_hit" in rows[0] and "llm_unsupported_rate" in rows[0]
    with tempfile.TemporaryDirectory() as d:
        key = ds.key.replace(":", "_")
        for m, rs in (("cgar@seed42", P), ("gnnrag_all", F), ("gnnrag_top3", GNNRAGMethod(3).run(ds, verbose=False))):
            with open(os.path.join(d, f"records_{key}_{m}.jsonl"), "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
        E.main(["--results", d, "--dataset", ds.key, "--primary", "cgar", "--fallback", "gnnrag_all", "--plot", "--quiet"])
        out = os.path.join(d, "escalation")
        files = os.listdir(out)
        assert any(f.endswith(".json") for f in files) and any(f.endswith(".md") for f in files) and any(f.endswith(".png") for f in files)
        R = json.load(open(os.path.join(out, [f for f in files if f.endswith(".json")][0])))
        assert any(d_["method"] == "gnnrag_top3" for d_ in R["references"])
        assert R["primary"] == "cgar" and R["rows"][-1]["tau"] == "oracle"


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("  ok", name)
            except Exception:
                fails += 1; print("  FAIL", name); traceback.print_exc()
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
