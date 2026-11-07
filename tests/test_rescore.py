"""Tests for kgqa_bench.rescore (CPU, synthetic, dummy reader).

Run:  cd colab/graph_rag && python -m pytest -q tests/test_rescore.py
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from kgqa_bench import cli, rescore as RS  # noqa: E402


def test_rescore_dir_abstention_and_parser():
    with tempfile.TemporaryDirectory() as d:
        cli.main(["run", "--eval", "synthetic:12:21", "--methods", "gnnrag_all", "gnnrag_top2",
                  "--reader", "dummy", "--embedder", "hash", "--device", "cpu", "--compare-ref", "gnnrag_all", "--out", d, "--n-bootstrap", "100", "--quiet"])
        key = "synthetic21_test"
        p = os.path.join(d, f"records_{key}_gnnrag_top2.jsonl")
        assert os.path.exists(p), os.listdir(d)
        recs = [json.loads(l) for l in open(p)]
        # 1) an abstention: legacy proxy scored "unknown" as unsupported, rescore must not
        recs[0]["llm_pred"] = "unknown"; recs[0]["llm_unsupported_rate"] = 1.0; recs[0]["llm_hit"] = 0
        # 2) a markdown-wrapped gold answer that the old parser mangled
        gold = None
        for r in recs[1:]:
            tails = [t.split("->")[-1].strip() for t in r["selected_paths"] if "->" in t]
            if tails:
                gold = tails[0]; r["llm_pred"] = f"The answer is:\n1. **{gold}**"; r["llm_hit1"] = 0; break
        assert gold is not None
        with open(p, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        ch = RS.rescore_dir(d, n_bootstrap=100, verbose=False)
        assert os.path.exists(p + ".orig")
        new = [json.loads(l) for l in open(p)]
        assert new[0]["llm_abstained"] == 1 and new[0]["llm_unsupported_rate"] == 0.0 and new[0]["llm_hit"] == 0
        assert new[0]["llm_unsupported_rate_legacy"] == 1.0
        r2 = next(r for r in new if r["llm_pred"].startswith("The answer is:"))
        assert r2["llm_abstained"] == 0 and r2["llm_unsupported_rate"] == 0.0
        R = json.load(open(os.path.join(d, "results.json")))
        s = R["summaries"]["synthetic21:test"]["gnnrag_top2"]
        assert s["llm_abstained"] > 0 and R["config"]["rescored"] is True
        assert "gnnrag_top2_vs_gnnrag_all:llm_hit1" in R["comparisons"]["synthetic21:test"]
        assert "gnnrag_top2" in ch["synthetic21:test"]
        # idempotent: second pass does not clobber the legacy column
        RS.rescore_dir(d, n_bootstrap=50, verbose=False)
        again = [json.loads(l) for l in open(p)]
        assert again[0]["llm_unsupported_rate_legacy"] == 1.0


if __name__ == "__main__":
    test_rescore_dir_abstention_and_parser(); print("ok")
