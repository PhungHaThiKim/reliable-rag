"""kgqa_bench.snapshot: the demo snapshot carries summaries for every reader dir, per-question files only for the
requested readers/methods, and questions.json for the copied qids. CPU, synthetic data.

Run:  cd seminar2_kgrag && python -m pytest -q tests/test_snapshot.py
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from kgqa_bench import cli, hallucination as H  # noqa: E402
from kgqa_bench.snapshot import make_snapshot  # noqa: E402


def test_snapshot_layout():
    with tempfile.TemporaryDirectory() as root:
        run = f"{root}/run"
        cli.main(["run", "--eval", "synthetic:8:45", "--methods", "gnnrag_all", "sbert_top5", "--embedder", "hash",
                  "--device", "cpu", "--reader", "dummy", "--no-retrieval-baseline", "--n-bootstrap", "50", "--out", run, "--quiet"])
        H.run(run, ["synthetic:8:45"], methods=["gnnrag_all", "sbert_top5"], ref="gnnrag_all", n_bootstrap=50, verbose=False)
        res = f"{root}/results"
        for name in ("b08_reader_Qwen2.5-7B-Instruct", "b08_reader_Meta-Llama-3.1-8B-Instruct", "b08_reader_Qwen2.5-7B-Instruct_old"):
            shutil.copytree(run, f"{res}/{name}")
        out = f"{root}/snap"
        man = make_snapshot(res, "synthetic:8:45", out, readers=["Qwen2.5-7B-Instruct"], methods=["gnnrag_all", "no_retrieval"], verbose=False)
        assert man["reader_dirs"] == ["b08_reader_Meta-Llama-3.1-8B-Instruct", "b08_reader_Qwen2.5-7B-Instruct"]   # *_old skipped
        q = f"{out}/b08_reader_Qwen2.5-7B-Instruct"; l = f"{out}/b08_reader_Meta-Llama-3.1-8B-Instruct"
        assert os.path.exists(f"{q}/results.json") and os.path.exists(f"{q}/hallucination/hallucination.json")
        assert os.path.exists(f"{q}/records_synthetic45_test_gnnrag_all.jsonl") and os.path.exists(f"{q}/records_synthetic45_test_no_retrieval.jsonl")
        assert os.path.exists(f"{q}/hallucination/claims_synthetic45_test_gnnrag_all.jsonl")
        assert not os.path.exists(f"{q}/records_synthetic45_test_sbert_top5.jsonl")             # method not requested
        assert os.path.exists(f"{l}/results.json") and not os.path.exists(f"{l}/records_synthetic45_test_gnnrag_all.jsonl")
        qs = json.load(open(f"{out}/questions.json"))
        assert len(qs) == man["n_questions"] > 0 and all(v["question"] and v["answers"] for v in qs.values())
        assert json.load(open(f"{out}/MANIFEST.json"))["n_files"] == man["n_files"]


if __name__ == "__main__":
    test_snapshot_layout(); print("ok")
