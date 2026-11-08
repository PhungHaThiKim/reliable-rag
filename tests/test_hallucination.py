"""Tests for kgqa_bench.hallucination (CPU, synthetic data, no LLM).

Run:  cd seminar2_kgrag && python -m pytest -q tests/test_hallucination.py
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from kgqa_bench.data import BenchDataset  # noqa: E402
from kgqa_bench.methods import GNNRAGMethod  # noqa: E402
from kgqa_bench.llm_reader import DummyReader, attach_llm_answers, build_prompt, GROUNDED_INSTRUCTION  # noqa: E402
from kgqa_bench import hallucination as H  # noqa: E402


def _records(ds, k=None):
    recs = GNNRAGMethod(k).run(ds, verbose=False)
    attach_llm_answers(recs, ds, DummyReader(), verbose=False)
    return recs


def test_parse_path_text_roundtrip():
    from grl_rag import path_to_string
    tr = [("a", "r1", "b"), ("b", "r2", "c")]
    assert H.parse_path_text(path_to_string(tr)) == tr
    assert H.parse_path_text("a") == []


def test_oracle_reader_never_hallucinates():
    ds = BenchDataset.synthetic(12, seed=11).annotate_hops()
    recs = _records(ds, 3)
    vs = H.evaluate_records(ds, recs)
    assert len(vs) == len(recs)
    # DummyReader only emits path tails -> every claim is a path endpoint, none unsupported
    for v in vs:
        assert v["n_unsupported"] == 0 and v["n_contradicted"] == 0 and v["hallucinated"] == 0
        for cl in v["claims"]:
            assert cl["evidence_type"] == "path_endpoint"
        # gold endpoint -> SUPPORTED; correct answers are always grounded here
        if v["llm_hit"]:
            assert v["grounded_correct"] == 1 and v["correct_ungrounded"] == 0
    s = H.summarize_verdicts(vs)
    assert s["question_hallucination_rate"] == 0.0 and s["claim_hallucination_rate"] == 0.0
    assert s["graph_proof_support_rate"] == 1.0
    assert "aurc_hallucination" in s and 0.0 <= s["aurc_hallucination"] <= 1.0
    assert s["grounded_correct_rate"] <= s["llm_hit"]


def test_claim_labels_absent_intermediate_and_abstention():
    ds = BenchDataset.synthetic(6, seed=12).annotate_hops()
    recs = _records(ds, 3)
    # pick a record with a >=2-hop path so there is an intermediate node
    rec = next(r for r in recs if any(len(H.parse_path_text(t)) >= 2 for t in r["selected_paths"]))
    q = next(x for x in ds if x.qid == rec["qid"])
    path = next(H.parse_path_text(t) for t in rec["selected_paths"] if len(H.parse_path_text(t)) >= 2)
    intermediate = path[0][2]
    # 1) fabricated entity not in the subgraph
    r1 = dict(rec, llm_pred="Berlin_fabricated_entity", llm_hit=0)
    v1 = H.verify_question(q, r1)
    assert v1["n_claims"] == 1 and v1["claims"][0]["label"] == H.UNSUPPORTED
    assert v1["claims"][0]["evidence_type"] == "absent" and v1["hallucinated"] == 1
    # 2) entity present in context only as intermediate node -> no proof -> UNSUPPORTED
    r2 = dict(rec, llm_pred=intermediate, llm_hit=0)
    v2 = H.verify_question(q, r2)
    assert v2["claims"][0]["evidence_type"] == "in_context_no_proof" and v2["claims"][0]["label"] == H.UNSUPPORTED
    assert v2["claims"][0]["legacy_in_context"] is True       # the old proxy would have called it supported
    assert v2["error_category"] in ("wrong_relation_evidence", "path_selection_failure", "candidate_omission",
                                    "missing_kg_evidence")
    # 3) entity in the subgraph but not retrieved
    not_ctx = next(e for h, _r, t in q.triples for e in (h, t)
                   if e not in {x for p in map(H.parse_path_text, rec["selected_paths"]) for hh, _rr, tt in p for x in (hh, tt)})
    v3 = H.verify_question(q, dict(rec, llm_pred=not_ctx, llm_hit=0))
    assert v3["claims"][0]["evidence_type"] == "in_subgraph_only"
    # 4) abstention
    v4 = H.verify_question(q, dict(rec, llm_pred="I cannot determine the answer from the given paths.", llm_hit=0))
    assert v4["abstained"] == 1 and v4["answered"] == 0 and v4["n_claims"] == 0 and v4["error_category"] == "abstained"
    # 5) correct-but-ungrounded: gold answered together with a fabricated entity
    gold = q.answers[0]
    v5 = H.verify_question(q, dict(rec, llm_pred=f"{gold}\nBerlin_fabricated_entity", llm_hit=1))
    assert v5["llm_hit"] == 1 and v5["hallucinated"] == 1 and v5["grounded_correct"] == 0 and v5["correct_ungrounded"] == 1
    assert v5["error_category"] == "correct_ungrounded"


def test_judge_hook_refines_uncertain():
    ds = BenchDataset.synthetic(8, seed=13).annotate_hops()
    recs = _records(ds, 3)
    base = H.evaluate_records(ds, recs)
    n_unc = sum(v["n_uncertain"] for v in base)
    assert n_unc > 0, "synthetic non-gold tails should yield UNCERTAIN claims"
    contra = H.evaluate_records(ds, recs, judge=lambda q, paths, c: H.CONTRADICTED)
    assert sum(v["n_contradicted"] for v in contra) == n_unc and sum(v["n_uncertain"] for v in contra) == 0
    s = H.summarize_verdicts(contra)
    assert s["contradiction_rate"] > 0 and s["question_hallucination_rate"] > 0


def test_compare_and_review_and_cli(tmp_path=None):
    ds = BenchDataset.synthetic(10, seed=14, name="synthetic14").annotate_hops()   # same key as load_spec("synthetic:10:14")
    ra, rb = _records(ds, 5), _records(ds, 2)
    va, vb = H.evaluate_records(ds, ra), H.evaluate_records(ds, rb)
    cmp = H.compare_methods(va, vb, n_bootstrap=200)
    assert cmp["hallucinated"]["n"] == 10 and "p_value" in cmp["hallucinated"]["bootstrap"]
    rows = H.review_sample({"gnnrag_top5": va, "gnnrag_top2": vb}, ds, n=6, records_by_method={"gnnrag_top5": ra, "gnnrag_top2": rb})
    assert 0 < len(rows) <= 6 and {"qid", "method", "reason", "human_hallucinated"} <= set(rows[0])
    with tempfile.TemporaryDirectory() as d:
        key = ds.key.replace(":", "_")
        for m, rs in (("gnnrag_top5", ra), ("gnnrag_top2", rb)):
            with open(os.path.join(d, f"records_{key}_{m}.jsonl"), "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
        H.main(["--results", d, "--eval", "synthetic:10:14", "--ref", "gnnrag_top5",
                "--review-sample", "5", "--n-bootstrap", "100", "--quiet"])
        out = os.path.join(d, "hallucination")
        R = json.load(open(os.path.join(out, "hallucination.json")))
        dsk = list(R["datasets"])[0]
        assert set(R["datasets"][dsk]["summaries"]) == {"gnnrag_top5", "gnnrag_top2"}
        assert "gnnrag_top2" in R["datasets"][dsk]["comparisons"]
        assert os.path.exists(os.path.join(out, f"report_{key}.md"))
        assert os.path.exists(os.path.join(out, f"review_sample_{key}.csv"))
        assert os.path.exists(os.path.join(out, f"claims_{key}_gnnrag_top5.jsonl"))
        rep = open(os.path.join(out, f"report_{key}.md")).read()
        assert "## A. Faithfulness" in rep and "## C. Risk" in rep and "Error taxonomy" in rep


def test_grounded_prompt_variant():
    p = build_prompt("who?", ["a -> r -> b"], "grounded")
    assert GROUNDED_INSTRUCTION in p and "unknown" in p and "a -> r -> b" in p
    assert H.is_abstention("unknown") and not H.is_abstention("Paris")


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
