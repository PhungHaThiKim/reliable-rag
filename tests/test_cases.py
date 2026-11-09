"""Controlled cases C01-C06 (test-case document): graph evidence, E3a = exactly E3's edges, verification of
negated claims under dynamic dispatch, required evidence lines, case metrics and the CLI on the cases."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypi_rag.analysis import analyze_package  # noqa: E402
from pypi_rag.cli import main  # noqa: E402
from pypi_rag.data.cases import CASES, make_cases, required_evidence  # noqa: E402
from pypi_rag.evaluation.evidence import answer_path_ok, required_evidence_recall  # noqa: E402
from pypi_rag.evaluation.statistical_tests import paired_bootstrap  # noqa: E402
from pypi_rag.graph import build_behavior_graph, flatten_paths, graph_evidence  # noqa: E402
from pypi_rag.schema import Evidence  # noqa: E402
from pypi_rag.verification import apply_policy, check_claims  # noqa: E402


def _case(cid):
    s = next(x for x in make_cases() if x.meta["case"] == cid)
    p = s.public_view()
    f = analyze_package(p)
    G = build_behavior_graph(p, f)
    items, _ = graph_evidence(G, p)
    return s, p, f, G, items


def _patterns(items):
    return {x for i in items if i.source_type == "graph_path" for x in (i.meta.get("patterns") or [])}


def test_required_evidence_lines_match_markers():
    for c in CASES:
        req = required_evidence(c["files"], c["markers"])
        for (path, marker), loc in zip(c["markers"], req):
            ln = int(loc.rsplit(":", 1)[1])
            assert marker in c["files"][path].splitlines()[ln - 1]


def test_c05_cross_file_path_with_three_functions():
    s, p, f, G, items = _case("C05")
    gp = [i for i in items if i.source_type == "graph_path" and "P03" in (i.meta.get("patterns") or [])]
    assert gp, "C05 must yield a P03 exfiltration path"
    vf = gp[0].meta.get("via_functions") or []
    for fn in s.gold["gold_function_path"]:
        assert any(fn in v for v in vf), (fn, vf)
    # every call site in main.py and both endpoint lines are shown; the transform line inside
    # prepare_payload (processor.py:2) is not a graph span: a measured gap, reported as 5/6
    shown = [r for r in s.gold["required_evidence"] if required_evidence_recall([r], items) == 1.0]
    assert set(s.gold["required_evidence"]) - set(shown) == {"demo_package/processor.py:2"}


def test_c04_only_local_file_upload_pattern():
    _s, _p, _f, _G, items = _case("C04")
    assert _patterns(items) == {"P04"}


def test_e3a_exact_uses_exactly_e3_edges():
    for cid in ("C03", "C05"):
        _s, p, _f, G, items = _case(cid)
        gitems = [i for i in items if i.source_type == "graph_path"]
        e3_edges = set()
        for i in gitems:
            if "|" in (i.source_ref or ""):
                ops = i.source_ref.split("|")
                e3_edges.update(zip(ops, ops[1:]))
        flat = flatten_paths(G, p, gitems, seed_key="t")
        f_edges = {tuple(x.meta["edge"]) for x in flat if x.meta.get("edge")}
        assert f_edges == e3_edges and e3_edges
        assert flatten_paths(G, p, gitems, seed_key="t")[0].content == flat[0].content   # seeded order


def test_c06_negated_claim_unsupported_and_abstains():
    s, p, f, G, items = _case("C06")
    assert f.dispatch_sites and not f.complete()
    vis = {i.evidence_id: i for i in items}
    gid = items[0].evidence_id
    claims = [{"text": "no credentials are sent", "behavior": "network_send", "evidence": [gid], "negated": True}]
    chk = check_claims(claims, vis, f, p)
    assert chk["claims"][0]["label"] == "UNSUPPORTED" and chk["dispatch_unresolved"]
    assert apply_policy("benign", chk, vis, "abstain", abstain_on_dispatch=True)["verdict"] == "uncertain"
    assert apply_policy("benign", chk, vis, "abstain", abstain_on_dispatch=False)["verdict"] == "benign"


def test_answer_path_metric():
    chk = [{"label": "SUPPORTED", "negated": False, "behaviors": ["data_exfiltration"]}]
    gold = ["load_secret", "prepare_payload", "submit"]
    txt = "run() calls load_secret, then prepare_payload, then submit sends it"
    assert answer_path_ok(gold, txt, [], chk, ["data_exfiltration"]) == 1.0
    assert answer_path_ok(gold, "load_secret then submit", [], chk, ["data_exfiltration"]) == 0.0
    assert answer_path_ok([], txt, [], chk, ["data_exfiltration"]) is None
    ev = Evidence("G1", "x", "graph_path", "a|b", "...", meta={"spans": [["m.py", 3]], "call_sites": ["f()@main.py:6"]})
    assert required_evidence_recall(["m.py:3", "main.py:6", "main.py:7"], [ev]) == 2 / 3


def test_group_bootstrap_resamples_groups():
    a = [1, 1, 1, 1, 0, 0]; b = [0, 0, 0, 0, 0, 0]
    r = paired_bootstrap(a, b, n_bootstrap=500, groups=["g1", "g1", "g1", "g1", "g2", "g3"])
    assert r["resample_unit"] == "group" and r["n_groups"] == 3
    assert paired_bootstrap(a, b, n_bootstrap=200)["resample_unit"] == "sample"


def test_cli_on_cases_dummy():
    with tempfile.TemporaryDirectory() as d:
        main(["prepare", "--spec", "cases:", "--strategy", "keep", "--out", f"{d}/data"])
        main(["build-kb", "--data", f"{d}/data", "--out", f"{d}/kb.jsonl", "--sources", "builtin", "case_rules", "--workers", "1"])
        main(["run", "--data", f"{d}/data", "--kb", f"{d}/kb.jsonl", "--methods", "e0", "e2", "e3", "e3a_exact", "e4",
              "--llm", "dummy", "--embedder", "hash", "--out", f"{d}/run", "--use-sample-question", "--strata-key", "case",
              "--n-bootstrap", "50", "--workers", "1", "--quiet"])
        R = json.load(open(f"{d}/run/results.json"))
        assert R["n_samples"] == 6
        assert {r["case"] for r in R["cases"]} == {f"C0{i}" for i in range(1, 7)}
        assert "case/abstention_accuracy" in R["summaries"]["e3"]
        assert "e3_vs_e3a_exact" in R["comparisons"]
        g = R["grounding_vs_e3"]                         # E4 scored on E3's original claims and all items
        assert g["e4"]["original_claims"] == g["e3"]["original_claims"] and g["e4"]["n_items"] == 6
        assert g["e4"]["unsupported_over_original_claims"] <= g["e3"]["unsupported_over_original_claims"]
        assert "evidence/unsupported_over_original_claims" in R["summaries"]["e4"] and R["cost"]["methods"]["e4"]["reuses_e3_generation"]
        kb = [json.loads(x) for x in open(f"{d}/kb.jsonl", encoding="utf-8")]
        assert any(x["doc_id"] == "case:PATTERN-EXFIL-001" for x in kb)


def test_flatten_paths_every_synthetic_package():
    from pypi_rag.data.synthetic import make_synthetic
    for s in make_synthetic(80, 5):
        p = s.public_view()
        G = build_behavior_graph(p, analyze_package(p))
        items, _ = graph_evidence(G, p)
        flat = flatten_paths(G, p, items, seed_key="k")
        assert flat or not items
