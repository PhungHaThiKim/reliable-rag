"""Task B: deterministic generator, KG parsed from documents agrees with the org, gold answers are
recomputable from the documents alone, G4 questions are really undecidable, CLI runs end to end."""
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypi_rag.cli import main  # noqa: E402
from pypi_rag.supplychain import INSUFFICIENT, KGIndex, build_kg, in_spec, make_org, make_questions, render_corpus  # noqa: E402
from pypi_rag.supplychain.qa import normalize_answer, parse_qa  # noqa: E402


def _kg_affected(G, adv):
    """Affected active services computed ONLY from the document-parsed graph (independent of org.py)."""
    a = f"adv:{adv}"
    targets = {v for _u, v, k in G.out_edges(a, keys=True) if k == "MATCHES"}
    out = set()

    def walk2(n, depth, extras_chain):
        # extras on a REQUIRES edge are only active when that edge starts at the service's direct dependency
        for u, _v, k, d in G.in_edges(n, keys=True, data=True):
            if k == "REQUIRES" and depth < 6:
                walk2(u, depth + 1, [d.get("extra", "")] + extras_chain)
            elif k == "DEPENDS_ON":
                a_ = G.nodes[u]
                if d["scope"] != "runtime" or a_.get("status") != "active" or d.get("unpinned"):
                    continue
                first, rest = (extras_chain[0], extras_chain[1:]) if extras_chain else ("", [])
                if first and first not in d.get("extras", []):
                    continue
                if any(rest):
                    continue
                out.add(a_["name"])
    for t in targets:
        walk2(t, 0, [])
    return sorted(out)


def test_generator_is_deterministic():
    d1 = [(d.doc_id, d.text) for d in render_corpus(make_org(5))]
    d2 = [(d.doc_id, d.text) for d in render_corpus(make_org(5))]
    assert d1 == d2
    q1, q2 = make_questions(make_org(5), 30), make_questions(make_org(5), 30)
    assert [q["question"] for q in q1] == [q["question"] for q in q2]
    assert d1 != [(d.doc_id, d.text) for d in render_corpus(make_org(6))]


def test_kg_from_documents_matches_org():
    for seed in (1, 2, 3):
        org = make_org(seed)
        G = build_kg(render_corpus(org))
        assert all(d.get("doc") for *_x, d in G.edges(data=True))
        n_dep = sum(1 for _u, _v, k in G.edges(keys=True) if k == "DEPENDS_ON")
        assert n_dep == sum(len(s.deps) for s in org.services.values())
        for a in org.advisories:
            assert _kg_affected(G, a.adv_id) == org.affected_services(a.adv_id), (seed, a.adv_id)
            assert org.affected_services(a.adv_id), "every advisory reaches at least one service"


def test_questions_gold_and_groups():
    org = make_org(2)
    qs = make_questions(org, 60)
    groups = {g: [q for q in qs if q["group"] == g] for g in ("G1", "G2", "G3", "G4")}
    assert all(len(v) == 60 for v in groups.values())
    assert all(q["answerable"] for g in ("G1", "G2", "G3") for q in groups[g])
    assert all(q["gold"] == INSUFFICIENT for q in groups["G4"])
    assert len({q["question"] + str(q["code"]) for q in qs}) == len(qs)
    for g in ("G2", "G3", "G4"):                         # several question types and traps per group
        assert len({q["qtype"] for q in groups[g]}) >= 3 and len({q["trap"] for q in groups[g]}) >= 3
    for q in groups["G3"]:
        if q["qtype"] == "is_affected":
            adv = re.search(r"ADV-\d{4}-\d{3}", q["question"]).group(0)
            svc = re.search(r"svc-[a-z0-9-]+", q["question"]).group(0)
            assert q["gold"] == ("yes" if org.affected(svc, adv) else "no")


def test_g4_is_not_decidable_from_the_corpus():
    org = make_org(2)
    text = "\n".join(d.text for d in render_corpus(org))
    qs = make_questions(org, 60)
    for q in qs:
        if q["group"] != "G4":
            continue
        if q["trap"] == "unknown_entity":
            ent = re.search(r"(svc-[a-z0-9-]+|ADV-\d{4}-\d{3})", q["question"]).group(0)
            assert ent not in text
        if q["trap"] == "unpinned":
            svc = re.search(r"svc-[a-z0-9-]+", q["question"]).group(0)
            assert "unpinned" in next(d.text for d in render_corpus(org) if d.doc_id == f"svc:{svc}")
        if q["trap"] == "conflicting_policies":
            assert all(t.status == "current" for t in org.token_policies)
        if q["trap"] == "not_documented":
            assert "node" not in text.lower() and "deployed on" not in text.lower()


def test_parse_and_normalize():
    p = parse_qa('ok {"answer": ["svc-a", "SVC-B"], "insufficient_evidence": false, "confidence": 80, '
                 '"claims": [{"text": "x", "evidence": ["G1", "Z9"]}]}', "service_set", ["G1"])
    assert p["answer"] == ["svc-a", "svc-b"] and p["confidence"] == 0.8 and p["claims"][0]["invalid_ids"] == ["Z9"]
    assert parse_qa('{"answer": null, "insufficient_evidence": true}', "yes_no")["insufficient"]
    assert parse_qa('{"answer": "Insufficient evidence", "claims": []}', "yes_no")["insufficient"]
    assert normalize_answer("2 days", "hours") == 48 and normalize_answer("Yes, it is", "yes_no") == "yes"
    assert normalize_answer("acme-a -> acme-b==1.0.0 -> x", "package_path") == ["acme-a", "acme-b", "x"]
    assert in_spec("1.2.1", ">=1.2.1,<1.3.0") and not in_spec("1.3.0", ">=1.2.1,<1.3.0") and in_spec("9.9.9", "*")


def test_kg_evidence_shows_trap_routes_without_filtering():
    org = make_org(2)
    K = KGIndex(build_kg(render_corpus(org)))
    a = org.advisories[0]
    items, diag = K.evidence("q", f"Which active services deploy, at runtime, a version of {a.package} affected by advisory {a.adv_id}?")
    assert diag["n_paths"] >= len(org.affected_services(a.adv_id))
    paths = [i.content for i in items if i.meta["kind"] == "path"]
    for s in org.affected_services(a.adv_id):
        assert any(p.startswith(s + " ") for p in paths), s
    flat = K.flatten("q", items, "k")
    assert flat and all(f.evidence_id.startswith("F") for f in flat)
    unk, d2 = K.evidence("q", f"Is {org.held_out_services[0]} exposed to {org.held_out_advisories[0]}?")
    assert d2["n_paths"] == 0 and "No entity" in unk[0].content


def test_cli_sc_end_to_end_dummy():
    with tempfile.TemporaryDirectory() as d:
        main(["sc-generate", "--seed", "1", "--out", f"{d}/dev", "--n-per-group", "12"])
        main(["sc-generate", "--seed", "2", "--out", f"{d}/test", "--n-per-group", "12"])
        main(["sc-run", "--data", f"{d}/dev", "--out", f"{d}/rd", "--llm", "dummy", "--n-bootstrap", "50"])
        main(["sc-tune", "--results", f"{d}/rd", "--data", f"{d}/dev", "--out", f"{d}/pol.json"])
        main(["sc-run", "--data", f"{d}/test", "--out", f"{d}/rt", "--llm", "dummy", "--policy", f"{d}/pol.json",
              "--n-bootstrap", "50"])
        R = json.load(open(f"{d}/rt/results.json"))
        assert set(R["summaries"]) == {"e0", "e1", "e2", "e2_bm25", "e3", "e3a_exact", "e3a", "e4"}
        for m, s in R["summaries"].items():
            assert "all" not in s, "no pooled accuracy over answerable and unanswerable questions"
            assert s["grounding_all"]["n"] == 48 and all(s[g]["n"] == 12 for g in ("G1", "G2", "G3", "G4"))
            assert "correct_abstention_rate" in s["G4"] and "exact_accuracy" not in s["G4"]
            assert {"exact_accuracy", "set_f1", "answer_coverage", "aurc"} <= set(s["answerable"])
        assert "e3_vs_e3a_exact/G3" in R["comparisons"] and not any(k.endswith("/all") for k in R["comparisons"])
        assert R["comparisons"]["e4_vs_e3/G4"]["metric"] == "correct_abstention"
        # E3 and E4 share the original claims: same denominator; E4 cannot keep more unsupported claims than E3
        e3, e4 = R["summaries"]["e3"]["grounding_all"], R["summaries"]["e4"]["grounding_all"]
        assert e4["unsupported_claim_rate"] == e3["unsupported_claim_rate"]
        assert e4["unsupported_over_original_claims"] <= e3["unsupported_over_original_claims"]
        assert R["policy_sweep"] and R["cost"]["methods"]["e4"]["reuses_e3_generation"]
        pol = json.load(open(f"{d}/pol.json"))
        assert pol["objective"].startswith("0.5*answerable_exact")
        ids = [json.loads(x)["qid"] for x in open(f"{d}/rt/predictions_e0.jsonl", encoding="utf-8")]
        assert ids == [json.loads(x)["qid"] for x in open(f"{d}/rt/predictions_e3.jsonl", encoding="utf-8")]
        main(["sc-rescore", "--results", f"{d}/rt", "--data", f"{d}/test", "--n-bootstrap", "50", "--gpu-price-hour", "1.09"])
        R2 = json.load(open(f"{d}/rt/results.json"))
        assert R2["summaries"]["e3"]["G1"]["exact_accuracy"] == R["summaries"]["e3"]["G1"]["exact_accuracy"]
        assert R2["cost"]["price_per_hour"] == 1.09
