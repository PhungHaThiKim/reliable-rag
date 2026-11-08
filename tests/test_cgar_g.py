"""CGAR-G (Nov 2026): escalating retriever, candidate prompt, answer grounding, offline verify,
SFT example builder. CPU, synthetic data, DummyReader.

Run:  cd seminar2_kgrag && python -m pytest -q tests/test_cgar_g.py
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from kgqa_bench.data import BenchDataset  # noqa: E402
from kgqa_bench.methods import CGAREscalateMethod, CGARMethod, GNNRAGMethod  # noqa: E402
from kgqa_bench.llm_reader import (DummyReader, attach_llm_answers, build_prompt, candidates_of,  # noqa: E402
                                   ground_answer, CAND_INSTRUCTION)
from kgqa_bench import rescore as RS  # noqa: E402
from kgqa_bench import hallucination as H  # noqa: E402
from kgqa_bench.reader_ft import build_sft_examples, gold_in_candidates  # noqa: E402
from kgqa_bench import cli  # noqa: E402
from grl_rag import Embedder  # noqa: E402


def _cgar(ds):
    m = CGARMethod(Embedder(fallback="hash"), device="cpu", hidden_dim=32, max_paths=2)
    m.controller.eval()
    return m


def test_records_carry_candidates():
    ds = BenchDataset.synthetic(6, seed=41).annotate_hops()
    recs = GNNRAGMethod(None).run(ds, verbose=False)
    for r in recs:
        tails = [t.split(" -> ")[-1] for t in r["selected_paths"]]
        assert r["candidates"] == list(dict.fromkeys(tails))
        assert candidates_of({"selected_paths": r["selected_paths"]}) == r["candidates"]


def test_escalation_rule_and_fallback_all():
    ds = BenchDataset.synthetic(10, seed=42).annotate_hops()
    m = _cgar(ds)
    base = {r["qid"]: r for r in m.run(ds, verbose=False)}
    allr = {r["qid"]: r for r in GNNRAGMethod(None).run(ds, verbose=False)}
    # synthetic questions share one pool size, so split on confidence (median) and check the n_cand rule at both ends
    confs = sorted(r["confidence"] for r in base.values())
    tau = confs[len(confs) // 2]
    me = CGAREscalateMethod(m, n_cand=None, tau=tau, fallback="all")
    recs = me.run(ds, verbose=False)
    assert len(recs) == len(ds) and {0, 1} >= {r["escalated"] for r in recs}
    assert any(r["escalated"] for r in recs) and any(not r["escalated"] for r in recs)
    n_max = max(len(q.top_candidates) for q in ds)
    assert all(r["escalated"] for r in CGAREscalateMethod(m, n_cand=0, fallback="all").run(ds, verbose=False))
    assert not any(r["escalated"] for r in CGAREscalateMethod(m, n_cand=n_max, fallback="all").run(ds, verbose=False))
    for r in recs:
        assert r["escalated"] == int(base[r["qid"]]["confidence"] < tau)
        assert r["confidence"] == base[r["qid"]]["confidence"]           # confidence stays CGAR's
        if r["escalated"]:
            assert set(r["selected_paths"]) == set(allr[r["qid"]]["selected_paths"])   # = gnnrag_all's record
        else:
            assert r["selected_paths"] == base[r["qid"]]["selected_paths"]
        assert r["cost"] == len(r["selected_paths"]) and r["cost_primary"] == base[r["qid"]]["cost"]
    # tau rule: tau above every confidence -> everything escalates
    me2 = CGAREscalateMethod(m, n_cand=None, tau=10.0, fallback="all")
    assert all(r["escalated"] for r in me2.run(ds, verbose=False))
    # budget fallback: more paths than the primary budget where available, never fewer
    me3 = CGAREscalateMethod(m, n_cand=None, tau=10.0, fallback="budget:4")
    for r in me3.run(ds, verbose=False):
        assert r["cost"] >= r["cost_primary"]
    assert m.controller.max_paths == 2                                   # restored


def test_cand_prompt_lists_candidates():
    p = build_prompt("who?", ["a -> r -> b", "a -> r2 -> c -> r3 -> d"], "cand")
    assert CAND_INSTRUCTION in p and "Candidate answers:\n- b\n- d" in p and p.rstrip().endswith("who??") is False
    p2 = build_prompt("who?", ["a -> r -> b"], "cand", candidates=["X", "Y"])
    assert "- X\n- Y" in p2 and "- b" not in p2
    assert "Candidate answers" not in build_prompt("who?", ["a -> r -> b"], "chat")


def test_ground_answer_rules():
    c = ["Barack Obama", "Michelle Obama", "Chicago"]
    g = ground_answer("Here is the answer:\n* Barack Obama\n* Berlin_fabricated", c)
    assert g["kept"] == ["Barack Obama"] and g["dropped"] == ["Berlin_fabricated"] and g["fallback"] == 0
    assert g["pred"] == "Barack Obama"
    g2 = ground_answer("Berlin_fabricated", c)                           # answered, nothing grounded -> first candidate
    assert g2["kept"] == ["Barack Obama"] and g2["fallback"] == 1
    assert ground_answer("Berlin_fabricated", c, fallback=False)["kept"] == []
    g3 = ground_answer("unknown", c)                                     # abstention stays an abstention
    assert g3["abstained"] == 1 and g3["kept"] == [] and g3["fallback"] == 0
    g4 = ground_answer("obama", c)                                       # alias / containment -> candidate spelling
    assert g4["kept"] and all(k in c for k in g4["kept"])
    g5 = ground_answer("Paris", [])                                      # no candidates: cannot ground
    assert g5["grounded"] == 0 and g5["kept"] == ["Paris"]


def test_attach_with_verify_and_offline_rescore_roundtrip():
    ds = BenchDataset.synthetic(8, seed=43).annotate_hops()
    recs = GNNRAGMethod(2).run(ds, verbose=False)

    class NoisyReader(DummyReader):
        prompt_type = "cand"

        def answer(self, question, paths_text, candidates=None):
            base = super().answer(question, paths_text, candidates)
            return "Based on the paths, the answers are:\n" + base + "\nBerlin_fabricated_entity"

    attach_llm_answers(recs, ds, NoisyReader(), verbose=False, verify=True)
    for r in recs:
        assert "Berlin_fabricated_entity" in r["llm_pred_raw"] and "Berlin_fabricated_entity" not in r["llm_pred"]
        assert r["llm_n_dropped"] >= 1 and r["llm_grounded"] == 1 and r["llm_unsupported_rate"] == 0.0
    hit_verified = sum(r["llm_hit"] for r in recs)
    # offline: rescore without verify restores raw scoring; with verify reproduces the same numbers
    with tempfile.TemporaryDirectory() as d:
        cli.main(["run", "--eval", "synthetic:8:43", "--methods", "gnnrag_top2", "--reader", "dummy",
                  "--embedder", "hash", "--device", "cpu", "--out", d, "--n-bootstrap", "50", "--quiet"])
        path = f"{d}/records_synthetic43_test_gnnrag_top2.jsonl"
        rows = [json.loads(l) for l in open(path)]
        for r in rows:                                                       # simulate a verbose reader output
            r["llm_pred"] = r["llm_pred_raw"] = "Based on the paths, the answers are:\n" + r["llm_pred"] + "\nBerlin_fabricated_entity"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        RS.rescore_dir(d, verbose=False, n_bootstrap=50)                     # raw scoring
        raw = [json.loads(l) for l in open(path)]
        assert all("Berlin_fabricated_entity" in r["llm_pred"] and r["llm_unsupported_rate"] > 0 for r in raw)
        RS.rescore_dir(d, verbose=False, n_bootstrap=50, verify=True)        # grounded scoring
        ver = [json.loads(l) for l in open(path)]
        assert all("Berlin_fabricated_entity" not in r["llm_pred"] and r["llm_unsupported_rate"] == 0.0 for r in ver)
        assert sum(r["llm_hit"] for r in ver) == hit_verified
        R = json.load(open(f"{d}/results.json"))
        s = R["summaries"]["synthetic43:test"]["gnnrag_top2"]
        assert R["config"]["reader_verify"] is True and s["llm_n_dropped"] >= 1 and s["llm_unsupported_rate"] == 0.0
        RS.rescore_dir(d, verbose=False, n_bootstrap=50)                     # back to raw, extra keys gone
        R = json.load(open(f"{d}/results.json"))
        assert "llm_n_dropped" not in R["summaries"]["synthetic43:test"]["gnnrag_top2"]
        # verifier sees the grounded answer and keeps the raw text for review
        RS.rescore_dir(d, verbose=False, n_bootstrap=50, verify=True)
        vs = H.evaluate_records(ds, [json.loads(l) for l in open(path)])
        assert all(v["n_unsupported"] == 0 for v in vs) and all(v["llm_pred_raw"] for v in vs)


def test_cli_cgar_esc_and_reader_cand_verify():
    with tempfile.TemporaryDirectory() as d:
        cli.main(["run", "--train", "synthetic:16:44", "--eval", "synthetic:8:45", "--methods", "gnnrag_all", "cgar", "cgar_esc",
                  "--embedder", "hash", "--device", "cpu", "--out", d, "--n-bootstrap", "50", "--quiet",
                  "--cgar-s1-epochs", "1", "--cgar-s2-steps", "0", "--cgar-max-paths", "2", "--dev-max", "8",
                  "--cgar-esc-ncand", "0", "--cgar-esc-frac", "0.5", "--reader", "dummy", "--reader-verify",
                  "--compare-ref", "gnnrag_all"])
        R = json.load(open(f"{d}/results.json"))
        s = R["summaries"]["synthetic45:test"]
        assert {"gnnrag_all", "cgar", "cgar_esc"} <= set(s)
        assert s["cgar_esc"]["cost"] >= s["cgar"]["cost"] and s["cgar_esc"]["llm_grounded"] == 1.0
        assert "cgar_esc_vs_gnnrag_all:llm_hit" in R["comparisons"]["synthetic45:test"]
        log = R["train_logs"]["cgar_esc_tau@seed42"]
        assert log["from"] == "dev" and log["frac"] == 0.5 and log["tau"] is not None
        # tau comes from the DEV split, so on 8 eval questions the escalated share may be 0 or 1 (it depends on
        # PYTHONHASHSEED through the synthetic data); what must always hold is the rule itself (n_cand rule off):
        esc = [json.loads(l) for l in open(f"{d}/records_synthetic45_test_cgar_esc.jsonl")]
        assert esc and all(r["escalated"] == int(r["confidence"] < log["tau"]) for r in esc)
        assert abs(s["cgar_esc"]["escalated"] - sum(r["escalated"] for r in esc) / len(esc)) < 1e-9
        assert os.path.exists(f"{d}/records_synthetic45_test_cgar_esc.jsonl")


def test_sft_examples():
    ds = BenchDataset.synthetic(10, seed=46).annotate_hops()
    recs = GNNRAGMethod(None).run(ds, verbose=False) + GNNRAGMethod(1).run(ds, verbose=False)
    ex = build_sft_examples(recs, ds, prompt_type="cand")
    assert ex and all(e["target"] and CAND_INSTRUCTION in e["prompt"] for e in ex)
    for e in ex:
        q = next(x for x in ds if x.qid == e["qid"])
        for t in e["target"].split("\n"):
            assert t in q.answers and f"- {t}" in e["prompt"]              # every target is a listed candidate
    assert gold_in_candidates(["Paris"], ["paris", "Lyon"]) == ["paris"]
    kept = build_sft_examples(recs, ds, prompt_type="cand", keep_empty=True)
    assert len(kept) >= len(ex)
    one = build_sft_examples(recs, ds, prompt_type="chat", max_per_qid=1)
    assert len({e["qid"] for e in one}) == len(one) and all("Candidate answers" not in e["prompt"] for e in one)


def test_disable_stale_torchao_patches_every_peft_reference(monkeypatch):
    """Colab: peft's LoRA dispatcher probes torchao for each Linear and raises on an old torchao."""
    import types
    from kgqa_bench.llm_reader import disable_stale_torchao

    def boom():
        raise ImportError("Found an incompatible version of torchao")
    piu = types.ModuleType("peft.import_utils"); piu.is_torchao_available = boom
    lora_t = types.ModuleType("peft.tuners.lora.torchao"); lora_t.is_torchao_available = boom
    other = types.ModuleType("peft.tuners.lora.model")                     # no reference -> untouched
    pkg = types.ModuleType("peft"); pkg.import_utils = piu
    for name, m in [("peft", pkg), ("peft.import_utils", piu), ("peft.tuners.lora.torchao", lora_t), ("peft.tuners.lora.model", other)]:
        monkeypatch.setitem(sys.modules, name, m)
    assert disable_stale_torchao() is True
    assert piu.is_torchao_available() is False and lora_t.is_torchao_available() is False
    assert not hasattr(other, "is_torchao_available")
    assert disable_stale_torchao() is False                                 # probe now passes: nothing to do
    monkeypatch.setitem(sys.modules, "peft", None); monkeypatch.setitem(sys.modules, "peft.import_utils", None)
    assert disable_stale_torchao() is False                                 # peft absent: no-op


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
