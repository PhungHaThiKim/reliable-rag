"""Safe static reading, loaders, grouped split and leakage audit (CPU, no network)."""
import io
import json
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypi_rag.data.leakage_check import kb_leakage, split_leakage  # noqa: E402
from pypi_rag.data.loaders import load_malllm, load_spec, name_version_from_filename, sample_from_path  # noqa: E402
from pypi_rag.data.safe_io import Limits, read_package_sources, safe_member_name, tar_bytes_for_test  # noqa: E402
from pypi_rag.data.splitter import assign_splits  # noqa: E402
from pypi_rag.knowledge.documents import KnowledgeDoc  # noqa: E402


def test_member_names_reject_traversal():
    assert safe_member_name("pkg-1.0/setup.py") == "pkg-1.0/setup.py"
    assert safe_member_name("../etc/passwd") is None
    assert safe_member_name("/abs/x.py") is None
    assert safe_member_name("a/../../x.py") is None
    assert safe_member_name("C:/x.py") is None


def test_tar_reading_rejects_links_and_keeps_only_sources():
    data = tar_bytes_for_test({"pkg-1.0/setup.py": b"from setuptools import setup\nsetup()\n",
                               "pkg-1.0/pkg/__init__.py": b"x = 1\n", "pkg-1.0/README.md": b"hello",
                               "../evil.py": b"print(1)\n", "pkg-1.0/PKG-INFO": b"Name: pkg\nVersion: 1.0\n"},
                              symlinks={"pkg-1.0/link.py": "/etc/passwd"})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pkg-1.0.tar.gz")
        open(p, "wb").write(data)
        files, rep, sha = read_package_sources(p)
        paths = {f.path for f in files}
        assert paths == {"setup.py", "pkg/__init__.py", "PKG-INFO"}
        reasons = {r for _, r in rep.rejected}
        assert "path traversal" in reasons and "link" in reasons
        s = sample_from_path(p, "benign")
        assert s.package_name == "pkg" and s.version == "1.0" and s.sha256 == sha


def test_zip_bomb_ratio_and_size_limits():
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("p/big.py", "#" * 3_000_000)
        z.writestr("p/ok.py", "x = 1\n")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "p.zip"); open(p, "wb").write(bio.getvalue())
        files, rep, _ = read_package_sources(p, Limits(max_file_bytes=1_000_000))
        assert [f.path for f in files] == ["ok.py"]
        assert any("too large" in r or "ratio" in r for _, r in rep.rejected)


def test_name_version_from_filename():
    assert name_version_from_filename("requests-2.31.0.tar.gz") == ("requests", "2.31.0")
    assert name_version_from_filename("my_pkg-0.1-py3-none-any.whl") == ("my_pkg", "0.1")


def test_malllm_loader_format():
    with tempfile.TemporaryDirectory() as d:
        rows_m = [{"package_name": "evil-pkg-1.0", "file_list": ["setup.py", "evil/__init__.py"],
                   "setup.py": "import os\nos.system('id')\n"}]
        rows_b = [{"package_name": "good", "file_list": "setup.py good/__init__.py", "setup.py": "Not Available"}]
        for split in ("train", "test"):
            json.dump(rows_m, open(os.path.join(d, f"{split}_malicious_packages_final.json"), "w"))
            json.dump(rows_b, open(os.path.join(d, f"{split}_benign_packages_final.json"), "w"))
        xs = load_malllm(d)
        assert len(xs) == 4
        mal = [x for x in xs if x.label == "malicious"][0]
        assert mal.files[0].path == "setup.py" and mal.meta["upstream_split"] in ("train", "test")
        ben = [x for x in xs if x.label == "benign"][0]
        assert ben.files == [] and ben.meta["file_list"] == ["setup.py", "good/__init__.py"]


def test_grouped_split_keeps_groups_together_and_leakage_clean():
    xs = load_spec("synthetic:90:3")
    rep = assign_splits(xs, "grouped", (0.7, 0.15, 0.15), seed=1)
    by_group = {}
    for s in xs:
        by_group.setdefault(s.meta["group_id"], set()).add(s.split)
    assert all(len(v) == 1 for v in by_group.values())
    leak = split_leakage(xs)
    assert leak["family_across_splits"] == 0 and leak["package_name_across_splits"] == 0
    assert sum(v for k, v in rep["counts"].items() if k.startswith("test/")) > 0


def test_kb_leakage_flags_eval_derived_docs_and_advisories():
    xs = load_spec("synthetic:30:4")
    assign_splits(xs, "grouped", seed=0)
    test = [s for s in xs if s.split == "test"][0]
    docs = [KnowledgeDoc("a", "malicious_example", "t", "code", provenance={"origin_sample_id": test.sample_id}),
            KnowledgeDoc("b", "advisory", "t", f"Malware in {test.package_name}", provenance={}),
            KnowledgeDoc("c", "rule", "t", f"mentions {test.package_name} legitimately", provenance={}),
            KnowledgeDoc("d", "rule", "t", "clean", provenance={})]
    rep = kb_leakage(docs, xs)
    assert rep["docs_from_eval_samples"] == ["a"]
    assert "b" in rep["docs_mentioning_eval_package"] and "c" not in rep["docs_mentioning_eval_package"]
