"""Dataset specs -> list[Sample]. Everything is read statically (safe_io); nothing is installed.

    synthetic:N[:seed]                 planted fixtures with gold annotations (data/synthetic.py)
    manifest:<samples.jsonl.gz>        output of `cli prepare` (texts + frozen split)
    malllm:<dir>                       mal-LLM `RQ_experiments/data/{train,test}_{malicious,benign}_packages_final.json`
                                       (setup.py only; label from the file name; upstream train/test kept in meta)
    pyguard_obf:<repo or dir>          PyGuard `Dataset/PyPI/Experiment/Obfuscation/{Malware,Benign}/*.tgz` (full packages)
    archives:<csv>                     columns path,label[,package_name,version,family_group,collection_timestamp,label_source]
    dirs:<root>                        <root>/{benign,malicious}/<archive or extracted dir>
    cases:[C01,C05,...]                controlled test cases C01-C06 (data/cases.py), split test; all when empty
Unknown labels are kept as `unknown` and excluded from scoring (plan 5.2).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from typing import Dict, List, Optional

from ..schema import Sample, SourceFile, load_samples, sha256_text
from .safe_io import Limits, UnsafeArchive, is_archive, read_package_sources
from .synthetic import make_synthetic

LFS_HEADER = "version https://git-lfs"


def _norm_label(x) -> str:
    s = str(x).strip().lower()
    if s in ("1", "malicious", "malware", "mal", "true", "yes"):
        return "malicious"
    if s in ("0", "benign", "clean", "false", "no"):
        return "benign"
    return "unknown"


def parse_pkg_info(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines()[:60]:
        m = re.match(r"^(Name|Version|Summary|Author|Home-page):\s*(.*)$", line)
        if m:
            out[m.group(1).lower()] = m.group(2).strip()
    return out


def name_version_from_filename(fn: str):
    base = os.path.basename(fn)
    for suf in (".tar.gz", ".tgz", ".tar.bz2", ".zip", ".whl", ".egg", ".tar"):
        if base.lower().endswith(suf):
            base = base[: -len(suf)]; break
    if base.count("-") >= 1:
        if fn.lower().endswith(".whl"):
            parts = base.split("-")
            return parts[0], parts[1] if len(parts) > 1 else ""
        name, _, ver = base.rpartition("-")
        if re.match(r"^v?\d", ver):
            return name, ver
    return base, ""


def sample_from_path(path: str, label: str, sample_id: Optional[str] = None, limits: Optional[Limits] = None,
                     **kw) -> Optional[Sample]:
    try:
        files, rep, sha = read_package_sources(path, limits)
    except (UnsafeArchive, OSError, EOFError, ValueError) as e:   # corrupt / unsafe archive: skip, keep a record
        print(f"[skip] {os.path.basename(path)}: {type(e).__name__}: {str(e)[:100]}")
        return None
    info = {}
    for f in files:
        if f.path.rsplit("/", 1)[-1] in ("PKG-INFO", "METADATA"):
            info = parse_pkg_info(f.text); break
    fn_name, fn_ver = name_version_from_filename(path)
    name = kw.pop("package_name", None) or info.get("name") or fn_name
    ver = kw.pop("version", None) or info.get("version") or fn_ver
    sid = sample_id or f"pypi-{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}-{ver or 'na'}-{sha[:8]}"
    meta = {"read_report": rep.as_dict(), "archive": os.path.basename(path)}
    meta.update(kw.pop("meta", {}) or {})
    return Sample(sample_id=sid, package_name=name, version=ver, label=label, files=files, sha256=sha,
                  source_ref=kw.pop("source_ref", path), meta=meta, **kw)


# ---------------------------------------------------------------------------------- mal-LLM
MALLLM_FILES = {("train", "malicious"): "train_malicious_packages_final.json",
                ("train", "benign"): "train_benign_packages_final.json",
                ("test", "malicious"): "test_malicious_packages_final.json",
                ("test", "benign"): "test_benign_packages_final.json"}


def _find_malllm_dir(root: str) -> str:
    for cand in (root, os.path.join(root, "RQ_experiments", "data"), os.path.join(root, "data")):
        if os.path.exists(os.path.join(cand, MALLLM_FILES[("test", "benign")])):
            return cand
    raise FileNotFoundError(f"mal-LLM *_packages_final.json not found under {root}")


def load_malllm(root: str, max_per_file: Optional[int] = None) -> List[Sample]:
    d = _find_malllm_dir(root)
    out: List[Sample] = []
    for (split, label), fn in MALLLM_FILES.items():
        p = os.path.join(d, fn)
        with open(p, encoding="utf-8") as f:
            head = f.read(64)
        if head.startswith(LFS_HEADER):
            raise RuntimeError(f"{fn} is a Git LFS pointer: run `git lfs pull` (or download from "
                               "media.githubusercontent.com) before loading")
        with open(p, encoding="utf-8") as f:
            rows = json.load(f)
        for i, r in enumerate(rows[:max_per_file] if max_per_file else rows):
            name = str(r.get("package_name", f"pkg{i}"))
            fl = r.get("file_list", [])
            fl = fl.split() if isinstance(fl, str) else list(fl or [])
            sp = r.get("setup.py")
            files: List[SourceFile] = []
            if isinstance(sp, str) and sp.strip() and sp.strip() != "Not Available":
                files.append(SourceFile(path="setup.py", sha256=sha256_text(sp), text=sp, n_lines=sp.count("\n") + 1))
            sha = sha256_text((sp or "") + "\n" + " ".join(fl))
            base, ver = name_version_from_filename(name)
            out.append(Sample(
                sample_id=f"malllm-{split}-{label[0]}-{i:05d}", package_name=base, version=ver, label=label,
                files=files, label_source=f"mal-LLM:{fn}", sha256=sha, source_ref=f"mal-LLM/{fn}#{i}",
                meta={"upstream_split": split, "file_list": fl, "setup_py_available": bool(files),
                      "upstream_name": name}))
    return out


# ---------------------------------------------------------------------------------- PyGuard
def load_pyguard_obfuscation(root: str, limits: Optional[Limits] = None, max_per_class: Optional[int] = None) -> List[Sample]:
    base = root
    sub = os.path.join(root, "Dataset", "PyPI", "Experiment", "Obfuscation")
    if os.path.isdir(sub):
        base = sub
    out: List[Sample] = []
    for cls, label in (("Malware", "malicious"), ("Benign", "benign")):
        paths = sorted(glob.glob(os.path.join(base, cls, "*")))
        paths = [p for p in paths if is_archive(p) or os.path.isdir(p)]
        if not paths:
            raise FileNotFoundError(f"no archives in {os.path.join(base, cls)} (did the sparse checkout include it?)")
        for p in paths[:max_per_class] if max_per_class else paths:
            sid = f"pyguard-obf-{label[0]}-{os.path.basename(p).split('.')[0][:16]}"
            s = sample_from_path(p, label, sample_id=sid, limits=limits, label_source="PyGuard:Experiment/Obfuscation",
                                 meta={"upstream_subset": "obfuscation"})
            if s is not None:
                out.append(s)
    return out


# ---------------------------------------------------------------------------------- generic
def load_archives_csv(path: str, limits: Optional[Limits] = None) -> List[Sample]:
    out = []
    root = os.path.dirname(os.path.abspath(path))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = row["path"] if os.path.isabs(row["path"]) else os.path.join(root, row["path"])
            kw = {k: row[k] for k in ("package_name", "version", "family_group", "collection_timestamp", "label_source")
                  if row.get(k)}
            s = sample_from_path(p, _norm_label(row.get("label", "")), limits=limits, **kw)
            if s is not None:
                out.append(s)
    return out


def load_dirs(root: str, limits: Optional[Limits] = None) -> List[Sample]:
    out = []
    for cls in sorted(os.listdir(root)):
        lab = _norm_label(cls)
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for p in sorted(os.listdir(d)):
            s = sample_from_path(os.path.join(d, p), lab, limits=limits, label_source=f"dir:{cls}")
            if s is not None:
                out.append(s)
    return out


def load_spec(spec: str, max_items: Optional[int] = None, split: Optional[str] = None) -> List[Sample]:
    kind, _, rest = spec.partition(":")
    if kind == "synthetic":
        parts = rest.split(":")
        out = make_synthetic(int(parts[0]), int(parts[1]) if len(parts) > 1 and parts[1] else 0)
    elif kind == "manifest":
        out = load_samples(rest)
    elif kind == "malllm":
        out = load_malllm(rest)
    elif kind == "pyguard_obf":
        out = load_pyguard_obfuscation(rest)
    elif kind == "archives":
        out = load_archives_csv(rest)
    elif kind == "dirs":
        out = load_dirs(rest)
    elif kind == "cases":
        from .cases import make_cases
        want = {x.strip().upper() for x in rest.split(",") if x.strip()}
        out = [s for s in make_cases() if not want or s.meta.get("case") in want]
    else:
        raise ValueError(f"unknown dataset spec: {spec}")
    if split:
        out = [s for s in out if s.split == split]
    return out[:max_items] if max_items else out
