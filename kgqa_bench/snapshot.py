"""Copy the light-weight result files the demo notebook (notebooks/03_demo_seminar.ipynb) reads into
`results/snapshot/`, so the demo runs without the Colab Drive and the folder can be submitted as is.

What is copied (sizes are small except the per-question files):
  * every reader directory  <prefix>_reader_<model>  for prefix in b08 / b08g / b08v / b08c / b08f:
        results.json, hallucination/hallucination.json, hallucination/report_*.md
  * per-question records and claims (for the worked examples) only for --readers and --methods
  * results.json of the retrieval runs (b02_baselines, b03_cgar, b06d_cgar_esc)
  * questions.json = {qid: {question, answers}} for the qids in the copied records (replaces the 100+ MB pickle)
  * MANIFEST.json  (what was copied, from where, total size)

    python -m kgqa_bench.snapshot --results <graph_rag>/results_gnn \
        --data-spec gnnrag:<graph_rag>/ckpt/gnn_features_webqsp_test.pkl:webqsp:test --out results/snapshot
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import time

READER_PREFIXES = ("b08", "b08g", "b08v", "b08c", "b08f")
RETRIEVAL_DIRS = ("b02_baselines", "b03_cgar", "b06d_cgar_esc")
DEFAULT_READERS = ("Qwen2.5-7B-Instruct",)
DEFAULT_METHODS = ("no_retrieval", "gnnrag_all", "gnnrag_top5", "sbert_top5", "cgar", "cgar_esc")


def _copy(src: str, dst: str, copied: list):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)


def make_snapshot(results: str, data_spec: str, out: str, readers=DEFAULT_READERS,
                  methods=DEFAULT_METHODS, verbose: bool = True) -> dict:
    os.makedirs(out, exist_ok=True)
    copied: list = []
    qids: set = set()
    reader_dirs = sorted(d for p in READER_PREFIXES for d in glob.glob(f"{results}/{p}_reader_*")
                         if not re.search(r"_old\d*$", d) and os.path.exists(f"{d}/results.json"))
    for d in reader_dirs:
        name = os.path.basename(d); dst = f"{out}/{name}"
        _copy(f"{d}/results.json", f"{dst}/results.json", copied)
        _copy(f"{d}/hallucination/hallucination.json", f"{dst}/hallucination/hallucination.json", copied)
        for rep in glob.glob(f"{d}/hallucination/report_*.md"):
            _copy(rep, f"{dst}/hallucination/{os.path.basename(rep)}", copied)
        if not any(name.endswith(f"_reader_{r}") for r in readers):
            continue
        for m in methods:
            for p in glob.glob(f"{d}/records_*_{m}.jsonl"):
                _copy(p, f"{dst}/{os.path.basename(p)}", copied)
                with open(p) as f:
                    qids.update(json.loads(l)["qid"] for l in f if l.strip())
            for p in glob.glob(f"{d}/hallucination/claims_*_{m}.jsonl"):
                _copy(p, f"{dst}/hallucination/{os.path.basename(p)}", copied)
    for r in RETRIEVAL_DIRS:
        _copy(f"{results}/{r}/results.json", f"{out}/{r}/results.json", copied)
    _copy(f"{results}/p9_hallucination_summary.md", f"{out}/p9_hallucination_summary.md", copied)

    n_q = 0
    if data_spec and qids:
        from .data import load_spec
        ds = load_spec(data_spec)
        qs = {q.qid: {"question": q.question, "answers": list(q.answers)} for q in ds if q.qid in qids}
        with open(f"{out}/questions.json", "w") as f:
            json.dump(qs, f, ensure_ascii=False)
        copied.append(f"{out}/questions.json"); n_q = len(qs)

    size = sum(os.path.getsize(p) for p in copied)
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "source_results": results, "data_spec": data_spec,
                "reader_dirs": [os.path.basename(d) for d in reader_dirs], "records_for_readers": list(readers),
                "records_for_methods": list(methods), "n_files": len(copied), "n_questions": n_q,
                "size_mb": round(size / 2 ** 20, 2)}
    with open(f"{out}/MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    if verbose:
        print(f"snapshot -> {out}: {len(copied)} files, {manifest['size_mb']} MB, {len(reader_dirs)} reader dirs, {n_q} questions")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="directory with b08*_reader_* (graph_rag/results_gnn)")
    ap.add_argument("--data-spec", default=None, help="dataset spec for question text and gold answers")
    ap.add_argument("--out", default="results/snapshot")
    ap.add_argument("--readers", nargs="+", default=list(DEFAULT_READERS), help="readers whose per-question files are copied")
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    a = ap.parse_args(argv)
    make_snapshot(a.results, a.data_spec, a.out, a.readers, a.methods)


if __name__ == "__main__":
    main()
