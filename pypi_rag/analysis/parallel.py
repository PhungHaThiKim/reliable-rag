"""Parallel, cached static analysis.

`analyze_many(samples)` runs `analyze_package` in worker processes (one package per task), keeps a
per-package time budget (Unix: SIGALRM inside the worker; a package that exceeds it is re-scanned with
the regex fallback only and marked `analysis_timeout`, so coverage is reported as incomplete), and
caches every PackageFacts on disk under a key made of the sample hash, the analysis code hash and the
active rule fingerprint. Every later `cli` step and every rerun reuses the cache.
"""
from __future__ import annotations

import glob
import hashlib
import os
import pickle
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import taxonomy as T
from .ast_parser import PackageAnalyzer, PackageFacts, analyze_package

DEFAULT_TIMEOUT_S = 180


def code_hash() -> str:
    """Hash of the analysis package source (facts depend only on it and the rules)."""
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.py"))):
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def available_cpus() -> int:
    """CPUs this process may use: cgroup quota (containers report the host's cores in os.cpu_count()),
    then the affinity mask, then os.cpu_count()."""
    n = os.cpu_count() or 2
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            with open(path) as f:
                parts = f.read().split()
            if path.endswith("cpu.max") and parts and parts[0] != "max":
                n = min(n, max(1, int(int(parts[0]) / int(parts[1]))))
            elif path.endswith("quota_us") and parts and int(parts[0]) > 0:
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                    n = min(n, max(1, int(int(parts[0]) / int(f.read().split()[0]))))
            break
        except (OSError, ValueError, IndexError, ZeroDivisionError):
            continue
    return n


def _key(sample, chash: str) -> str:
    ruleset = (sample.meta or {}).get("fixture_ruleset", "")
    raw = f"{sample.sample_id}|{sample.sha256}|{len(sample.files)}|{chash}|{ruleset}|{T.rules_fingerprint()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def light_facts(sample, reason: str) -> PackageFacts:
    """Regex-only facts (linear time) for a package whose full analysis exceeded its budget."""
    pa = PackageAnalyzer(sample)
    n = 0
    for f in sample.files:
        if f.path.endswith((".py", ".pyw")):
            n += 1
            pa.parse_errors[f.path] = reason
            pa._regex_fallback(f.path, f.text)
    pa.limit("analysis_timeout")
    pa._assign_contexts({})
    return PackageFacts(sample_id=sample.sample_id, calls=pa.calls, flows=[], parse_errors=pa.parse_errors,
                        indicators=list(pa.indicators.values()), n_files=n, limits_hit=list(pa.limits))


def _work(args) -> Tuple[str, PackageFacts, float]:
    sample, timeout = args
    T.ensure_ruleset((sample.meta or {}).get("fixture_ruleset", ""))
    t0 = time.time()
    use_alarm = bool(timeout) and hasattr(signal, "SIGALRM")
    old = None
    if use_alarm:
        old = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(int(timeout))
    try:
        facts = analyze_package(sample)
    except _Timeout:
        facts = light_facts(sample, f"analysis timeout after {timeout}s (regex scan only)")
    except (RecursionError, MemoryError) as e:
        facts = light_facts(sample, f"analysis failed: {type(e).__name__} (regex scan only)")
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    return sample.sample_id, facts, time.time() - t0


def analyze_many(samples: Sequence, workers: int = 0, cache_dir: Optional[str] = None,
                 timeout: int = DEFAULT_TIMEOUT_S, log: Optional[Callable[[str], None]] = print
                 ) -> Dict[str, PackageFacts]:
    """sample_id -> PackageFacts for every sample (cached ones are loaded, others computed)."""
    for s in samples:                                     # activate fixture rules in this process too
        T.ensure_ruleset((s.meta or {}).get("fixture_ruleset", ""))
    chash = code_hash()
    out: Dict[str, PackageFacts] = {}
    todo = []
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    for s in samples:
        if s.sample_id in out:
            continue
        p = os.path.join(cache_dir, _key(s, chash) + ".pkl") if cache_dir else None
        if p and os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    out[s.sample_id] = pickle.load(f)     # files we wrote ourselves
                continue
            except Exception:                             # noqa: BLE001  corrupt cache entry: recompute
                pass
        todo.append(s)
    if not todo:
        if log:
            log(f"[analysis] {len(out)} packages from cache")
        return out
    n_workers = workers or max(1, min(64, available_cpus() - 1))
    n_workers = min(n_workers, len(todo))
    t0 = time.time()
    timings: List[Tuple[float, str]] = []
    if n_workers <= 1:
        results = map(_work, [(s, timeout) for s in todo])
        for sid, facts, dt in results:
            out[sid] = facts; timings.append((dt, sid))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for sid, facts, dt in ex.map(_work, [(s, timeout) for s in todo], chunksize=1):
                out[sid] = facts; timings.append((dt, sid))
    if cache_dir:
        by_id = {s.sample_id: s for s in todo}
        for _, sid in timings:
            with open(os.path.join(cache_dir, _key(by_id[sid], chash) + ".pkl"), "wb") as f:
                pickle.dump(out[sid], f, protocol=pickle.HIGHEST_PROTOCOL)
    if log:
        timings.sort(reverse=True)
        n_to = sum(1 for _, sid in timings if "analysis_timeout" in out[sid].limits_hit)
        log(f"[analysis] {len(todo)} packages analysed in {time.time() - t0:.0f}s with {n_workers} workers "
            f"({len(out) - len(todo)} from cache, {n_to} hit the {timeout}s budget); slowest: "
            + ", ".join(f"{sid} {dt:.1f}s" for dt, sid in timings[:5]))
    return out
