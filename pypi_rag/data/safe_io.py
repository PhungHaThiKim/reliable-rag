"""Static, in-memory reading of untrusted package archives (plan section 5.4).

Nothing is ever extracted to disk, installed, imported or executed. Members are read
into memory one by one with hard limits, and rejected when they look like:
  * path traversal (absolute paths, `..`), symlinks / hardlinks / devices,
  * decompression bombs (member size, total size, compression ratio),
  * too many members, oversized files, undecodable text.
Only source-like files are kept (`.py`, `.pth`, `setup.cfg`, `pyproject.toml`, `PKG-INFO`, ...).
"""
from __future__ import annotations

import io
import os
import posixpath
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..schema import SourceFile, sha256_bytes

KEEP_SUFFIXES = (".py", ".pth", ".pyw", ".pyi")
KEEP_NAMES = ("setup.cfg", "pyproject.toml", "PKG-INFO", "METADATA", "entry_points.txt", "requirements.txt")
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".tar", ".zip", ".whl", ".egg")


@dataclass
class Limits:
    max_members: int = 20000
    max_file_bytes: int = 2_000_000        # a single kept file
    max_total_bytes: int = 50_000_000      # all kept files of one package
    max_ratio: float = 200.0               # uncompressed / compressed size of one zip member
    max_kept_files: int = 400
    max_chars_per_file: int = 400_000      # text kept per file (rest truncated, flagged)


@dataclass
class ReadReport:
    kept: int = 0
    skipped: int = 0
    rejected: List[Tuple[str, str]] = field(default_factory=list)   # (member, reason)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {"kept": self.kept, "skipped": self.skipped, "rejected": self.rejected[:50],
                "n_rejected": len(self.rejected), "warnings": self.warnings[:50]}


class UnsafeArchive(Exception):
    pass


def is_archive(path: str) -> bool:
    return path.lower().endswith(ARCHIVE_SUFFIXES)


def _wanted(name: str) -> bool:
    base = posixpath.basename(name)
    return base.endswith(KEEP_SUFFIXES) or base in KEEP_NAMES


def safe_member_name(name: str) -> Optional[str]:
    """Normalized posix member path, or None when it escapes the archive root."""
    n = name.replace("\\", "/")
    if n.startswith("/") or (len(n) > 1 and n[1] == ":"):
        return None
    norm = posixpath.normpath(n)
    if norm.startswith("..") or "/../" in f"/{norm}/" or norm in (".", ""):
        return None
    return norm


def decode_source(b: bytes) -> Tuple[str, str]:
    """(text, encoding). Unusual encodings are decoded with replacement and flagged."""
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return b.decode(enc), enc
        except UnicodeDecodeError:
            pass
    if b.count(b"\x00") > len(b) // 100 + 1:
        raise UnicodeDecodeError("binary", b[:1], 0, 1, "looks binary")
    return b.decode("latin-1", errors="replace"), "latin-1(fallback)"


def strip_root(paths: List[str]) -> str:
    """Common first directory of an sdist (`pkg-1.0/`), stripped from member names."""
    firsts = {p.split("/", 1)[0] for p in paths if "/" in p}
    if len(firsts) == 1 and all("/" in p for p in paths):
        return next(iter(firsts)) + "/"
    return ""


def _to_source_files(raw: Dict[str, bytes], lim: Limits, rep: ReadReport) -> List[SourceFile]:
    root = strip_root(sorted(raw))
    out: List[SourceFile] = []
    for name in sorted(raw):
        b = raw[name]
        try:
            text, enc = decode_source(b)
        except UnicodeDecodeError:
            rep.rejected.append((name, "undecodable")); continue
        if enc != "utf-8":
            rep.warnings.append(f"{name}: encoding {enc}")
        trunc = len(text) > lim.max_chars_per_file
        if trunc:
            text = text[:lim.max_chars_per_file]
            rep.warnings.append(f"{name}: truncated to {lim.max_chars_per_file} chars")
        path = name[len(root):] if root and name.startswith(root) else name
        out.append(SourceFile(path=path, sha256=sha256_bytes(b), text=text,
                              n_lines=text.count("\n") + 1, truncated=trunc, encoding=enc))
    return out


def read_tar(path: str, lim: Limits, rep: ReadReport) -> Dict[str, bytes]:
    raw: Dict[str, bytes] = {}
    total = 0
    with tarfile.open(path, "r:*") as tf:
        for i, m in enumerate(tf):
            if i >= lim.max_members:
                raise UnsafeArchive(f"more than {lim.max_members} members")
            name = safe_member_name(m.name)
            if name is None:
                rep.rejected.append((m.name, "path traversal")); continue
            if m.issym() or m.islnk():
                rep.rejected.append((m.name, "link")); continue
            if not m.isfile():
                continue                                # dirs, devices, fifos: never materialized
            if not _wanted(name):
                rep.skipped += 1; continue
            if m.size > lim.max_file_bytes:
                rep.rejected.append((m.name, f"too large ({m.size} B)")); continue
            if len(raw) >= lim.max_kept_files:
                rep.warnings.append("max_kept_files reached"); break
            f = tf.extractfile(m)
            if f is None:
                continue
            b = f.read(lim.max_file_bytes + 1)
            if len(b) > lim.max_file_bytes:
                rep.rejected.append((m.name, "size mismatch")); continue
            total += len(b)
            if total > lim.max_total_bytes:
                raise UnsafeArchive("total uncompressed size limit exceeded")
            raw[name] = b; rep.kept += 1
    return raw


def read_zip(path: str, lim: Limits, rep: ReadReport) -> Dict[str, bytes]:
    raw: Dict[str, bytes] = {}
    total = 0
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        if len(infos) > lim.max_members:
            raise UnsafeArchive(f"more than {lim.max_members} members")
        for m in infos:
            name = safe_member_name(m.filename)
            if name is None:
                rep.rejected.append((m.filename, "path traversal")); continue
            mode = (m.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                rep.rejected.append((m.filename, "link")); continue
            if m.is_dir() or not _wanted(name):
                rep.skipped += 1; continue
            if m.file_size > lim.max_file_bytes:
                rep.rejected.append((m.filename, f"too large ({m.file_size} B)")); continue
            if m.compress_size and m.file_size / max(m.compress_size, 1) > lim.max_ratio:
                rep.rejected.append((m.filename, "compression ratio (bomb?)")); continue
            if len(raw) >= lim.max_kept_files:
                rep.warnings.append("max_kept_files reached"); break
            with zf.open(m) as f:
                b = f.read(lim.max_file_bytes + 1)
            if len(b) > lim.max_file_bytes:
                rep.rejected.append((m.filename, "size mismatch")); continue
            total += len(b)
            if total > lim.max_total_bytes:
                raise UnsafeArchive("total uncompressed size limit exceeded")
            raw[name] = b; rep.kept += 1
    return raw


def read_dir(root: str, lim: Limits, rep: ReadReport) -> Dict[str, bytes]:
    """Already-extracted package directory. Symlinks are never followed."""
    raw: Dict[str, bytes] = {}
    total = 0
    root = os.path.abspath(root)
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns[:] = [d for d in dns if not os.path.islink(os.path.join(dp, d))]
        for fn in fns:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if os.path.islink(full):
                rep.rejected.append((rel, "link")); continue
            if not _wanted(rel):
                rep.skipped += 1; continue
            sz = os.path.getsize(full)
            if sz > lim.max_file_bytes:
                rep.rejected.append((rel, f"too large ({sz} B)")); continue
            if len(raw) >= lim.max_kept_files:
                rep.warnings.append("max_kept_files reached"); return raw
            with open(full, "rb") as f:
                b = f.read(lim.max_file_bytes + 1)
            total += len(b)
            if total > lim.max_total_bytes:
                raise UnsafeArchive("total size limit exceeded")
            raw[rel] = b; rep.kept += 1
    return raw


def read_package_sources(path: str, limits: Optional[Limits] = None) -> Tuple[List[SourceFile], ReadReport, str]:
    """Read an archive (.tar.gz/.zip/.whl/...) or a directory. Returns (files, report, archive_sha256).
    The archive hash is computed over the archive bytes; for a directory, over the sorted file hashes."""
    lim = limits or Limits()
    rep = ReadReport()
    if os.path.isdir(path):
        raw = read_dir(path, lim, rep)
        files = _to_source_files(raw, lim, rep)
        h = sha256_bytes("\n".join(f"{f.path}:{f.sha256}" for f in files).encode())
        return files, rep, h
    with open(path, "rb") as f:
        archive_sha = sha256_bytes(f.read())
    low = path.lower()
    if low.endswith((".zip", ".whl", ".egg")):
        raw = read_zip(path, lim, rep)
    elif tarfile.is_tarfile(path):
        raw = read_tar(path, lim, rep)
    elif zipfile.is_zipfile(path):
        raw = read_zip(path, lim, rep)
    else:
        raise UnsafeArchive(f"unknown archive type: {os.path.basename(path)}")
    return _to_source_files(raw, lim, rep), rep, archive_sha


def files_from_texts(texts: Dict[str, str]) -> List[SourceFile]:
    """In-memory fixture -> SourceFile list (synthetic data, tests)."""
    out = []
    for p in sorted(texts):
        t = texts[p]
        out.append(SourceFile(path=p, sha256=sha256_bytes(t.encode()), text=t, n_lines=t.count("\n") + 1))
    return out


def tar_bytes_for_test(members: Dict[str, bytes], symlinks: Dict[str, str] = None) -> bytes:
    """Build a tar.gz in memory (tests only)."""
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tf:
        for name, b in members.items():
            ti = tarfile.TarInfo(name); ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
        for name, target in (symlinks or {}).items():
            ti = tarfile.TarInfo(name); ti.type = tarfile.SYMTYPE; ti.linkname = target
            tf.addfile(ti)
    return bio.getvalue()
