"""Static AST analysis of one package. Code is parsed with `ast.parse`, NEVER imported or executed.

Produces `PackageFacts`:
  * call sites with a canonical API name (import aliases, `from x import y as z`, object types such as
    `s = socket.socket(); s.send(..)` -> `socket.socket.send`, `__import__('os').system`, `getattr(os, 'system')`),
    their behaviors (taxonomy), string literals / indicators and execution context
    (`install` = setup.py top level, setuptools cmdclass, .pth; `import` = __init__.py top level; `runtime`);
  * data flows between sensitive call sites, each with provenance and an extractor tag:
      ast_nested         value of one call is directly an argument of another (confidence 1.0)
      intraproc_def_use  def-use through local variables in the same function (0.8)
      interproc_summary  through a package-local function's return value / parameter (0.6)
      regex              files that do not parse (python 2, obfuscation): API names found by regex, no flows (0.3)
Two calls in the same file WITHOUT a def-use chain never get a flow (plan 10.3, critical distinction).
"""
from __future__ import annotations

import ast
import posixpath
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from . import taxonomy as T

BUILTIN_NAMES = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "input", "execfile",
                 "str", "bytes", "dict", "list", "tuple", "set", "repr", "format", "bytearray", "sorted",
                 "reversed", "map", "filter", "zip", "iter", "next", "print", "len", "chr", "ord", "globals",
                 "locals", "vars", "hex", "int", "float", "bool", "type", "object", "super", "isinstance"}
OPEN_APIS = ("builtins.open", "io.open", "codecs.open", "os.fdopen")
INSTALL_BASES = ("setuptools.command.", "distutils.command.", "setuptools.Command", "distutils.core.Command")
# calls whose result is an object we track the type of (so `s.send(..)` resolves to `socket.socket.send`)
FACTORIES = {"socket.socket", "socket.create_connection", "builtins.open", "io.open", "codecs.open", "os.fdopen",
             "urllib.request.urlopen", "urllib2.urlopen", "requests.session", "sqlite3.connect", "tempfile.NamedTemporaryFile",
             "pathlib.Path", "mss.mss", "telebot.TeleBot", "pycurl.Curl", "urllib3.PoolManager"}


PATH_FUNCS = {"os.path.expanduser", "os.path.join", "os.path.expandvars", "os.path.abspath", "os.path.normpath",
              "pathlib.Path", "builtins.str", "pathlib.Path.home"}


def _is_factory(api: str) -> bool:
    last = api.rsplit(".", 1)[-1]
    return api in FACTORIES or (last[:1].isupper() and not api.startswith("builtins."))
Origins = Dict[str, Tuple[str, ...]]          # origin site id -> variable names it travelled through
# Deterministic cost limits (obfuscated packages can otherwise make the taint analysis quadratic).
MAX_ORIGINS = 32            # origin sites tracked per value
MAX_FLOWS = 20000           # data-flow edges per package
MAX_CHAINS = 2000           # flow chains enumerated per package
MAX_AST_LINES = 20000       # larger files are scanned with the regex fallback only
MAX_AST_CHARS = 2_000_000
MAX_REGEX_LINE = 5000       # regex fallback looks at the first N characters of a line


@dataclass
class CallSite:
    cid: str
    file: str
    line: int
    end_line: int
    col: int
    api: str
    raw: str
    behaviors: List[str]
    scope: str                       # function key "file::qualname" or "file::<module>"
    kind: str = "call"               # call | attribute | regex | hook
    literals: List[str] = field(default_factory=list)
    indicators: List[Tuple[str, str]] = field(default_factory=list)
    context: str = "runtime"         # install | import | runtime
    extractor: str = "ast"
    confidence: float = 1.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Flow:
    src: str
    dst: str
    via: Tuple[str, ...]
    extractor: str
    confidence: float
    file: str
    line_start: int
    line_end: int
    via_functions: Tuple[str, ...] = ()      # package functions the value passed through (interprocedural)
    call_sites: Tuple[str, ...] = ()         # "file:line" of the calls that moved the value between functions

    def to_dict(self):
        return asdict(self)


@dataclass
class FunctionInfo:
    key: str
    file: str
    qualname: str
    line: int
    end_line: int
    params: List[str] = field(default_factory=list)
    is_install_cmd: bool = False


@dataclass
class PackageFacts:
    sample_id: str
    calls: Dict[str, CallSite] = field(default_factory=dict)
    flows: List[Flow] = field(default_factory=list)
    functions: Dict[str, FunctionInfo] = field(default_factory=dict)
    modules: List[str] = field(default_factory=list)
    local_calls: List[Tuple[str, str, str, int]] = field(default_factory=list)   # (caller, callee, file, line)
    indicators: List[Dict] = field(default_factory=list)                          # {kind, value, file, line}
    parse_errors: Dict[str, str] = field(default_factory=dict)
    n_files: int = 0
    dispatch_sites: List[Dict] = field(default_factory=list)   # unresolved dynamic import / getattr targets
    limits_hit: List[str] = field(default_factory=list)        # cost limits that truncated the analysis

    # ---- views ----------------------------------------------------------------
    def sensitive_calls(self) -> List[CallSite]:
        return sorted((c for c in self.calls.values() if c.behaviors), key=lambda c: (c.file, c.line, c.col))

    def flow_graph(self) -> Dict[str, List[Flow]]:
        g: Dict[str, List[Flow]] = defaultdict(list)
        for f in self.flows:
            g[f.src].append(f)
        return g

    def flow_chains(self, max_hops: int = 3, max_chains: int = MAX_CHAINS) -> List[List[Flow]]:
        """Simple flow chains of 1..max_hops edges starting at a site with no incoming flow (or any source
        site); at most `max_chains` (deterministic order)."""
        g = self.flow_graph()
        incoming = {f.dst for f in self.flows}
        starts = [c for c in g if c not in incoming or set(self.calls[c].behaviors) & set(T.SOURCE_BEHAVIORS)]
        out: List[List[Flow]] = []
        for s in starts:
            stack = [(s, [])]
            while stack and len(out) < max_chains:
                node, path = stack.pop()
                for e in g.get(node, []):
                    if any(e.dst == p.src for p in path) or e.dst == s:
                        continue
                    np_ = path + [e]
                    out.append(np_)
                    if len(np_) < max_hops:
                        stack.append((e.dst, np_))
        return out

    def relational_behaviors(self, max_hops: int = 3) -> Dict[str, List[List[Flow]]]:
        res: Dict[str, List[List[Flow]]] = defaultdict(list)
        for ch in self.flow_chains(max_hops):
            b0 = set(self.calls[ch[0].src].behaviors); bl = set(self.calls[ch[-1].dst].behaviors)
            if b0 & {"env_access", "file_read", "sensitive_data", "system_info", "surveillance"} and "network_send" in bl:
                res["data_exfiltration"].append(ch)
            if "network_download" in b0 and bl & {"code_exec", "process_exec"}:
                res["download_exec"].append(ch)
        return dict(res)

    def behaviors(self, include_relational: bool = True) -> Dict[str, int]:
        """behavior -> count of supporting sites / chains (package-level facts used by the verifier)."""
        cnt: Dict[str, int] = defaultdict(int)
        for c in self.calls.values():
            for b in c.behaviors:
                cnt[b] += 1
        for ind in self.indicators:
            for b in T.INDICATOR_BEHAVIORS.get(ind["kind"], ()):
                cnt[b] += 1
        if include_relational:
            for b, chains in self.relational_behaviors().items():
                cnt[b] += len(chains)
        return dict(cnt)

    def parse_coverage(self) -> float:
        return 1.0 - len(self.parse_errors) / max(self.n_files, 1)

    def complete(self) -> bool:
        """True when absence of a fact can be read as absence of the behavior: every file parsed, no
        unresolved dynamic dispatch, no cost limit hit."""
        return self.parse_coverage() >= 1.0 and not self.dispatch_sites and not self.limits_hit

    def summary(self) -> Dict:
        ctx = defaultdict(int)
        for c in self.sensitive_calls():
            ctx[c.context] += 1
        return {"n_files": self.n_files, "n_calls": len(self.calls), "n_sensitive": len(self.sensitive_calls()),
                "n_flows": len(self.flows), "n_parse_errors": len(self.parse_errors),
                "behaviors": self.behaviors(), "contexts": dict(ctx)}


# =====================================================================================

class _Scope:
    def __init__(self, key: str, parent: Optional["_Scope"] = None, cls: Optional[str] = None):
        self.key, self.parent, self.cls = key, parent, cls
        self.taint: Dict[str, Origins] = {}
        self.types: Dict[str, str] = {}
        self.consts: Dict[str, str] = {}

    def lookup(self, attr: str, name: str):
        s = self
        while s is not None:
            d = getattr(s, attr)
            if name in d:
                return d[name]
            s = s.parent
        return None


def _merge(a: Origins, b: Origins) -> Origins:
    if not b:
        return a
    out = dict(a)
    for k, v in b.items():
        if k in out:
            if len(v) < len(out[k]):
                out[k] = v
        elif len(out) < MAX_ORIGINS:
            out[k] = v
    return out


def _end(node) -> int:
    return getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)


class _FileAnalyzer:
    """One file, one pass (called twice by PackageAnalyzer so function summaries can propagate)."""

    def __init__(self, pkg: "PackageAnalyzer", path: str, tree: ast.Module):
        self.pkg, self.path, self.tree = pkg, path, tree
        self.base = posixpath.basename(path)
        self.alias: Dict[str, str] = {}
        self.local_alias: Dict[str, str] = {}      # name -> local function short name (relative / package imports)
        self.mod_scope = _Scope(f"{path}::<module>")
        self._collect_imports()

    # ---- imports -----------------------------------------------------------------
    def _collect_imports(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.alias[a.asname] = a.name
                    else:
                        top = a.name.split(".")[0]
                        self.alias.setdefault(top, top)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                local = node.level > 0 or mod.split(".")[0] in self.pkg.top_names
                for a in node.names:
                    if a.name == "*":
                        continue
                    nm = a.asname or a.name
                    if local:
                        self.local_alias[nm] = a.name
                    else:
                        self.alias[nm] = f"{mod}.{a.name}" if mod else a.name

    # ---- name resolution ------------------------------------------------------------
    def _dotted(self, node) -> Optional[Tuple[str, List[str]]]:
        attrs = []
        while isinstance(node, ast.Attribute):
            attrs.append(node.attr); node = node.value
        if isinstance(node, ast.Name):
            return node.id, list(reversed(attrs))
        return None

    def _resolve_base(self, name: str, sc: _Scope) -> Optional[str]:
        t = sc.lookup("types", name)
        if t:
            return t
        if name in self.alias:
            return self.alias[name]
        if name in BUILTIN_NAMES:
            return f"builtins.{name}"
        return None

    def _local_key(self, short: str, sc: _Scope, attr_of_self: bool = False) -> Optional[str]:
        if attr_of_self and sc.cls:
            k = f"{self.path}::{sc.cls}.{short}"
            if k in self.pkg.functions:
                return k
        k = f"{self.path}::{short}"
        if k in self.pkg.functions:
            return k
        target = self.local_alias.get(short, short if short not in BUILTIN_NAMES else None)
        if target:
            cands = self.pkg.by_short.get(target, [])
            if len(cands) == 1:
                return cands[0]
        return None

    def _call_target(self, node: ast.Call, sc: _Scope):
        """-> (api or None, raw, receiver node or None, local function key or None)."""
        f = node.func
        d0 = self._dotted(f) if isinstance(f, (ast.Name, ast.Attribute)) else None
        raw = ".".join([d0[0]] + d0[1])[:80] if d0 else type(f).__name__   # never unparse big expressions
        if isinstance(f, ast.Name):
            lk = None
            if f.id not in self.alias and sc.lookup("types", f.id) is None:
                lk = self._local_key(f.id, sc)
            if lk:
                return None, raw, None, lk
            return self._resolve_base(f.id, sc), raw, None, None
        if isinstance(f, ast.Attribute):
            d = self._dotted(f)
            if d:
                base, attrs = d
                if base in ("self", "cls") and len(attrs) == 1:
                    lk = self._local_key(attrs[0], sc, attr_of_self=True)
                    return None, raw, (None if lk else f.value), lk
                if base in ("self", "cls") and len(attrs) >= 2 and sc.cls:
                    t = self.pkg.class_types.get((self.path, sc.cls), {}).get(attrs[0])
                    if t:
                        return t + "." + ".".join(attrs[1:]), raw, f.value, None
                rb = self._resolve_base(base, sc)
                if rb:
                    return rb + "." + ".".join(attrs), raw, f.value, None
                if base in self.local_alias and len(attrs) == 1:      # from . import mod; mod.fn()
                    lk = self._local_key(attrs[0], sc)
                    if lk:
                        return None, raw, None, lk
                return None, raw, f.value, None
            inner = f.value
            if isinstance(inner, ast.Call):
                ia, _, _, _ = self._call_target(inner, sc)
                if ia in ("builtins.__import__", "importlib.import_module") and inner.args:
                    lit = self._const_str(inner.args[0], sc)
                    if lit:
                        return f"{lit}.{f.attr}", raw, None, None
                if ia:
                    return f"{ia}.{f.attr}", raw, inner, None
                return None, raw, inner, None
            if isinstance(inner, ast.Constant) and isinstance(inner.value, (str, bytes)):
                return f"builtins.{type(inner.value).__name__}.{f.attr}", raw, inner, None
            return None, raw, inner, None
        if isinstance(f, ast.Call):        # getattr(os, "system")(cmd)
            ia, _, _, _ = self._call_target(f, sc)
            if ia == "builtins.getattr" and len(f.args) >= 2:
                d = self._dotted(f.args[0]); lit = self._const_str(f.args[1], sc)
                if d and lit:
                    rb = self._resolve_base(d[0], sc)
                    if rb:
                        return ".".join([rb] + d[1] + [lit]), raw, None, None
            return None, raw, f, None
        return None, raw, None, None

    # ---- constants ---------------------------------------------------------------------
    def _const_str(self, node, sc: _Scope, depth: int = 0) -> Optional[str]:
        if node is None or depth > 6:
            return None
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            if isinstance(node.value, bytes):
                return node.value.decode("latin-1")
            return None
        if isinstance(node, ast.Name):
            return sc.lookup("consts", node.id) or self.mod_scope.consts.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = self._const_str(node.left, sc, depth + 1), self._const_str(node.right, sc, depth + 1)
            if a is not None or b is not None:
                return (a or "") + (b or "")
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                s = self._const_str(v.value if isinstance(v, ast.FormattedValue) else v, sc, depth + 1)
                parts.append(s if s is not None else "{}")
            return "".join(parts)
        if isinstance(node, ast.Call):
            api, _, _, _ = self._call_target(node, sc)
            if api in PATH_FUNCS:
                parts = [self._const_str(a, sc, depth + 1) for a in node.args]
                parts = [p for p in parts if p]
                return "/".join(parts) if parts else None
            return None
        if isinstance(node, (ast.List, ast.Tuple)):
            parts = [self._const_str(e, sc, depth + 1) for e in node.elts]
            parts = [p for p in parts if p]
            return " ".join(parts) if parts else None
        return None

    # ---- expression evaluation: origins of a value ------------------------------------------
    def _eval(self, node, sc: _Scope) -> Origins:
        if node is None:
            return {}
        if isinstance(node, ast.Call):
            return self._call(node, sc)
        if isinstance(node, ast.Name):
            got = sc.lookup("taint", node.id) or {}
            return {k: v + (node.id,) for k, v in got.items()}
        if isinstance(node, ast.Attribute):
            d = self._dotted(node)
            if d:
                rb = self._resolve_base(d[0], sc)
                full = ".".join([rb] + d[1]) if rb else None
                for src, bs in T.ATTRIBUTE_SOURCES.items():
                    if full and (full == src or full.startswith(src + ".")) and bs:
                        site = self._site(node, src, src, list(bs), sc, kind="attribute")
                        return {site.cid: ()}
            return self._eval(node.value, sc)
        if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return {}
        out: Origins = {}
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.stmt):
                continue
            out = _merge(out, self._eval(ch, sc))
        return out

    def _site(self, node, api: str, raw: str, behaviors: List[str], sc: _Scope, kind: str = "call",
              literals: Iterable[str] = (), indicators: Iterable[Tuple[str, str]] = ()) -> CallSite:
        cid = f"{self.path}:{node.lineno}:{node.col_offset}"
        site = self.pkg.calls.get(cid)
        if site is None:
            site = CallSite(cid=cid, file=self.path, line=node.lineno, end_line=_end(node), col=node.col_offset,
                            api=api, raw=raw, behaviors=list(dict.fromkeys(behaviors)), scope=sc.key, kind=kind,
                            literals=[l[:200] for l in literals][:8], indicators=list(indicators)[:8])
            self.pkg.calls[cid] = site
        else:
            for b in behaviors:
                if b not in site.behaviors:
                    site.behaviors.append(b)
        return site

    def _add_flow(self, origins: Origins, dst: CallSite, sc: _Scope, extractor: Optional[str] = None,
                  conf: Optional[float] = None, param_hook: Optional[str] = None):
        for src, via_raw in origins.items():
            if src.startswith("param:"):
                fkey, idx = src[6:].rsplit(":", 1)
                self.pkg.param_sinks[fkey][int(idx)].add(dst.cid)
                continue
            if src == dst.cid or src not in self.pkg.calls:
                continue
            if len(self.pkg._flows) >= MAX_FLOWS:
                self.pkg.limit("max_flows")
                return
            # function hops are recorded as "name()@file:line"
            via = tuple(v.split("@", 1)[0] for v in via_raw)
            via_fns = tuple(v for v in via if v.endswith("()"))
            sites = tuple(v.split("@", 1)[1] for v in via_raw if "@" in v)
            ex = extractor or ("ast_nested" if not via else
                               "interproc_summary" if via_fns else "intraproc_def_use")
            cf = conf if conf is not None else {"ast_nested": 1.0, "intraproc_def_use": 0.8}.get(ex, 0.6)
            s = self.pkg.calls[src]
            # a network call that receives data from a local source / transform is SENDING it
            if set(dst.behaviors) & set(T.NETWORK) and "network_send" not in dst.behaviors and                     set(s.behaviors) & (set(T.SOURCE_BEHAVIORS) - {"network_download"} | set(T.TRANSFORM_BEHAVIORS)):
                dst.behaviors.append("network_send")
            lo, hi = sorted((s.line, dst.line))
            self.pkg.add_flow(Flow(src, dst.cid, via, ex, cf, dst.file if s.file == dst.file else f"{s.file}|{dst.file}",
                                   lo, max(hi, dst.end_line), via_fns, sites))

    def _call(self, node: ast.Call, sc: _Scope) -> Origins:
        api, raw, recv, lkey = self._call_target(node, sc)
        args = list(node.args) + [k.value for k in node.keywords]
        arg_o: List[Origins] = [self._eval(a, sc) for a in args]
        all_args: Origins = {}
        for o in arg_o:
            all_args = _merge(all_args, o)
        recv_o = self._eval(recv, sc) if recv is not None else {}

        if lkey is not None:                                   # package-local function
            self.pkg.local_calls_set.add((sc.key, lkey, self.path, node.lineno))
            hop = f"{lkey.split('::')[-1].split('.')[-1]}()@{self.path}:{node.lineno}"
            for i, o in enumerate(arg_o[:len(node.args)]):
                for sink in self.pkg.param_sinks.get(lkey, {}).get(i, ()):
                    if sink in self.pkg.calls:
                        self._add_flow({k: v + (hop,) for k, v in o.items()}, self.pkg.calls[sink], sc,
                                       "interproc_summary", 0.6)
            ret: Origins = {}
            for src, via in self.pkg.ret_origins.get(lkey, {}).items():
                if src.startswith("param:"):
                    idx = int(src.rsplit(":", 1)[1])
                    if idx < len(arg_o):
                        ret = _merge(ret, {k: v + (hop,) for k, v in arg_o[idx].items()})
                elif len(ret) < MAX_ORIGINS:
                    ret[src] = via + (hop,)
            return ret

        if api in ("importlib.import_module", "builtins.__import__", "builtins.getattr", "importlib.__import__"):
            target = node.args[1] if api == "builtins.getattr" and len(node.args) >= 2 else (node.args[0] if node.args else None)
            if target is None or self._const_str(target, sc) is None:
                self.pkg.dispatch.setdefault(f"{self.path}:{node.lineno}:{node.col_offset}", {
                    "file": self.path, "line": node.lineno, "api": api, "scope": sc.key})
        behaviors = list(T.behaviors_of(api)) if api else []
        if api in OPEN_APIS:
            mode = None
            if len(node.args) >= 2:
                mode = self._const_str(node.args[1], sc)
            for k in node.keywords:
                if k.arg == "mode":
                    mode = self._const_str(k.value, sc)
            behaviors.append("file_write" if mode and any(ch in mode for ch in "wax+") else "file_read")
        if api in ("urllib.request.urlopen", "urllib2.urlopen", "urllib.urlopen") and (
                len(node.args) >= 2 or any(k.arg == "data" for k in node.keywords)):
            behaviors.append("network_send")
        if api in T.TRANSPARENT:
            behaviors = []
        literals = [s for s in (self._const_str(a, sc) for a in args) if s]
        inds = [ind for s in literals for ind in T.string_indicators(s)]
        if behaviors:
            for kind, _ in inds:
                for b in T.INDICATOR_BEHAVIORS.get(kind, ()):
                    if b == "network_send" and not set(behaviors) & set(T.NETWORK):
                        continue
                    behaviors.append(b)
        if not behaviors:
            return _merge(all_args, recv_o)                     # transparent / unknown: value carries its inputs

        site = self._site(node, api, raw, behaviors, sc, literals=literals, indicators=inds)
        self._add_flow(all_args, site, sc)
        return {site.cid: ()}

    # ---- statements ----------------------------------------------------------------------------
    def _bind(self, target, origins: Origins, sc: _Scope, value=None):
        if isinstance(target, ast.Name):
            if origins:
                sc.taint[target.id] = _merge(sc.taint.get(target.id, {}), origins)
            if value is not None:
                s = self._const_str(value, sc)
                if s is not None:
                    sc.consts[target.id] = s
                if isinstance(value, ast.Call):
                    api, _, _, _ = self._call_target(value, sc)
                    if api and _is_factory(api):
                        sc.types[target.id] = api
        elif isinstance(target, (ast.Tuple, ast.List)):
            for e in target.elts:
                self._bind(e, origins, sc)
        elif isinstance(target, ast.Starred):
            self._bind(target.value, origins, sc)
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
                    and sc.cls and isinstance(value, ast.Call)):
                api, _, _, _ = self._call_target(value, sc)
                if api and _is_factory(api):
                    self.pkg.class_types[(self.path, sc.cls)][target.attr] = api
            d = self._dotted(target.value) if isinstance(target, ast.Subscript) else self._dotted(target)
            if d and origins:   # x[k] = tainted  /  self.x = tainted  -> taint the base name
                nm = d[0] if isinstance(target, ast.Subscript) else ".".join([d[0]] + d[1])
                sc.taint[nm] = _merge(sc.taint.get(nm, {}), origins)

    def _stmts(self, body: List[ast.stmt], sc: _Scope):
        for st in body:
            try:
                self._stmt(st, sc)
            except RecursionError:
                self.pkg.parse_errors.setdefault(self.path, "recursion limit in statement")

    def _stmt(self, st: ast.stmt, sc: _Scope):
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return                                                   # analysed as its own scope
        if isinstance(st, ast.ClassDef):
            for d in st.decorator_list:
                self._eval(d, sc)
            self._stmts([s for s in st.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))], sc)
            return
        if isinstance(st, ast.Assign):
            o = self._eval(st.value, sc)
            for t in st.targets:
                self._bind(t, o, sc, st.value)
            return
        if isinstance(st, (ast.AnnAssign, ast.AugAssign)):
            if st.value is not None:
                o = self._eval(st.value, sc)
                self._bind(st.target, o, sc, st.value if isinstance(st, ast.AnnAssign) else None)
            return
        if isinstance(st, (ast.For, ast.AsyncFor)):
            self._bind(st.target, self._eval(st.iter, sc), sc)
            self._stmts(st.body, sc); self._stmts(st.orelse, sc); return
        if isinstance(st, (ast.With, ast.AsyncWith)):
            for it in st.items:
                o = self._eval(it.context_expr, sc)
                if it.optional_vars is not None:
                    self._bind(it.optional_vars, o, sc, it.context_expr)
            self._stmts(st.body, sc); return
        if isinstance(st, ast.Return):
            if st.value is not None:
                o = self._eval(st.value, sc)
                self.pkg.ret_origins[sc.key] = _merge(self.pkg.ret_origins.get(sc.key, {}), o)
            return
        if isinstance(st, (ast.If, ast.While)):
            self._eval(st.test, sc); self._stmts(st.body, sc); self._stmts(st.orelse, sc); return
        if isinstance(st, ast.Try) or type(st).__name__ == "TryStar":
            self._stmts(st.body, sc)
            for h in st.handlers:
                self._stmts(h.body, sc)
            self._stmts(st.orelse, sc); self._stmts(st.finalbody, sc); return
        if type(st).__name__ == "Match":
            self._eval(st.subject, sc)
            for c in st.cases:
                self._stmts(c.body, sc)
            return
        for ch in ast.iter_child_nodes(st):                         # Expr, Raise, Assert, Delete, ...
            if isinstance(ch, ast.expr):
                self._eval(ch, sc)

    # ---- driver ----------------------------------------------------------------------------------
    def functions(self) -> List[Tuple[str, ast.AST, Optional[str], str]]:
        """(qualname, node, class name, parent qualname) for every def in the file."""
        out = []

        def rec(body, prefix, cls):
            for st in body:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    q = f"{prefix}{st.name}"
                    out.append((q, st, cls, prefix[:-1] if prefix else ""))
                    rec(st.body, q + ".", cls)
                elif isinstance(st, ast.ClassDef):
                    rec(st.body, f"{prefix}{st.name}.", f"{prefix}{st.name}")
                elif isinstance(st, (ast.If, ast.Try, ast.With)) or type(st).__name__ == "TryStar":
                    for blk in ("body", "orelse", "finalbody"):
                        rec(getattr(st, blk, []) or [], prefix, cls)
                    for h in getattr(st, "handlers", []) or []:
                        rec(h.body, prefix, cls)
        rec(self.tree.body, "", None)
        return out

    def install_cmd_classes(self) -> Set[str]:
        out = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                for b in node.bases:
                    d = self._dotted(b)
                    if not d:
                        continue
                    rb = self._resolve_base(d[0], self.mod_scope) or d[0]
                    full = ".".join([rb] + d[1])
                    if full.startswith(INSTALL_BASES) or full in ("install", "develop", "egg_info", "build_py", "sdist"):
                        out.add(node.name)
        return out

    def run(self):
        # module scope first (constants and module-level taint visible to functions)
        self.mod_scope = _Scope(f"{self.path}::<module>")
        self._stmts(self.tree.body, self.mod_scope)
        for q, fn, cls, _parent in self.functions():
            key = f"{self.path}::{q}"
            sc = _Scope(key, parent=self.mod_scope, cls=cls)
            args = fn.args
            params = [a.arg for a in list(getattr(args, "posonlyargs", [])) + list(args.args)]
            if cls and params and params[0] in ("self", "cls"):
                params = params[1:]
            for i, p in enumerate(params):
                sc.taint[p] = {f"param:{key}:{i}": ()}
            for d in fn.decorator_list:
                self._eval(d, self.mod_scope)
            self._stmts(fn.body, sc)
            self._stmts(fn.body, sc)              # second pass: loops / later definitions

        # string indicators anywhere in the file
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) >= 6:
                for kind, val in T.string_indicators(node.value):
                    self.pkg.add_indicator(kind, val, self.path, node.lineno)


class PackageAnalyzer:
    def __init__(self, sample):
        self.sample = sample
        self.calls: Dict[str, CallSite] = {}
        self._flows: Dict[Tuple[str, str], Flow] = {}
        self.functions: Dict[str, FunctionInfo] = {}
        self.by_short: Dict[str, List[str]] = defaultdict(list)
        self.ret_origins: Dict[str, Origins] = {}
        self.param_sinks: Dict[str, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.local_calls_set: Set[Tuple[str, str, str, int]] = set()
        self.class_types: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
        self.indicators: Dict[Tuple[str, str, str, int], Dict] = {}
        self.parse_errors: Dict[str, str] = {}
        self.dispatch: Dict[str, Dict] = {}
        self.limits: List[str] = []
        self.top_names = self._top_names()
        T.ensure_ruleset((getattr(sample, "meta", None) or {}).get("fixture_ruleset", ""))

    def limit(self, name: str):
        if name not in self.limits:
            self.limits.append(name)

    def _top_names(self) -> Set[str]:
        names = set()
        for f in self.sample.files:
            if f.path.endswith(".py"):
                parts = f.path.split("/")
                if len(parts) > 1:
                    names.add(parts[0])
                elif parts[0] not in ("setup.py",):
                    names.add(parts[0][:-3])
        if self.sample.package_name:
            names.add(self.sample.package_name.replace("-", "_"))
        return names

    def add_flow(self, fl: Flow):
        k = (fl.src, fl.dst)
        old = self._flows.get(k)
        if old is None or fl.confidence > old.confidence:
            self._flows[k] = fl

    def add_indicator(self, kind, value, path, line):
        self.indicators.setdefault((kind, value, path, line), {"kind": kind, "value": value, "file": path, "line": line})

    def _parse(self, path: str, text: str) -> Optional[ast.Module]:
        try:
            return ast.parse(text, filename=path)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
            self.parse_errors[path] = f"{type(e).__name__}: {str(e)[:120]}"
            return None

    def run(self) -> PackageFacts:
        trees: Dict[str, ast.Module] = {}
        pth_lines: List[Tuple[str, int, str]] = []
        n_files = 0
        for f in self.sample.files:
            if f.path.endswith((".py", ".pyw")):
                n_files += 1
                if f.n_lines > MAX_AST_LINES or len(f.text) > MAX_AST_CHARS:
                    self.parse_errors[f.path] = "too large for AST analysis (regex scan only)"
                    self.limit("large_file_regex_only")
                    self._regex_fallback(f.path, f.text)
                    continue
                t = self._parse(f.path, f.text)
                if t is not None:
                    trees[f.path] = t
                else:
                    self._regex_fallback(f.path, f.text)
            elif f.path.endswith(".pth"):
                for i, line in enumerate(f.text.splitlines(), 1):
                    if line.lstrip().startswith(("import ", "import\t")):
                        pth_lines.append((f.path, i, line))
        analyzers = {p: _FileAnalyzer(self, p, t) for p, t in trees.items()}
        for p, fa in analyzers.items():
            install_cls = fa.install_cmd_classes() if posixpath.basename(p) == "setup.py" else set()
            for q, fn, cls, _ in fa.functions():
                key = f"{p}::{q}"
                self.functions[key] = FunctionInfo(key, p, q, fn.lineno, _end(fn),
                                                   [a.arg for a in fn.args.args],
                                                   is_install_cmd=bool(cls and cls.split(".")[-1] in install_cls))
                self.by_short[q.split(".")[-1]].append(key)
        for _round in range(2):                                   # round 2 sees round-1 function summaries
            for fa in analyzers.values():
                fa.run()
        for path, i, line in pth_lines:                            # .pth "import ..." lines run at interpreter start
            n_files += 1
            t = self._parse(f"{path}#L{i}", line.strip())
            if t is not None:
                for node in ast.walk(t):
                    if hasattr(node, "lineno"):
                        node.lineno = i; node.end_lineno = i
                fa = _FileAnalyzer(self, path, t); fa.run()
            self.calls[f"{path}:{i}:hook"] = CallSite(f"{path}:{i}:hook", path, i, i, 0, "pth.import", line.strip()[:80],
                                                      ["install_hook"], f"{path}::<module>", kind="hook", context="install")
        self._assign_contexts(analyzers)
        facts = PackageFacts(sample_id=self.sample.sample_id, calls=self.calls, flows=list(self._flows.values()),
                             functions=self.functions, modules=sorted(trees), parse_errors=self.parse_errors,
                             local_calls=sorted(self.local_calls_set), indicators=list(self.indicators.values()),
                             n_files=n_files, dispatch_sites=[self.dispatch[k] for k in sorted(self.dispatch)],
                             limits_hit=list(self.limits))
        return facts

    def _assign_contexts(self, analyzers):
        roots: Dict[str, str] = {}
        for p in analyzers:
            b = posixpath.basename(p)
            if b == "setup.py":
                roots[f"{p}::<module>"] = "install"
            elif b == "__init__.py" and p.count("/") <= 1:
                roots[f"{p}::<module>"] = "import"
        for k, fi in self.functions.items():
            if fi.is_install_cmd:
                roots[k] = "install"
        callees: Dict[str, Set[str]] = defaultdict(set)
        for caller, callee, _f, _l in self.local_calls_set:
            callees[caller].add(callee)
        ctx: Dict[str, str] = {}
        rank = {"install": 2, "import": 1}
        for root, c in sorted(roots.items(), key=lambda kv: -rank[kv[1]]):
            q = deque([(root, 0)])
            while q:
                k, d = q.popleft()
                if k in ctx and rank[ctx[k]] >= rank[c]:
                    continue
                ctx[k] = c
                if d < 4:
                    for nk in callees.get(k, ()):
                        q.append((nk, d + 1))
        for s in self.calls.values():
            if s.kind == "hook":
                continue
            b = posixpath.basename(s.file)
            if s.file.endswith(".pth"):
                s.context = "install"
            elif s.kind == "regex" and s.scope.endswith("::<module>") and b == "setup.py":
                s.context = "install"
            elif s.kind == "regex" and s.scope.endswith("::<module>") and b == "__init__.py":
                s.context = "import"
            else:
                s.context = ctx.get(s.scope, "runtime")
        # install hook pseudo-sites: setup.py executing sensitive code / custom install commands
        for k, fi in self.functions.items():
            if fi.is_install_cmd:
                cid = f"{fi.file}:{fi.line}:hook"
                self.calls[cid] = CallSite(cid, fi.file, fi.line, fi.end_line, 0, "setuptools.cmdclass",
                                           fi.qualname, ["install_hook"], k, kind="hook", context="install")
        for s in list(self.calls.values()):
            if s.context == "install" and s.kind != "hook" and posixpath.basename(s.file) == "setup.py" \
                    and set(s.behaviors) & set(T.CRITICAL + T.NETWORK + ("env_access", "system_info")):
                cid = f"{s.file}:0:hook"
                if cid not in self.calls:
                    self.calls[cid] = CallSite(cid, s.file, s.line, s.line, 0, "setup.py.toplevel",
                                               "setup.py executes code at install time", ["install_hook"],
                                               s.scope, kind="hook", context="install")

    # ---- python2 / obfuscated files ----------------------------------------------------------------
    _CALL_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")
    _IMP_RE = re.compile(r"^\s*import\s+([\w\.]+)(?:\s+as\s+(\w+))?")
    _FROM_RE = re.compile(r"^\s*from\s+([\w\.]+)\s+import\s+([\w\s,]+)")
    _STR_RE = re.compile(r"'([^'\n]{6,400})'|\"([^\"\n]{6,400})\"")

    def _regex_fallback(self, path: str, text: str):
        alias: Dict[str, str] = {}
        lines = text.splitlines()
        for line in lines:
            m = self._IMP_RE.match(line)
            if m:
                alias[m.group(2) or m.group(1).split(".")[0]] = m.group(1) if m.group(2) else m.group(1).split(".")[0]
            m = self._FROM_RE.match(line)
            if m:
                for nm in m.group(2).split(","):
                    nm = nm.strip().split(" as ")
                    if nm[0]:
                        alias[nm[-1].strip()] = f"{m.group(1)}.{nm[0].strip()}"
        scope = f"{path}::<module>"
        for i, line in enumerate(lines[:20000], 1):
            if len(line) > MAX_REGEX_LINE:
                line = line[:MAX_REGEX_LINE]
                self.limit("long_line_truncated")
            for m in self._CALL_RE.finditer(line):
                name = m.group(1); head, _, rest = name.partition(".")
                api = (alias[head] + ("." + rest if rest else "")) if head in alias else (
                    f"builtins.{name}" if name in BUILTIN_NAMES else name)
                bs = list(T.behaviors_of(api))
                if api in OPEN_APIS:
                    bs.append("file_read")
                if bs and api not in T.TRANSPARENT:
                    cid = f"{path}:{i}:{m.start()}"
                    self.calls[cid] = CallSite(cid, path, i, i, m.start(), api, name, bs, scope, kind="regex",
                                               extractor="regex", confidence=0.3)
            if "os.environ" in line or "getenv" in line:
                cid = f"{path}:{i}:env"
                self.calls.setdefault(cid, CallSite(cid, path, i, i, 0, "os.environ", "os.environ", ["env_access"],
                                                    scope, kind="regex", extractor="regex", confidence=0.3))
            for m in self._STR_RE.finditer(line):
                for kind, val in T.string_indicators(m.group(1) or m.group(2)):
                    self.add_indicator(kind, val, path, i)


def analyze_package(sample) -> PackageFacts:
    """Static facts for one Sample (or its public view)."""
    return PackageAnalyzer(sample).run()
