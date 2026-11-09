"""Task B: a deterministic synthetic organization ("ACME") for supply-chain questions.

`make_org(seed)` builds teams, services, Python packages with pinned dependency trees, advisories for
malicious packages (typosquats and compromised versions), a security policy with a superseded and a
current version, exception records (ADRs), token-lifetime policies (one deliberate conflict), incidents.
Everything is decided by `seed`. The org is rendered into documents (corpus.py); questions and their
answers are computed from the org itself (questions.py), never from the documents or from a model.

Traps are planted on purpose and apply to every method:
  superseded policy, expired exception, dev-only dependency, optional extra that is not enabled,
  version outside the advisory range, decommissioned service, near-identical names, unpinned dependency,
  configuration-loaded plugins, two current policies that conflict.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

TODAY = "2026-11-01"

PUBLIC = ["requests", "urllib3", "certifi", "idna", "charset-normalizer", "pyyaml", "jinja2", "markupsafe", "click",
          "flask", "werkzeug", "itsdangerous", "django", "sqlalchemy", "psycopg2-binary", "redis", "celery", "kombu",
          "boto3", "botocore", "s3transfer", "jmespath", "cryptography", "cffi", "pycparser", "pyjwt", "pydantic",
          "fastapi", "starlette", "uvicorn", "gunicorn", "httpx", "httpcore", "anyio", "sniffio", "h11", "protobuf",
          "grpcio", "numpy", "pandas", "python-dateutil", "pytz", "six", "attrs", "lxml", "pillow", "reportlab", "tqdm",
          "rich", "pygments", "paramiko", "websockets", "aiohttp", "orjson", "structlog", "sentry-sdk",
          "prometheus-client", "tenacity", "tomli", "packaging"]
DEV_TOOLS = ["pytest", "pytest-cov", "coverage", "black", "flake8", "mypy", "tox", "isort", "pre-commit", "sphinx"]
INTERNAL_WORDS = ["core", "auth", "authz", "billing", "ledger", "search", "index", "reports", "telemetry", "metrics",
                  "config", "secrets", "notify", "mailer", "cache", "queue", "events", "storage", "images", "pdf", "geo",
                  "pricing", "catalog", "orders", "cart", "users", "session", "audit", "flags", "schema", "client",
                  "http", "logging", "tracing", "crypto", "tokens", "sync", "export", "importer", "scheduler"]
SERVICE_WORDS = ["checkout", "wallet", "gateway", "profile", "signup", "login", "search-api", "recommender", "inventory",
                 "shipping", "invoices", "payouts", "fraud", "ledger-api", "notifications", "email-relay", "media",
                 "thumbnails", "analytics", "etl", "warehouse", "reporting", "admin", "support", "chatbot", "pricing-api",
                 "catalog-api", "orders-api", "partner-api", "status", "feature-flags", "audit-log", "backoffice",
                 "mobile-bff", "web-bff", "returns", "loyalty", "geo-api", "kyc", "scheduler-api"]
TEAMS = ["payments", "identity", "search", "data-platform", "growth", "infra"]
FIRST = ["An", "Binh", "Chi", "Dung", "Giang", "Hoa", "Khanh", "Linh", "Minh", "Nam", "Phuong", "Quan", "Thao", "Tuan",
         "Alex", "Sam", "Jordan", "Robin", "Kim", "Lee"]
LAST = ["Nguyen", "Tran", "Le", "Pham", "Hoang", "Vu", "Dang", "Bui", "Do", "Ngo", "Smith", "Garcia", "Kowalski", "Okafor"]
EXTRAS = ["pdf", "s3", "cli", "async", "plugins"]
PAYLOADS = ["sends environment variables to a remote host at import time",
            "uploads ~/.ssh and ~/.aws credential files during installation",
            "downloads and executes a second-stage payload from a paste site",
            "posts the hostname, user name and CI tokens to a webhook",
            "installs a startup hook that exfiltrates browser cookies"]


# ---------------------------------------------------------------------------------- versions
def vkey(v: str) -> Tuple[int, ...]:
    xs = [int(x) for x in re.findall(r"\d+", v)[:3]]
    return tuple(xs + [0] * (3 - len(xs)))


def in_spec(v: Optional[str], spec: str) -> bool:
    """True when version `v` satisfies `spec` ('>=1.2.0,<1.3.0', '==2.0.1', '<3.0', '*' = all versions)."""
    if v is None:
        raise ValueError("unpinned version")
    spec = spec.strip()
    if spec in ("*", "all", ">=0"):
        return True
    a = vkey(v)
    for clause in spec.split(","):
        m = re.match(r"\s*(==|!=|>=|<=|<|>)\s*([\d.]+)\s*$", clause)
        if not m:
            raise ValueError(f"bad spec {spec!r}")
        op, b = m.group(1), vkey(m.group(2))
        ok = {"==": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b, "<": a < b, ">": a > b}[op]
        if not ok:
            return False
    return True


def spec_text(spec: str) -> str:
    return "all versions" if spec.strip() in ("*", "all", ">=0") else spec


# ---------------------------------------------------------------------------------- data model
@dataclass
class Dep:
    name: str
    version: Optional[str] = None          # None = unpinned (resolved at build time, not recorded)
    scope: str = "runtime"                 # service level: runtime | dev
    extra: str = ""                        # package level: required only when the parent is installed with this extra
    extras: List[str] = field(default_factory=list)   # service level: extras enabled on this direct dependency


@dataclass
class Package:
    name: str
    kind: str                              # public | dev-tool | internal | typosquat
    layer: int
    versions: Dict[str, List[Dep]]
    maintainer: str = "external (PyPI)"


@dataclass
class Service:
    name: str
    team: str
    tier: int
    internet_facing: bool
    handles_pii: bool
    status: str                            # active | decommissioned
    deps: List[Dep]
    calls: List[str] = field(default_factory=list)
    plugins_from_config: bool = False
    runtime: str = "python3.11"


@dataclass
class Advisory:
    adv_id: str
    package: str
    spec: str
    summary: str
    published: str
    kind: str                              # typosquat | compromised


@dataclass
class PolicyVersion:
    pol_id: str
    version: int
    status: str                            # current | superseded
    effective: str
    supersedes: str
    bans: List[Tuple[str, str, str]]       # (package, spec, reason)
    windows: List[Tuple[str, int]]         # (service class, hours), first match wins
    dev_exempt: bool
    advisory_ban: bool


@dataclass
class TokenPolicy:
    pol_id: str
    status: str
    effective: str
    owner: str
    rules: List[Tuple[str, int]]           # (internet-facing | internal, days)


@dataclass
class ADR:
    adr_id: str
    service: str
    package: str
    spec: str
    until: str
    decided: str
    approver: str
    reason: str


@dataclass
class Incident:
    inc_id: str
    date: str
    services: List[str]
    packages: List[str]
    summary: str


@dataclass
class Team:
    name: str
    lead: str
    approver: str


@dataclass
class Org:
    seed: int
    today: str
    teams: List[Team]
    services: Dict[str, Service]
    packages: Dict[str, Package]
    advisories: List[Advisory]
    policies: List[PolicyVersion]
    token_policies: List[TokenPolicy]
    adrs: List[ADR]
    incidents: List[Incident]
    held_out_services: List[str]           # plausible names that are NOT in the corpus (G4)
    held_out_advisories: List[str]
    ban_targets: Dict[str, str]            # role -> package (B1 / B2 / B3)

    # ---------------------------------------------------------------- ground truth
    def adv(self, adv_id: str) -> Advisory:
        return next(a for a in self.advisories if a.adv_id == adv_id)

    def current_policy(self) -> PolicyVersion:
        return next(p for p in self.policies if p.status == "current")

    def paths(self, svc: str, target: Optional[str] = None, max_depth: int = 8) -> List[Dict]:
        """Every dependency path of `svc` (to `target` if given): {nodes: [(pkg, ver)], dev, extra_ok, unpinned}."""
        s = self.services[svc]
        out: List[Dict] = []

        def walk(nodes, dev, extra_ok, enabled, depth):
            pkg, ver = nodes[-1]
            if target is None or pkg == target:
                out.append({"nodes": list(nodes), "dev": dev, "extra_ok": extra_ok, "unpinned": ver is None})
            if ver is None or depth >= max_depth:
                return
            for c in self.packages[pkg].versions.get(ver, []):
                active = (not c.extra) or (c.extra in enabled)
                walk(nodes + [(c.name, c.version)], dev, extra_ok and active, [], depth + 1)

        for d in s.deps:
            walk([(d.name, d.version)], d.scope == "dev", True, list(d.extras), 1)
        return out

    def affected(self, svc: str, adv_id: str) -> Optional[bool]:
        """True / False / None (cannot be decided: unpinned version on an otherwise deployed path)."""
        a = self.adv(adv_id)
        if self.services[svc].status != "active":
            return False
        unknown = False
        for p in self.paths(svc, a.package):
            if p["dev"] or not p["extra_ok"]:
                continue
            ver = p["nodes"][-1][1]
            if ver is None:
                unknown = True
            elif in_spec(ver, a.spec):
                return True
        return None if unknown else False

    def affected_services(self, adv_id: str) -> List[str]:
        return sorted(s for s in self.services if self.affected(s, adv_id) is True)

    def affecting_paths(self, svc: str, adv_id: str) -> List[List[Tuple[str, str]]]:
        a = self.adv(adv_id)
        return [p["nodes"] for p in self.paths(svc, a.package)
                if not p["dev"] and p["extra_ok"] and p["nodes"][-1][1] is not None and in_spec(p["nodes"][-1][1], a.spec)]

    def window_hours(self, svc: str, pol: Optional[PolicyVersion] = None) -> int:
        pol = pol or self.current_policy()
        s = self.services[svc]
        for cls, hours in pol.windows:
            if _class_match(cls, s):
                return hours
        raise ValueError("no window")

    def banned(self, pkg: str, ver: str, pol: PolicyVersion) -> bool:
        if any(p == pkg and in_spec(ver, sp) for p, sp, _ in pol.bans):
            return True
        return pol.advisory_ban and any(a.package == pkg and in_spec(ver, a.spec) for a in self.advisories)

    def valid_adr(self, svc: str, pkg: str, ver: str) -> Optional[ADR]:
        for r in self.adrs:
            if r.service == svc and r.package == pkg and in_spec(ver, r.spec) and r.until >= self.today:
                return r
        return None

    def allowed(self, svc: str, pkg: str, ver: str, scope: str = "runtime", pol: Optional[PolicyVersion] = None) -> bool:
        pol = pol or self.current_policy()
        if scope == "dev" and pol.dev_exempt:
            return True
        if not self.banned(pkg, ver, pol):
            return True
        return pol.status == "current" and self.valid_adr(svc, pkg, ver) is not None


def _class_match(cls: str, s: Service) -> bool:
    return {"internet-facing and handles PII": s.internet_facing and s.handles_pii,
            "internet-facing": s.internet_facing, "tier 1": s.tier == 1, "tier 2": s.tier == 2,
            "tier 3": s.tier == 3, "all other services": True}[cls]


# ---------------------------------------------------------------------------------- generation
def _person(rng) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _base_versions(rng) -> List[str]:
    major, minor = rng.randint(1, 4), rng.randint(0, 20)
    return [f"{major}.{minor}.0", f"{major}.{minor}.1", f"{major}.{minor + 1}.0"]


def _typo(rng, name: str, taken: set) -> Optional[str]:
    letters = [i for i, c in enumerate(name) if c.isalpha()]
    for _ in range(20):
        kind = rng.choice(["swap", "drop", "double", "suffix"])
        if kind == "swap" and len(letters) > 3:
            i = rng.choice(letters[:-1])
            if name[i + 1].isalpha() and name[i] != name[i + 1]:
                cand = name[:i] + name[i + 1] + name[i] + name[i + 2:]
            else:
                continue
        elif kind == "drop" and len(letters) > 4:
            i = rng.choice(letters[1:])
            cand = name[:i] + name[i + 1:]
        elif kind == "double":
            i = rng.choice(letters)
            cand = name[:i + 1] + name[i] + name[i + 1:]
        else:
            cand = name + rng.choice(["-utils", "-lib", "2", "-py"])
        if cand not in taken and cand != name:
            return cand
    return None


def make_org(seed: int, n_services: int = 30) -> Org:
    rng = random.Random(seed)
    teams = [Team(n, _person(rng), _person(rng)) for n in TEAMS]
    pkgs: Dict[str, Package] = {}

    def add(name, kind, layer, pool, k_range, maintainer="external (PyPI)", extra_p=0.0):
        vs = _base_versions(rng)
        versions: Dict[str, List[Dep]] = {}
        prev: List[Dep] = []
        for i, v in enumerate(vs):
            if i == 0:
                k = min(rng.randint(*k_range), len(pool))
                deps = [Dep(p, rng.choice(list(pkgs[p].versions))) for p in rng.sample(pool, k)]
            else:
                deps = [Dep(d.name, d.version, extra=d.extra) for d in prev]
                if deps and rng.random() < 0.4:
                    j = rng.randrange(len(deps))
                    deps[j] = Dep(deps[j].name, rng.choice(list(pkgs[deps[j].name].versions)), extra=deps[j].extra)
            if pool and rng.random() < extra_p:
                p = rng.choice(pool)
                if p not in {d.name for d in deps}:
                    deps.append(Dep(p, rng.choice(list(pkgs[p].versions)), extra=rng.choice(EXTRAS[:4])))
            versions[v] = deps
            prev = deps
        pkgs[name] = Package(name, kind, layer, versions, maintainer)

    pub = rng.sample(PUBLIC, 50)
    l0, l1 = pub[:25], pub[25:]
    for n in l0:
        add(n, "public", 0, [], (0, 0))
    for n in l1:
        add(n, "public", 1, l0, (0, 2))
    for n in DEV_TOOLS:
        pkgs[n] = Package(n, "dev-tool", 0, {v: [] for v in _base_versions(rng)[:2]})
    words = rng.sample(INTERNAL_WORDS, 32)
    l2, l3, l4 = [f"acme-{w}" for w in words[:12]], [f"acme-{w}" for w in words[12:24]], [f"acme-{w}" for w in words[24:]]
    for n in l2:
        add(n, "internal", 2, l0 + l1, (1, 3), maintainer=f"team {rng.choice(TEAMS)}")
    for n in l3:
        add(n, "internal", 3, l2 + l1, (1, 3), maintainer=f"team {rng.choice(TEAMS)}", extra_p=0.25)
    for n in l4:
        add(n, "internal", 4, l3 + l2, (1, 3), maintainer=f"team {rng.choice(TEAMS)}", extra_p=0.25)

    # ---- malicious packages: 4 typosquats pulled in by the latest version of an internal package,
    #      3 compromised releases (middle version of a real package)
    taken = set(pkgs) | set(PUBLIC) | set(DEV_TOOLS)
    advisories: List[Advisory] = []
    adv_nums = rng.sample(range(11, 99), 9)
    carriers = rng.sample(l2, 4)
    for j, orig in enumerate(rng.sample(l0 + l1, 4)):
        t = _typo(rng, orig, taken)
        taken.add(t)
        tv = pkgs[orig].versions and list(pkgs[orig].versions)[-1]
        pkgs[t] = Package(t, "typosquat", 0, {tv: []}, maintainer=f"unknown uploader ({_person(rng)})")
        car = carriers[j]
        latest = list(pkgs[car].versions)[-1]
        pkgs[car].versions[latest] = pkgs[car].versions[latest] + [Dep(t, tv)]
        advisories.append(Advisory(f"ADV-2026-0{adv_nums[j]:02d}", t, "*",
                                   f"Typosquat of {orig}. The package {rng.choice(PAYLOADS)}.",
                                   f"2026-{rng.randint(5, 10):02d}-{rng.randint(1, 28):02d}", "typosquat"))
    comp = rng.sample([p for p in l0 if p not in carriers], 2) + rng.sample([p for p in l2 if p not in carriers], 1)
    for j, p in enumerate(comp):
        vs = list(pkgs[p].versions)
        spec = f"=={vs[1]}" if j % 2 == 0 else f">={vs[1]},<{vs[2]}"
        advisories.append(Advisory(f"ADV-2026-0{adv_nums[4 + j]:02d}", p, spec,
                                   f"A compromised maintainer account published {p} {vs[1]}; this release "
                                   f"{rng.choice(PAYLOADS)}. Other releases are not affected.",
                                   f"2026-{rng.randint(5, 10):02d}-{rng.randint(1, 28):02d}", "compromised"))
    held_out_adv = [f"ADV-2026-0{n:02d}" for n in adv_nums[7:]]

    # ---- services
    names = rng.sample(SERVICE_WORDS, n_services + 4)
    held_out_svc = [f"svc-{w}" for w in names[n_services + 2:]]
    services: Dict[str, Service] = {}
    for i, w in enumerate(names[:n_services + 2]):
        deps: List[Dep] = []
        for p in rng.sample(l3 + l4, rng.randint(2, 3)) + rng.sample(l2, rng.randint(1, 2)) + rng.sample(l1, rng.randint(0, 2)):
            ver = rng.choice(list(pkgs[p].versions))
            ex = sorted({c.extra for c in pkgs[p].versions[ver] if c.extra})
            deps.append(Dep(p, ver, extras=[e for e in ex if rng.random() < 0.4]))
        for p in rng.sample(DEV_TOOLS, rng.randint(1, 3)):
            deps.append(Dep(p, rng.choice(list(pkgs[p].versions)), scope="dev"))
        services[f"svc-{w}"] = Service(f"svc-{w}", rng.choice(TEAMS), rng.choice([1, 1, 2, 2, 2, 3]),
                                       rng.random() < 0.4, rng.random() < 0.4,
                                       "decommissioned" if i >= n_services else "active", deps)
    snames = list(services)
    for s in services.values():
        s.calls = sorted(rng.sample([x for x in snames if x != s.name], rng.randint(0, 3)))

    org = Org(seed, TODAY, teams, services, pkgs, advisories, [], [], [], [], held_out_svc, held_out_adv, {})
    active = [s for s in snames if services[s].status == "active"]

    def has_direct(svc, pkg):
        return any(d.name == pkg for d in services[svc].deps)

    def carrier_pv(a: Advisory) -> Tuple[str, str]:
        """(package, version) that brings the advisory's package in: the carrier for typosquats."""
        if a.kind == "typosquat":
            car = next(p for p in l2 if any(c.name == a.package for v in pkgs[p].versions.values() for c in v))
            return car, list(pkgs[car].versions)[-1]
        vs = list(pkgs[a.package].versions)
        return a.package, vs[1]

    # every advisory reaches at least two active services at runtime
    for a in advisories:
        tries = 0
        while len(org.affected_services(a.adv_id)) < 2 and tries < 20:
            tries += 1
            s = rng.choice(active)
            p, v = carrier_pv(a)
            if not has_direct(s, p):
                services[s].deps.append(Dep(p, v))
    # trap: dev-only path
    for a in rng.sample(advisories, 3):
        cands = [s for s in active if org.affected(s, a.adv_id) is False and not has_direct(s, carrier_pv(a)[0])]
        if cands:
            p, v = carrier_pv(a)
            services[rng.choice(cands)].deps.append(Dep(p, v, scope="dev"))
    # trap: optional extra (one service enables it, two do not)
    for a in rng.sample(advisories, 2):
        p, v = carrier_pv(a)
        host = rng.choice([x for x in l3 if not any(c.name == p for vv in pkgs[x].versions.values() for c in vv)])
        hv = list(pkgs[host].versions)[-1]
        pkgs[host].versions[hv] = pkgs[host].versions[hv] + [Dep(p, v, extra="plugins")]
        cands = [s for s in active if org.affected(s, a.adv_id) is False and not has_direct(s, host)]
        rng.shuffle(cands)
        for k, s in enumerate(cands[:3]):
            services[s].deps.append(Dep(host, hv, extras=["plugins"] if k == 0 else []))
    # trap: version outside the advisory range
    for a in [x for x in advisories if x.kind == "compromised"]:
        vs = list(pkgs[a.package].versions)
        cands = [s for s in active if org.affected(s, a.adv_id) is False and not has_direct(s, a.package)]
        for s in rng.sample(cands, min(2, len(cands))):
            services[s].deps.append(Dep(a.package, rng.choice([vs[0], vs[2]])))
    # trap: decommissioned services still list affected versions
    for s in [x for x in snames if services[x].status == "decommissioned"]:
        a = rng.choice(advisories)
        p, v = carrier_pv(a)
        if not has_direct(s, p):
            services[s].deps.append(Dep(p, v))
    # trap: unpinned dependency on a compromised leaf package (cannot be decided)
    leafs = [a for a in advisories if a.kind == "compromised" and not pkgs[a.package].versions[list(pkgs[a.package].versions)[0]]]
    for a in leafs[:2]:
        cands = [s for s in active if org.affected(s, a.adv_id) is False and not has_direct(s, a.package)]
        for s in rng.sample(cands, min(2, len(cands))):
            services[s].deps.append(Dep(a.package, None))
    # plugins chosen by runtime configuration
    for s in rng.sample(active, 2):
        services[s].plugins_from_config = True

    # ---- security policy: v1 superseded, v2 current
    free = [p for p in l2 if p not in carriers and p not in comp]
    b1 = rng.choice(free)
    b2, b3 = rng.sample([p for p in l1 if p not in comp], 2)
    v1v, v3v = list(pkgs[b1].versions), list(pkgs[b3].versions)
    windows_v1 = [("internet-facing", 72), ("all other services", 336)]
    windows_v2 = [("internet-facing and handles PII", 24), ("tier 1", 48), ("tier 2", 168), ("tier 3", 720)]
    p1 = PolicyVersion("POL-SEC-001", 1, "superseded", "2025-03-01", "",
                       [(b1, f"<{v1v[1]}", "unmaintained release line"), (b2, "*", "license review pending")],
                       windows_v1, dev_exempt=False, advisory_ban=False)
    p2 = PolicyVersion("POL-SEC-001", 2, "current", "2026-06-01", "POL-SEC-001 v1",
                       [(b1, f"<{v1v[2]}", "unmaintained release line, extended"),
                        (b3, f"<{v3v[2]}", "known vulnerable releases")],
                       windows_v2, dev_exempt=True, advisory_ban=True)
    org.policies = [p1, p2]
    org.ban_targets = {"B1": b1, "B2": b2, "B3": b3}
    org.token_policies = [
        TokenPolicy("POL-IAM-003", "current", "2026-02-01", "identity", [("internet-facing", 30), ("internal", 90)]),
        TokenPolicy("POL-IAM-005", "current", "2026-04-15", "data-platform", [("internal", 180)])]

    # ---- exceptions (ADR): valid and expired
    approver = {t.name: t.approver for t in teams}
    adr_nums = rng.sample(range(100, 999), 4)
    plan = [(b1, f"=={v1v[0]}", "2027-03-31"), (b3, f"=={v3v[0]}", "2026-08-31"),
            (b1, f"=={v1v[1]}", "2027-01-15"), (b3, f"=={v3v[1]}", "2026-12-31")]
    used = set()
    for (p, spec, until), num in zip(plan, adr_nums):
        s = rng.choice([x for x in active if (x, p) not in used])
        used.add((s, p))
        org.adrs.append(ADR(f"ADR-{num}", s, p, spec, until, f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}",
                            approver[services[s].team], rng.choice(["migration blocked by a vendor integration",
                                                                   "replacement scheduled for the next quarter",
                                                                   "risk accepted after review"])))

    # ---- incidents (co-occurring names, mostly unrelated)
    inc_nums = rng.sample(range(100, 999), 4)
    for k, num in enumerate(inc_nums):
        a = rng.choice(advisories)
        s = rng.choice([x for x in active if org.affected(x, a.adv_id) is False])
        if k % 2 == 0:
            summ = (f"{s} reported elevated error rates after a deploy. During triage, the team checked whether "
                    f"{s} is affected by {a.adv_id} ({a.package}); it is not. Root cause: configuration drift. "
                    f"No security impact.")
        else:
            other = rng.choice(l3)
            summ = (f"{s} saw unexpected outbound traffic. The traffic came from a misconfigured health check in "
                    f"{other}. {a.package} was investigated and ruled out for {s}.")
        org.incidents.append(Incident(f"INC-2026-{num}", f"2026-{rng.randint(6, 10):02d}-{rng.randint(1, 28):02d}",
                                      [s], [a.package], summ))
    return org
