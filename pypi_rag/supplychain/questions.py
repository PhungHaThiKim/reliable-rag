"""Task B questions with gold answers and gold evidence, computed from the org (never from a model).

Groups (every method answers every question; the groups are NOT matched to methods):
  G1 code understanding: the answer is in the code shown with the question (templates + cases C01-C05)
  G2 internal policy / exceptions: the answer needs ACME documents (current vs superseded policy, ADRs)
  G3 multi-hop dependency impact: the answer needs several documents chained (service -> package -> ...)
  G4 insufficient evidence: the corpus or the code does not decide the question; gold = insufficient_evidence
Each question has a `trap` tag (superseded_policy, dev_exempt, adr_expired, extra_not_enabled, ...) so errors
can be broken down by cause, and an `entity` (cluster for the grouped bootstrap).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..data.cases import CASES
from .org import Org, in_spec

INSUFFICIENT = "insufficient_evidence"


def _q(group, qtype, question, answer_type, gold, evidence, trap, entity, code=None, **meta) -> Dict:
    return {"group": group, "qtype": qtype, "question": question, "answer_type": answer_type, "gold": gold,
            "answerable": gold != INSUFFICIENT, "gold_evidence": sorted(set(evidence)), "trap": trap, "entity": entity,
            "code": code, "meta": meta}


# ---------------------------------------------------------------------------------- G1 / G4 code
_SRC = {"env": ("import os", 'os.environ.get("{var}", "")'), "file": ("", 'open("{path}").read()'),
        "host": ("import socket", "socket.gethostname()"), "user": ("import getpass", "getpass.getuser()")}
_SINK = {"post": ("import requests", 'requests.post("https://{host}/{ep}", data={x}, timeout=10)', True),
         "get": ("import requests", 'requests.get("https://{host}/{ep}", params={{"v": {x}}}, timeout=10)', True),
         "urlopen": ("import urllib.request", 'urllib.request.urlopen("https://{host}/{ep}", data={x}.encode())', True),
         "log": ("import logging", 'logging.getLogger(__name__).info("value %s", {x})', False),
         "write": ("", 'open("{out}", "w").write({x})', False)}
_TR = {"none": ("", "{x}"), "b64": ("import base64", "base64.b64encode({x}.encode()).decode()"),
       "json": ("import json", 'json.dumps({{"value": {x}}})')}
_VARS = ["API_TOKEN", "AWS_SECRET_ACCESS_KEY", "CI_JOB_TOKEN", "DATABASE_URL", "SENTRY_DSN", "GITHUB_TOKEN", "HOME"]
_HOSTS = ["stats.example.org", "hooks.example.net", "upload.example.invalid", "telemetry.example.com",
          "collector.example.org", "api.example.net"]
_PATHS = ["config/settings.ini", "~/.aws/credentials", "data/report.csv", "~/.ssh/id_rsa", "/etc/hostname"]
_FN = ["sync", "report", "collect", "refresh", "export", "heartbeat", "backup"]


def code_snippet(rng: random.Random) -> Dict:
    src, sink, tr = rng.choice(list(_SRC)), rng.choice(list(_SINK)), rng.choice(list(_TR))
    shape = rng.choice(["inline", "split", "unconnected"])
    var, host, path = rng.choice(_VARS), rng.choice(_HOSTS), rng.choice(_PATHS)
    ep, out, fn = rng.choice(["v1/ingest", "upload", "events", "collect"]), "out/cache.txt", rng.choice(_FN)
    s_imp, s_expr = _SRC[src]; k_imp, k_tpl, net = _SINK[sink]; t_imp, t_tpl = _TR[tr]
    s_expr = s_expr.format(var=var, path=path)
    imports = sorted({i for i in (s_imp, k_imp, t_imp) if i})
    if shape == "inline":
        body = [f"def {fn}():", f"    value = {s_expr}", f"    payload = {t_tpl.format(x='value')}",
                f"    {k_tpl.format(host=host, ep=ep, x='payload', out=out)}"]
    elif shape == "split":
        body = ["def collect_value():", f"    return {s_expr}", "", "def prepare(value):",
                f"    return {t_tpl.format(x='value')}", "", "def send(payload):",
                f"    {k_tpl.format(host=host, ep=ep, x='payload', out=out)}", "", f"def {fn}():",
                "    send(prepare(collect_value()))"]
    else:
        body = ["CACHE = {}", "", f"def {fn}():", f"    CACHE['value'] = {s_expr}", '    status = "ok"',
                f"    {k_tpl.format(host=host, ep=ep, x='status', out=out)}"]
    code = "\n".join(imports + ([""] if imports else []) + body) + "\n"
    return {"code": code, "src": src, "sink": sink, "net": net, "shape": shape, "var": var, "host": host,
            "flow": net and shape != "unconnected", "template": f"{src}-{sink}-{tr}-{shape}"}


def g1_questions(rng: random.Random, n: int) -> List[Dict]:
    out, seen = [], set()
    q_flow = ("Does data read from the local machine (environment variables, files, host or user names) reach a "
              "network destination in this code?")
    for c in CASES[:5]:                                    # C01-C05: flow question with document-based gold
        code = "\n".join(f"# file: {p}\n{t}" for p, t in c["files"].items())
        flow = c["id"] in ("C03", "C04", "C05")
        out.append(_q("G1", "code_flow", q_flow, "yes_no", "yes" if flow else "no", ["code"],
                      "co_occurrence" if c["id"] == "C04" else "none", f"case-{c['id']}", code, case=c["id"]))
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        sn = code_snippet(rng)
        kind = rng.choice(["flow", "flow", "host", "env"])
        if kind == "host" and not sn["net"]:
            continue
        if kind == "env" and sn["src"] != "env":
            continue
        key = (sn["code"], kind)
        if key in seen:
            continue
        seen.add(key)
        trap = "co_occurrence" if sn["shape"] == "unconnected" and sn["net"] else ("cross_function" if sn["shape"] == "split" else "none")
        if kind == "flow":
            out.append(_q("G1", "code_flow", q_flow, "yes_no", "yes" if sn["flow"] else "no", ["code"], trap,
                          sn["template"], sn["code"]))
        elif kind == "host":
            out.append(_q("G1", "code_host", "To which host does this code send a request?", "host", sn["host"],
                          ["code"], trap, sn["template"], sn["code"]))
        else:
            out.append(_q("G1", "code_env", "Which environment variables does this code read?", "env_set", [sn["var"]],
                          ["code"], trap, sn["template"], sn["code"]))
    return out[:n]


def g4_code_questions(rng: random.Random, n: int) -> List[Dict]:
    out = []
    c6 = CASES[5]
    out.append(_q("G4", "code_dispatch", "Does this code transmit credentials to an external endpoint during execution?",
                  "yes_no", INSUFFICIENT, ["code"], "dynamic_dispatch", "case-C06",
                  "\n".join(f"# file: {p}\n{t}" for p, t in c6["files"].items()), case="C06"))
    seen = set()
    while len(out) < n:
        var, k1, k2 = rng.choice(_VARS), rng.choice(["module", "plugin", "backend"]), rng.choice(["handler", "entry", "hook"])
        if rng.random() < 0.5:
            code = (f"import importlib\nimport os\n\ndef run(config):\n    module = importlib.import_module(config[\"{k1}\"])\n"
                    f"    handler = getattr(module, config[\"{k2}\"])\n    return handler(os.environ.get(\"{var}\"))\n")
            trap = "dynamic_dispatch"
        else:
            mod, fn = rng.choice(["helpers", "transport", "client"]), rng.choice(["push", "deliver", "ship"])
            code = (f"import os\nfrom .{mod} import {fn}\n\ndef sync():\n    token = os.environ.get(\"{var}\")\n"
                    f"    {fn}(token)\n")
            trap = "missing_code"
        if code in seen:
            continue
        seen.add(code)
        out.append(_q("G4", "code_" + trap, f"Does this code send the value of {var} to a network destination?", "yes_no",
                      INSUFFICIENT, ["code"], trap, f"code-{trap}", code))
    return out[:n]


# ---------------------------------------------------------------------------------- G2 policy
def g2_questions(org: Org, rng: random.Random, n: int) -> List[Dict]:
    cur = org.current_policy()
    old = next(p for p in org.policies if p.status == "superseded")
    pol = f"pol:{cur.pol_id}-v{cur.version}"
    active = sorted(s for s, x in org.services.items() if x.status == "active")
    adr_pairs = {(r.service, r.package) for r in org.adrs}
    cand: Dict[str, List[Dict]] = {k: [] for k in ("ban", "ban_dev", "adr", "adv_ban", "window", "token")}
    for role, pkg in org.ban_targets.items():
        for ver in org.packages[pkg].versions:
            for scope in ("runtime", "dev"):
                for svc in rng.sample(active, 3):
                    if (svc, pkg) in adr_pairs:
                        continue
                    g2, g1 = org.allowed(svc, pkg, ver, scope), org.allowed(svc, pkg, ver, scope, pol=old)
                    trap = "superseded_policy" if g1 != g2 else (
                        "dev_exempt" if scope == "dev" and org.banned(pkg, ver, cur) else "none")
                    what = "a runtime dependency" if scope == "runtime" else "a development-only (dev) dependency"
                    cand["ban_dev" if scope == "dev" else "ban"].append(_q(
                        "G2", "ban_allowed", f"Under ACME's current security policy, may {svc} deploy {pkg}=={ver} as {what}?",
                        "yes_no", "yes" if g2 else "no", [pol], trap, pkg))
    for r in org.adrs:
        ver = next(v for v in org.packages[r.package].versions if in_spec(v, r.spec))
        g = org.allowed(r.service, r.package, ver)
        trap = "adr_exception" if g else "adr_expired"
        cand["adr"].append(_q("G2", "ban_allowed",
                              f"Under ACME's current security policy, may {r.service} deploy {r.package}=={ver} as a runtime dependency?",
                              "yes_no", "yes" if g else "no", [pol, f"adr:{r.adr_id}"], trap, r.package))
        valid = r.until >= org.today
        cand["adr"].append(_q("G2", "exception_valid",
                              f"As of {org.today}, does {r.service} hold a valid exception that allows {r.package}=={ver}?",
                              "yes_no", "yes" if valid else "no", [f"adr:{r.adr_id}"], "adr_expired" if not valid else "none",
                              r.package))
        other = rng.choice([s for s in active if (s, r.package) not in adr_pairs])
        g = org.allowed(other, r.package, ver)
        cand["adr"].append(_q("G2", "ban_allowed",
                              f"Under ACME's current security policy, may {other} deploy {r.package}=={ver} as a runtime dependency?",
                              "yes_no", "yes" if g else "no", [pol, f"adr:{r.adr_id}"], "exception_other_service", r.package))
    for a in org.advisories:
        vs = list(org.packages[a.package].versions)
        for ver in vs:
            svc = rng.choice(active)
            g = org.allowed(svc, a.package, ver)
            trap = "none" if in_spec(ver, a.spec) else "version_out_of_range"
            cand["adv_ban"].append(_q("G2", "ban_allowed",
                                      f"Under ACME's current security policy, may {svc} deploy {a.package}=={ver} as a runtime dependency?",
                                      "yes_no", "yes" if g else "no", [pol, f"adv:{a.adv_id}"], trap, a.package))
    for svc in active:
        h2, h1 = org.window_hours(svc), org.window_hours(svc, old)
        cand["window"].append(_q("G2", "patch_window",
                                 f"Under ACME's current security policy, within how many hours must {svc} fix a vulnerable "
                                 "runtime dependency?", "hours", h2, [pol, f"svc:{svc}"],
                                 "superseded_policy" if h1 != h2 else "none", svc))
        if org.services[svc].internet_facing:
            cand["token"].append(_q("G2", "token_lifetime",
                                    f"What is the maximum lifetime, in days, of a service API token for {svc}?", "days", 30,
                                    ["pol:POL-IAM-003", f"svc:{svc}"], "partial_overlap", svc))
    share = {"ban": 0.25, "ban_dev": 0.12, "adr": 0.18, "adv_ban": 0.15, "window": 0.2, "token": 0.1}
    return _mix(cand, share, n, rng)


# ---------------------------------------------------------------------------------- G3 impact
def _trap_for(org: Org, svc: str, adv_id: str) -> str:
    a = org.adv(adv_id)
    if org.services[svc].status != "active":
        if any(not p["dev"] and p["extra_ok"] and p["nodes"][-1][1] and in_spec(p["nodes"][-1][1], a.spec)
               for p in org.paths(svc, a.package)):
            return "decommissioned"
        return "not_dependent"
    ps = org.paths(svc, a.package)
    ver_ok = [p for p in ps if p["nodes"][-1][1] and in_spec(p["nodes"][-1][1], a.spec)]
    if org.affected(svc, adv_id):
        return "multi_hop" if all(len(p) >= 2 for p in org.affecting_paths(svc, adv_id)) else "none"
    if any(p["dev"] for p in ver_ok):
        return "dev_only"
    if any(not p["extra_ok"] for p in ver_ok):
        return "extra_not_enabled"
    if ps:
        return "version_out_of_range"
    return "not_dependent"


def _path_docs(org: Org, svc: str, adv_id: str) -> List[str]:
    docs = [f"svc:{svc}", f"adv:{adv_id}"]
    for path in org.affecting_paths(svc, adv_id):
        docs += [f"pkg:{p}@{v}" for p, v in path[:-1]]
    return docs


def g3_questions(org: Org, rng: random.Random, n: int) -> List[Dict]:
    cand: Dict[str, List[Dict]] = {k: [] for k in ("set", "filtered", "teams", "is_pos", "is_trap", "is_neg", "path", "version")}
    for a in org.advisories:
        aff = org.affected_services(a.adv_id)
        ev = [d for s in aff for d in _path_docs(org, s, a.adv_id)] + [f"adv:{a.adv_id}"]
        cand["set"].append(_q("G3", "affected_set",
                              f"Which active services deploy, at runtime, a version of {a.package} that is affected by advisory "
                              f"{a.adv_id}? List every such service.", "service_set", aff, ev, "aggregation", a.adv_id))
        hot = [s for s in aff if org.services[s].internet_facing and org.services[s].handles_pii]
        cand["filtered"].append(_q("G3", "affected_filtered",
                                   f"Which active services that are internet-facing and handle PII are affected at runtime by "
                                   f"advisory {a.adv_id} ({a.package})?", "service_set", hot, ev, "aggregation", a.adv_id))
        teams = sorted({org.services[s].team for s in aff})
        cand["teams"].append(_q("G3", "teams_to_notify",
                                f"Which teams own an active service that is affected at runtime by advisory {a.adv_id} "
                                f"({a.package})?", "team_set", teams, ev + [f"team:{t}" for t in teams], "aggregation", a.adv_id))
        for svc in sorted(org.services):
            aff_s = org.affected(svc, a.adv_id)
            if aff_s is None:
                continue
            trap = _trap_for(org, svc, a.adv_id)
            key = "is_pos" if aff_s else ("is_neg" if trap == "not_dependent" else "is_trap")
            cand[key].append(_q("G3", "is_affected",
                                f"Is {svc} currently exposed at runtime to advisory {a.adv_id} ({a.package})?", "yes_no",
                                "yes" if aff_s else "no", _path_docs(org, svc, a.adv_id) if aff_s else [f"svc:{svc}", f"adv:{a.adv_id}"],
                                trap, a.adv_id))
            if aff_s:
                ps = org.affecting_paths(svc, a.adv_id)
                if len(ps) == 1 and len(ps[0]) >= 2:
                    cand["path"].append(_q("G3", "exposure_path",
                                           f"Through which chain of packages is {svc} exposed to advisory {a.adv_id}? List the "
                                           f"package names in order, starting with the direct dependency of {svc} and ending "
                                           f"with {a.package}.", "package_path", [p for p, _ in ps[0]],
                                           _path_docs(org, svc, a.adv_id), "multi_hop", a.adv_id))
    for svc in sorted(s for s, x in org.services.items() if x.status == "active"):
        found: Dict[str, set] = {}
        for p in org.paths(svc):
            if p["dev"] or not p["extra_ok"] or p["unpinned"] or len(p["nodes"]) < 2:
                continue
            name, ver = p["nodes"][-1]
            found.setdefault(name, set()).add(ver)
        direct = {d.name for d in org.services[svc].deps}
        for name in sorted(found):
            if len(found[name]) == 1 and name not in direct and org.packages[name].kind != "dev-tool":
                ver = next(iter(found[name]))
                path = next(p["nodes"] for p in org.paths(svc, name) if not p["dev"] and p["extra_ok"])
                cand["version"].append(_q("G3", "version_of", f"Which version of {name} does {svc} deploy at runtime?",
                                          "version", ver, [f"svc:{svc}"] + [f"pkg:{x}@{v}" for x, v in path[:-1]],
                                          "multi_hop", svc))
    share = {"set": 0.07, "filtered": 0.07, "teams": 0.07, "is_pos": 0.17, "is_trap": 0.22, "is_neg": 0.08,
             "path": 0.17, "version": 0.15}
    return _mix(cand, share, n, rng)


# ---------------------------------------------------------------------------------- G4 insufficient
def g4_questions(org: Org, rng: random.Random, n: int) -> List[Dict]:
    cand: Dict[str, List[Dict]] = {k: [] for k in ("unknown_svc", "unknown_adv", "unpinned", "plugins", "token", "undoc", "code")}
    for s in org.held_out_services:
        for a in org.advisories:
            cand["unknown_svc"].append(_q("G4", "is_affected", f"Is {s} currently exposed at runtime to advisory {a.adv_id} "
                                          f"({a.package})?", "yes_no", INSUFFICIENT, [], "unknown_entity", s))
    for adv in org.held_out_advisories:
        cand["unknown_adv"].append(_q("G4", "affected_set", f"Which active services are affected at runtime by advisory {adv}? "
                                      "List every such service.", "service_set", INSUFFICIENT, [], "unknown_entity", adv))
        cand["unknown_adv"].append(_q("G4", "teams_to_notify", f"Which teams own an active service that is affected at runtime "
                                      f"by advisory {adv}?", "team_set", INSUFFICIENT, [], "unknown_entity", adv))
    for a in org.advisories:
        for svc in sorted(org.services):
            if org.affected(svc, a.adv_id) is None:
                cand["unpinned"].append(_q("G4", "is_affected", f"Is {svc} currently exposed at runtime to advisory "
                                           f"{a.adv_id} ({a.package})?", "yes_no", INSUFFICIENT, [f"svc:{svc}", f"adv:{a.adv_id}"],
                                           "unpinned", a.adv_id))
                cand["unpinned"].append(_q("G4", "version_of", f"Which version of {a.package} does {svc} deploy at runtime?",
                                           "version", INSUFFICIENT, [f"svc:{svc}"], "unpinned", svc))
    plug = [s for s, x in org.services.items() if x.plugins_from_config]
    for s in plug:
        mine = {p[-1][0] for p in (q["nodes"] for q in org.paths(s))}
        for a in org.advisories:
            if a.package not in mine:
                cand["plugins"].append(_q("G4", "loads_package", f"Does {s} load {a.package} at runtime?", "yes_no",
                                          INSUFFICIENT, [f"svc:{s}"], "config_plugins", s))
    for s, x in sorted(org.services.items()):
        if x.status == "active" and not x.internet_facing:
            cand["token"].append(_q("G4", "token_lifetime", f"What is the maximum lifetime, in days, of a service API token "
                                    f"for {s}?", "days", INSUFFICIENT, ["pol:POL-IAM-003", "pol:POL-IAM-005", f"svc:{s}"],
                                    "conflicting_policies", s))
        if x.status == "active":
            cand["undoc"].append(_q("G4", "undocumented", f"Which Node.js version does {s} run in production?", "version",
                                    INSUFFICIENT, [f"svc:{s}"], "not_documented", s))
            cand["undoc"].append(_q("G4", "undocumented", f"On which date was {s} last deployed to production?", "date",
                                    INSUFFICIENT, [], "not_documented", s))
    cand["code"] = g4_code_questions(rng, max(4, n // 6))
    share = {"unknown_svc": 0.16, "unknown_adv": 0.08, "unpinned": 0.12, "plugins": 0.12, "token": 0.16, "undoc": 0.16,
             "code": 0.2}
    return _mix(cand, share, n, rng)


# ---------------------------------------------------------------------------------- mixing
def _mix(cand: Dict[str, List[Dict]], share: Dict[str, float], n: int, rng: random.Random) -> List[Dict]:
    """Take about share*n questions of each kind (fewer when not available), fill up from the rest, no duplicates."""
    pools = {}
    for k, xs in cand.items():
        seen, uniq = set(), []
        for q in xs:
            if q["question"] + str(q.get("code")) not in seen:
                seen.add(q["question"] + str(q.get("code"))); uniq.append(q)
        rng.shuffle(uniq)
        pools[k] = uniq
    out = []
    for k, sh in share.items():
        take = min(len(pools.get(k, [])), max(1, round(sh * n)))
        out += pools[k][:take]; pools[k] = pools[k][take:]
    rest = [q for k in share for q in pools.get(k, [])]
    rng.shuffle(rest)
    out += rest[: max(0, n - len(out))]
    return out[:n]


def make_questions(org: Org, n_per_group: int = 100, seed: Optional[int] = None) -> List[Dict]:
    rng = random.Random((org.seed if seed is None else seed) * 1000 + 17)
    raw = (g1_questions(rng, n_per_group) + g2_questions(org, rng, n_per_group) + g3_questions(org, rng, n_per_group)
           + g4_questions(org, rng, n_per_group))
    qs, seen = [], set()
    for q in raw:                                   # one question text (+ code) appears once in the whole set
        key = q["question"] + str(q.get("code"))
        if key not in seen:
            seen.add(key); qs.append(q)
    for i, q in enumerate(qs):
        q["qid"] = f"s{org.seed}-{q['group']}-{i:04d}"
    return qs
