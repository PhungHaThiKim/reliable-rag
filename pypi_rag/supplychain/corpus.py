"""Render an Org into documents: the ONLY source of truth for both the text RAG index and the KG.

Document ids (one entity or one record per document):
  svc:<service>            service manifest (team, tier, exposure, PII, status, calls, Python dependencies)
  pkg:<package>@<version>  package metadata (maintainer, pinned requirements, optional extras)
  adv:<ADV id>             advisory (package, affected versions, summary)
  pol:<POL id>-v<n>        security policy versions (status, supersedes, patch windows, bans, exemptions)
  pol:<POL-IAM id>         token-lifetime policies (two current documents overlap on purpose)
  adr:<ADR id>             exception records
  inc:<INC id>             incident reports
  team:<team>              team page          api:<service>   API notes (noise)
Lines are stable: the KG keeps `doc_id:line` provenance on every edge.
"""
from __future__ import annotations

import random
from typing import List

from ..knowledge.documents import KnowledgeDoc
from .org import Org, spec_text


def _doc(doc_id: str, st: str, title: str, lines: List[str], seed: int, entity: str) -> KnowledgeDoc:
    return KnowledgeDoc(doc_id=doc_id, source_type=st, title=title, text="\n".join(lines),
                        provenance={"origin": "synthetic_org", "seed": seed, "entity": entity})


def _dep_line(d, service_level: bool) -> str:
    ver = f"=={d.version}" if d.version else " (unpinned: version resolved at build time, not recorded)"
    bits = []
    if service_level:
        bits.append(d.scope)
        if d.extras:
            bits.append("extras: " + ", ".join(d.extras))
    elif d.extra:
        bits.append(f"only with extra: {d.extra}")
    return f"  - {d.name}{ver}" + (f" ({'; '.join(bits)})" if bits else "")


def render_corpus(org: Org) -> List[KnowledgeDoc]:
    docs: List[KnowledgeDoc] = []
    sd = org.seed
    for s in org.services.values():
        lines = [f"service: {s.name}", f"team: {s.team}", f"tier: {s.tier}",
                 f"internet_facing: {'yes' if s.internet_facing else 'no'}",
                 f"handles_pii: {'yes' if s.handles_pii else 'no'}", f"status: {s.status}",
                 f"runtime: {s.runtime}", "calls: " + (", ".join(s.calls) if s.calls else "none"),
                 "dependencies (Python, from the lockfile of the main branch):"]
        lines += [_dep_line(d, True) for d in s.deps]
        if s.plugins_from_config:
            lines.append("plugins: additional plugin packages are loaded from runtime configuration that is not "
                         "stored in this repository")
        if s.status != "active":
            lines.append("note: this service is decommissioned and no longer deployed")
        docs.append(_doc(f"svc:{s.name}", "service_manifest", f"Service manifest {s.name}", lines, sd, s.name))
    for p in org.packages.values():
        for v, deps in p.versions.items():
            lines = [f"package: {p.name}", f"version: {v}", f"maintainer: {p.maintainer}",
                     "requires:" if deps else "requires: none"] + [_dep_line(d, False) for d in deps]
            docs.append(_doc(f"pkg:{p.name}@{v}", "package_metadata", f"Package {p.name} {v}", lines, sd, p.name))
    for a in org.advisories:
        lines = [f"advisory: {a.adv_id}", f"package: {a.package}", f"affected versions: {spec_text(a.spec)}",
                 f"published: {a.published}", "status: open", f"summary: {a.summary}",
                 "remediation: remove or upgrade the package and rotate any credentials that were present"]
        docs.append(_doc(f"adv:{a.adv_id}", "advisory", f"Security advisory {a.adv_id} ({a.package})", lines, sd, a.adv_id))
    for p in org.policies:
        lines = [f"policy: {p.pol_id} version {p.version}", f"status: {p.status}", f"effective: {p.effective}"]
        if p.status == "superseded":
            lines.append(f"superseded by: {p.pol_id} version {p.version + 1}")
        if p.supersedes:
            lines.append(f"supersedes: {p.supersedes}")
        lines.append("section 2 scope:")
        lines.append("  2.1 applies to every Python dependency of every active service")
        lines.append("  2.3 dependencies declared with scope dev are not deployed; bans and patch windows do not apply to them"
                     if p.dev_exempt else "  2.3 the rules apply to runtime and dev dependencies alike")
        lines.append("section 3 patch windows (a vulnerable runtime dependency must be fixed within; first matching class wins):")
        lines += [f"  3.{i + 1} {cls}: {h} hours" for i, (cls, h) in enumerate(p.windows)]
        lines.append("section 4 banned packages:")
        lines += [f"  4.{i + 1} {pk} {spec_text(sp)} is banned ({why})" for i, (pk, sp, why) in enumerate(p.bans)]
        if p.advisory_ban:
            lines.append(f"  4.{len(p.bans) + 1} every package version listed in an open ACME security advisory is banned")
        lines.append("section 5 exceptions: a ban can be waived for one service by an approved ADR until its expiry date"
                     if p.status == "current" else "section 5 exceptions: none")
        docs.append(_doc(f"pol:{p.pol_id}-v{p.version}", "policy", f"Security policy {p.pol_id} version {p.version}",
                         lines, sd, p.pol_id))
    for t in org.token_policies:
        lines = [f"policy: {t.pol_id}", f"status: {t.status}", f"effective: {t.effective}", f"owner: team {t.owner}",
                 "topic: lifetime of service API tokens"]
        lines += [f"  rule {i + 1}: tokens of {cls} services expire after at most {d} days" for i, (cls, d) in enumerate(t.rules)]
        docs.append(_doc(f"pol:{t.pol_id}", "policy", f"Token policy {t.pol_id}", lines, sd, t.pol_id))
    for r in org.adrs:
        lines = [f"record: {r.adr_id}", f"decided: {r.decided}", "type: exception to POL-SEC-001 section 4",
                 f"service: {r.service}", f"package: {r.package} {spec_text(r.spec)}", f"valid until: {r.until}",
                 f"approved by: {r.approver}", f"reason: {r.reason}"]
        docs.append(_doc(f"adr:{r.adr_id}", "adr", f"Exception record {r.adr_id}", lines, sd, r.adr_id))
    for i in org.incidents:
        lines = [f"incident: {i.inc_id}", f"date: {i.date}", "services: " + ", ".join(i.services), f"summary: {i.summary}"]
        docs.append(_doc(f"inc:{i.inc_id}", "incident", f"Incident {i.inc_id}", lines, sd, i.inc_id))
    for t in org.teams:
        owned = sorted(s.name for s in org.services.values() if s.team == t.name)
        lines = [f"team: {t.name}", f"lead: {t.lead}", f"exception approver: {t.approver}", "services: " + ", ".join(owned)]
        docs.append(_doc(f"team:{t.name}", "team", f"Team {t.name}", lines, sd, t.name))
    rng = random.Random(org.seed + 7)
    for s in rng.sample(sorted(org.services), 15):
        eps = rng.sample(["/health", "/v1/items", "/v1/orders", "/v1/users/{id}", "/v1/search", "/v1/export",
                          "/v2/events", "/admin/reload"], 3)
        lines = [f"service: {s}", "API notes:"] + [f"  GET {e} returns JSON; rate limit {rng.choice([50, 100, 500])} rps" for e in eps]
        docs.append(_doc(f"api:{s}", "api_doc", f"API notes {s}", lines, sd, s))
    return docs


def entity_names(org: Org) -> List[str]:
    """Every named entity that exists in the corpus (services, packages, advisories, policies, records, teams)."""
    out = list(org.services) + list(org.packages) + [a.adv_id for a in org.advisories]
    out += sorted({p.pol_id for p in org.policies}) + [t.pol_id for t in org.token_policies]
    out += [r.adr_id for r in org.adrs] + [i.inc_id for i in org.incidents] + [t.name for t in org.teams]
    return out
