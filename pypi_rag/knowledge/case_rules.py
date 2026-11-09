"""Knowledge records of the test-case document (C03, C04), fixed BEFORE the case run and independent of the
case labels: one documented exfiltration pattern, one malicious rule, two benign counterexamples and one
general rule. Provenance: synthetic_expert_reviewed_rule. Used only by the KB of the cases dataset
(`build-kb --sources builtin case_rules`), never by the main benchmarks."""
from __future__ import annotations

from typing import List

from .documents import KnowledgeDoc

_RECORDS = [
    ("PATTERN-EXFIL-001", "rule", "Documented pattern: potential data exfiltration",
     "category: potential_data_exfiltration. source_behavior: read sensitive configuration or credentials. "
     "sink_behavior: transmit to an external destination. required_condition: evidence that the sensitive data "
     "actually reaches the sink.", ["file_read", "sensitive_data", "network_send", "data_exfiltration"], "malicious_indicator"),
    ("MAL-001", "rule", "Unauthorized sensitive-data transmission",
     "Sending credentials, tokens or private files to an external host without the user's request or consent is "
     "malicious, in particular when it happens at install or import time.",
     ["sensitive_data", "network_send", "data_exfiltration"], "malicious_indicator"),
    ("BEN-001", "benign_rule", "Benign: user-authorized report upload",
     "Reading a report or document chosen by the user and uploading it to the service the user configured, behind an "
     "explicit approval flag, is normal reporting behavior.", ["file_read", "network_send"], "benign_indicator"),
    ("BEN-002", "benign_rule", "Benign: normal reporting and telemetry",
     "Sending non-sensitive usage data or version information to the package's own endpoint, especially when the "
     "user opted in, is normal telemetry.", ["network_send", "env_access"], "benign_indicator"),
    ("RULE-001", "rule", "API presence alone is insufficient for malicious intent",
     "A file read plus an HTTP request in the same package does not prove malicious intent. Decide from what data is "
     "read, where it goes, whether the user authorized it, and when it runs.", ["file_read", "network_send"], "guidance"),
]


def case_rule_docs() -> List[KnowledgeDoc]:
    return [KnowledgeDoc(doc_id=f"case:{rid}", source_type=st, title=f"{rid} {title}", text=text, behaviors=list(beh), apis=[],
                         provenance={"origin": "synthetic_expert_reviewed_rule", "record_id": rid, "split": "external",
                                     "label_hint": hint})
            for rid, st, title, text, beh, hint in _RECORDS]
