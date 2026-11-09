"""Built-in, hand-written background knowledge (origin = builtin:v1).

* BEHAVIOR_PATTERNS: relational motifs over the behavior graph (used by E3 traversal, and rendered as
  text documents for E1/E2 so every method can see the same background knowledge).
* RULES: malicious-behavior explanations and BENIGN counterexamples (plan section 8: "rules, vetted
  behavioral explanations, benign counterexamples").
No rule mentions a dataset package name, so these documents cannot leak test labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BehaviorPattern:
    pattern_id: str
    title: str
    steps: List[List[str]]                 # behavior alternatives per step
    relation: str = "FLOWS_TO"             # FLOWS_TO | PRECEDES | SINGLE
    context: Optional[str] = None          # require the first op in this execution context
    severity: float = 0.8
    description: str = ""
    benign_note: str = ""
    origin: str = "builtin:v1"
    label_hint: str = "malicious_indicator"
    extra: dict = field(default_factory=dict)

    def text(self) -> str:
        chain = " -> ".join("|".join(s) for s in self.steps)
        ctx = f" in {self.context}-time code" if self.context else ""
        rel = {"FLOWS_TO": "data flows", "PRECEDES": "executes before (no confirmed data flow)",
               "SINGLE": "single operation"}[self.relation]
        return (f"Behavior pattern {self.pattern_id}: {self.title}. Structure: {chain} ({rel}{ctx}). "
                f"{self.description} Benign look-alike: {self.benign_note}")


P = BehaviorPattern
BEHAVIOR_PATTERNS: List[BehaviorPattern] = [
    P("P01", "Environment variable exfiltration", [["env_access"], ["network_send"]], severity=0.9,
      description="Values of environment variables (often CI secrets, cloud keys, tokens) are sent to a remote host.",
      benign_note="Reading one documented variable for configuration without sending it anywhere."),
    P("P02", "Encoded environment exfiltration", [["env_access"], ["obfuscation"], ["network_send"]], severity=0.95,
      description="The environment is serialized and encoded (base64/zlib) before being sent, hiding the payload in transit.",
      benign_note="None common; encoding plus sending the whole environment is rare in legitimate code."),
    P("P03", "Credential or secret file exfiltration", [["sensitive_data"], ["network_send"]], severity=0.95,
      description="Credential stores (~/.ssh, ~/.aws/credentials, browser Login Data, wallet files, Discord tokens) are read and uploaded.",
      benign_note="Credential helpers read their OWN config and never upload it to third-party hosts."),
    P("P04", "Local file upload", [["file_read"], ["network_send"]], severity=0.55,
      description="A local file's content is sent to a remote host.",
      benign_note="Upload clients, backup tools and API SDKs upload user-chosen files; check whether the path is user-provided."),
    P("P05", "Host fingerprint beacon", [["system_info"], ["network_send"]], severity=0.8,
      description="Hostname, username, OS or MAC address are collected and sent, typically as a first-stage beacon at install or import.",
      benign_note="Opt-in telemetry sends package version only, behind an explicit opt-in switch."),
    P("P06", "Download and execute code", [["network_download"], ["code_exec"]], severity=0.95,
      description="Remote content is passed to exec/eval/compile: a second-stage payload is fetched and run.",
      benign_note="Plugin loaders import vetted local modules; they do not exec network responses."),
    P("P07", "Download and run shell command", [["network_download"], ["process_exec"]], severity=0.9,
      description="Data received from the network (HTTP body or socket) is executed as a command: remote command execution / C2.",
      benign_note="Installers may download a binary but run it with fixed, local arguments."),
    P("P08", "Download and drop file", [["network_download"], ["file_write"]], severity=0.45,
      description="Remote content is written to disk (dropper stage).",
      benign_note="Dataset and model downloaders legitimately cache files under a user cache directory."),
    P("P09", "Decode and execute", [["obfuscation"], ["code_exec"]], severity=0.9,
      description="An encoded/compressed blob is decoded and executed, hiding the real code from review.",
      benign_note="Rare in legitimate packages; template engines compile code generated from local templates, not decoded blobs."),
    P("P10", "Decode and run shell command", [["obfuscation"], ["process_exec"]], severity=0.85,
      description="An encoded string is decoded and executed as a shell command.",
      benign_note="Build tools run plain, readable commands."),
    P("P11", "Network activity at install time", [["network_send", "network_download", "network_connect"]],
      relation="SINGLE", context="install", severity=0.85,
      description="setup.py / custom install commands / .pth files contact the network while pip installs the package.",
      benign_note="Legitimate builds sometimes download build dependencies, but modern packaging discourages any network use in setup.py."),
    P("P12", "Code or command execution at install time", [["process_exec", "code_exec"]], relation="SINGLE",
      context="install", severity=0.8,
      description="Commands or dynamic code run automatically during installation.",
      benign_note="Build steps that compile extensions or translations with fixed local commands."),
    P("P13", "Network activity at import time", [["network_send", "network_connect"]], relation="SINGLE",
      context="import", severity=0.6,
      description="The package contacts the network as soon as it is imported (top-level __init__ code).",
      benign_note="Libraries normally defer network calls until the user calls an API."),
    P("P14", "Persistence", [["persistence"]], relation="SINGLE", severity=0.85,
      description="Writes to shell rc files, crontab, autostart folders or registry Run keys so code survives reboot.",
      benign_note="Shell-completion installers may append to rc files, but only on explicit user command."),
    P("P15", "Surveillance data exfiltration", [["surveillance"], ["network_send"]], severity=0.9,
      description="Clipboard, keystrokes or screenshots are captured and sent out.",
      benign_note="Clipboard utilities copy/paste locally without network transfer."),
    P("P16", "Environment read before network send (unconfirmed flow)", [["env_access"], ["network_send"]],
      relation="PRECEDES", severity=0.35,
      description="Environment access happens before a network send in the same function but no data flow is confirmed.",
      benign_note="Opt-in telemetry and API clients read a switch or an API key and then call their own documented API."),
]

RULES = [
    # (id, title, behaviors, text, label_hint)
    ("R01", "Install-time code in setup.py", ["install_hook"],
     "setup.py runs with the user's privileges when pip builds a source distribution. Top-level statements other than the "
     "setup() call, custom cmdclass install/develop/egg_info commands, and .pth files with import lines are the most "
     "common place to hide malicious code in PyPI packages.", "malicious_indicator"),
    ("R02", "Typosquatting with a malicious payload", ["install_hook", "network_send"],
     "Malicious packages often copy the name of a popular package with a small typo and keep its description, but add "
     "install-time code that contacts a remote host. The package code itself is often tiny.", "malicious_indicator"),
    ("R03", "Credential stealers", ["sensitive_data", "file_read", "network_send", "data_exfiltration"],
     "Stealers enumerate browser profiles (Login Data, Cookies, Local State), Discord leveldb tokens, crypto wallets and "
     "~/.ssh or ~/.aws files, then upload them to a webhook (Discord, Telegram) or an HTTP endpoint.", "malicious_indicator"),
    ("R04", "Obfuscated payloads", ["obfuscation", "code_exec"],
     "exec(base64.b64decode(...)), exec(zlib.decompress(...)), marshal.loads, long high-entropy strings, chr() "
     "concatenation and __import__('os').system are typical obfuscation of a second-stage payload.", "malicious_indicator"),
    ("R05", "Droppers and remote execution", ["network_download", "file_write", "process_exec", "download_exec"],
     "Droppers download an executable or script, write it to a temp or startup folder and execute it with subprocess or "
     "os.system; C2 clients read commands from a socket and run them.", "malicious_indicator"),
    ("R06", "Reverse shells", ["network_connect", "process_exec"],
     "A socket connection to a hard-coded IP whose file descriptors are attached to a shell process gives an attacker "
     "interactive access.", "malicious_indicator"),
    ("R07", "Exfiltration endpoints", ["network_send"],
     "Hard-coded webhook URLs (discord.com/api/webhooks, api.telegram.org/bot), paste sites, request bins, ngrok or raw "
     "IP addresses in install-time code are strong indicators of exfiltration.", "malicious_indicator"),
    ("B01", "Benign: environment variables for configuration", ["env_access"],
     "Reading os.environ / os.getenv for settings such as DEBUG, LOG_LEVEL or an API key that is then used to call the "
     "package's own documented service is normal. It becomes suspicious only when the values are sent to a third-party "
     "host or the whole environment is collected.", "benign_indicator"),
    ("B02", "Benign: HTTP client libraries", ["network_send", "network_download"],
     "SDKs and API clients call requests/httpx with user-provided URLs and payloads inside functions the user invokes. "
     "Sending is not malicious by itself; look for data read from the local machine flowing into the request.", "benign_indicator"),
    ("B03", "Benign: subprocess with fixed commands", ["process_exec"],
     "Developer tools run git, pip, compilers or linters with constant argument lists. Commands built from network data "
     "or decoded strings are the malicious variant.", "benign_indicator"),
    ("B04", "Benign: base64 for data handling", ["obfuscation"],
     "base64 is used for data URIs, images, HTTP basic auth headers and binary serialization. Only decoding followed by "
     "execution, or encoding of collected local data before sending it, is suspicious.", "benign_indicator"),
    ("B05", "Benign: dataset and model downloaders", ["network_download", "file_write"],
     "Machine-learning and data libraries download datasets or weights into a cache directory when the user asks for "
     "them. They do not execute the downloaded content.", "benign_indicator"),
    ("B06", "Benign: custom build commands", ["install_hook", "file_read"],
     "Custom build_py/build_ext commands compile translations, generate version files or build C extensions with local "
     "inputs. They are install-time code without network or credential access.", "benign_indicator"),
    ("B07", "Benign: template engines and code generation", ["code_exec"],
     "Template engines and ORMs compile and exec code that they generate from local templates or model definitions. "
     "exec of locally generated code is not the same as exec of decoded or downloaded content.", "benign_indicator"),
]
