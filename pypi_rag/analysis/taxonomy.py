"""Behavior taxonomy: canonical API name -> behavior categories, plus string indicators.

Built-in and deliberately small (auditable). `knowledge.taxonomy_adapter` can extend
`API_RULES` with PyGuard's API taxonomy when the upstream repo is available; every
rule keeps its origin so a graph edge can say which taxonomy produced it.
"""
from __future__ import annotations

import fnmatch
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

BEHAVIORS: Dict[str, str] = {
    "network_send": "sends data over the network (HTTP POST/PUT, socket send, SMTP, webhook, query string with data)",
    "network_download": "fetches remote content (HTTP GET, urlopen, urlretrieve)",
    "network_connect": "opens raw sockets or connections",
    "file_read": "reads or enumerates local files",
    "file_write": "writes, copies or creates local files",
    "file_delete": "deletes files or directories",
    "sensitive_data": "touches credentials, keys, tokens, browser or wallet data",
    "env_access": "reads environment variables",
    "system_info": "collects host or user information (hostname, user, OS, MAC, IP)",
    "process_exec": "spawns processes or shell commands",
    "code_exec": "executes dynamically built code (exec, eval, compile, marshal/pickle loads)",
    "obfuscation": "encodes, decodes or compresses data or code (base64, zlib, hex, rot13)",
    "dynamic_import": "imports modules by computed name",
    "native_code": "calls native libraries (ctypes, cffi)",
    "persistence": "autostart, scheduled task, shell rc or registry run-key modification",
    "install_hook": "code that runs automatically at install / startup time (setup.py top level, cmdclass, .pth)",
    "surveillance": "clipboard, keyboard or screen capture",
    # relational behaviors: only produced from FLOWS_TO paths (never from a single call)
    "data_exfiltration": "data from a source (env, file, system info, sensitive data) flows into a network sink",
    "download_exec": "remote content flows into code or process execution",
}
RELATIONAL = ("data_exfiltration", "download_exec")
SOURCE_BEHAVIORS = ("env_access", "file_read", "sensitive_data", "system_info", "surveillance", "network_download")
SINK_BEHAVIORS = ("network_send", "code_exec", "process_exec", "file_write")
TRANSFORM_BEHAVIORS = ("obfuscation",)
CRITICAL = ("network_send", "code_exec", "process_exec", "sensitive_data", "persistence", "install_hook",
            "data_exfiltration", "download_exec", "surveillance")
NETWORK = ("network_send", "network_download", "network_connect")

# (glob on canonical name, behaviors). Order does not matter; all matching rules apply.
API_RULES: List[Tuple[str, Tuple[str, ...], str]] = []


def _r(patterns: Iterable[str], behaviors: Sequence[str], origin: str = "builtin"):
    for p in patterns:
        API_RULES.append((p, tuple(behaviors), origin))


_r(["requests.post", "requests.put", "requests.patch", "requests.Session.post", "requests.Session.put",
    "requests.sessions.Session.post", "httpx.post", "httpx.put", "httpx.Client.post", "httpx.AsyncClient.post",
    "aiohttp.ClientSession.post", "aiohttp.ClientSession.put", "http.client.HTTPConnection.request",
    "http.client.HTTPSConnection.request", "httplib.HTTPConnection.request", "httplib.HTTPSConnection.request",
    "socket.socket.send", "socket.socket.sendall", "socket.socket.sendto", "socket.create_connection.send",
    "socket.create_connection.sendall", "smtplib.SMTP.sendmail",
    "smtplib.SMTP.send_message", "smtplib.SMTP_SSL.sendmail", "smtplib.SMTP_SSL.send_message",
    "ftplib.FTP.storbinary", "ftplib.FTP.storlines", "telebot.TeleBot.send_message", "telebot.TeleBot.send_document",
    "discord_webhook.DiscordWebhook.execute", "dhooks.Webhook.send", "pycurl.Curl.perform",
    "paramiko.SFTPClient.put", "urllib3.PoolManager.request", "urllib3.request"], ["network_send"])
_r(["requests.get", "requests.head", "requests.request", "requests.Session.get", "requests.Session.request",
    "requests.sessions.Session.get", "httpx.get", "httpx.Client.get", "httpx.AsyncClient.get",
    "aiohttp.ClientSession.get", "urllib.request.urlopen", "urllib.request.urlretrieve", "urllib.urlopen",
    "urllib.urlretrieve", "urllib2.urlopen", "wget.download", "http.client.HTTPConnection.getresponse",
    "http.client.HTTPSConnection.getresponse", "socket.socket.recv", "socket.socket.recvfrom", "socket.create_connection.recv"], ["network_download"])
_r(["socket.socket", "socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex",
    "socket.socket.bind", "socket.socket.listen", "http.client.HTTPConnection", "http.client.HTTPSConnection",
    "httplib.HTTPConnection", "httplib.HTTPSConnection", "smtplib.SMTP", "smtplib.SMTP_SSL", "ftplib.FTP",
    "paramiko.SSHClient.connect", "websocket.create_connection", "websockets.connect"], ["network_connect"])
_r(["subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output", "subprocess.Popen",
    "subprocess.getoutput", "subprocess.getstatusoutput", "os.system", "os.popen", "os.popen2", "os.popen3",
    "os.execl", "os.execle", "os.execlp", "os.execv", "os.execve", "os.execvp", "os.execvpe", "os.spawnl",
    "os.spawnlp", "os.spawnv", "os.spawnvp", "os.startfile", "pty.spawn", "commands.getoutput",
    "commands.getstatusoutput", "asyncio.create_subprocess_shell", "asyncio.create_subprocess_exec"],
   ["process_exec"])
_r(["builtins.exec", "builtins.eval", "builtins.compile", "builtins.execfile", "marshal.loads", "pickle.loads",
    "pickle.load", "cPickle.loads", "dill.loads", "types.FunctionType", "types.CodeType", "runpy.run_path",
    "runpy.run_module", "code.InteractiveInterpreter.runsource"], ["code_exec"])
_r(["builtins.__import__", "importlib.import_module", "importlib.__import__", "imp.load_source",
    "importlib.util.spec_from_file_location", "importlib.machinery.SourceFileLoader"], ["dynamic_import"])
_r(["base64.*", "binascii.*", "codecs.decode", "codecs.encode", "zlib.decompress", "zlib.compress",
    "bz2.decompress", "lzma.decompress", "gzip.decompress", "bytes.fromhex", "builtins.bytes.fromhex",
    "Crypto.Cipher.AES.new", "cryptography.fernet.Fernet.decrypt", "cryptography.fernet.Fernet"], ["obfuscation"])
_r(["os.environ", "os.getenv", "os.environ.get", "os.environ.copy", "os.environ.items", "os.environ.keys",
    "os.environ.values", "os.environb", "os.getenvb"], ["env_access"])
_r(["platform.node", "platform.uname", "platform.system", "platform.platform", "platform.version",
    "platform.machine", "platform.processor", "platform.release", "getpass.getuser", "os.getlogin", "os.uname",
    "socket.gethostname", "socket.gethostbyname", "socket.getfqdn", "uuid.getnode", "psutil.net_if_addrs",
    "psutil.process_iter", "netifaces.interfaces"], ["system_info"])
# open()/io.open()/codecs.open() get file_read or file_write from their mode argument (ast_parser);
# `.read()` on the handle is transparent so the open() call stays the single source site.
_r(["pathlib.Path.read_text", "pathlib.Path.read_bytes", "os.listdir", "os.walk", "os.scandir", "glob.glob", "glob.iglob", "pathlib.Path.glob",
    "pathlib.Path.rglob", "sqlite3.connect", "shutil.copytree"], ["file_read"])
_r(["builtins.open.write", "builtins.open.writelines", "io.open.write", "codecs.open.write",
    "pathlib.Path.write_text", "pathlib.Path.write_bytes", "shutil.copy", "shutil.copy2", "shutil.copyfile",
    "shutil.move", "os.rename", "os.makedirs", "os.chmod", "tempfile.NamedTemporaryFile"], ["file_write"])
_r(["os.remove", "os.unlink", "os.rmdir", "shutil.rmtree", "pathlib.Path.unlink"], ["file_delete"])
_r(["ctypes.*", "cffi.*", "win32api.*", "win32con.*"], ["native_code"])
_r(["winreg.SetValue", "winreg.SetValueEx", "winreg.CreateKey", "_winreg.SetValueEx", "_winreg.CreateKey",
    "crontab.CronTab", "crontab.CronTab.new", "crontab.CronTab.write", "schtasks"], ["persistence"])
_r(["pyperclip.paste", "pyperclip.copy", "pynput.*", "keyboard.*", "PIL.ImageGrab.grab", "pyautogui.screenshot",
    "mss.mss", "mss.mss.shot", "tkinter.Tk.clipboard_get"], ["surveillance"])
_r(["browser_cookie3.*", "win32crypt.CryptUnprotectData", "keyring.get_password"], ["sensitive_data"])

# Calls whose result simply carries the data of their arguments (no behavior of their own).
TRANSPARENT = {
    "builtins.str", "builtins.bytes", "builtins.dict", "builtins.list", "builtins.tuple", "builtins.set",
    "builtins.repr", "builtins.format", "builtins.bytearray", "builtins.sorted", "builtins.reversed",
    "builtins.map", "builtins.filter", "builtins.zip", "builtins.iter", "builtins.next", "json.dumps",
    "json.loads", "json.dump", "urllib.parse.urlencode", "urllib.parse.quote", "urllib.parse.quote_plus",
    "urllib.request.Request", "urllib2.Request", "requests.Request",
}

# Attribute reads (not calls) that are behavior sources.
ATTRIBUTE_SOURCES = {"os.environ": ("env_access",), "os.environb": ("env_access",),
                     "sys.argv": (), "os.name": ("system_info",), "sys.platform": ("system_info",)}

# ---- string indicators -------------------------------------------------------------
URL_RE = re.compile(r"(?:https?|ftp|wss?)://[^\s'\"<>]+", re.I)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{2,5})?\b")
SENSITIVE_RE = re.compile(
    r"(\.ssh|id_rsa|id_ed25519|\.aws[/\\]credentials|\.git-credentials|\.npmrc|\.pypirc|\.netrc|"
    r"Login Data|Local State|\bCookies\b|\.mozilla|Google[/\\]Chrome|discord(?:canary|ptb)?[/\\]Local Storage|"
    r"leveldb|wallet\.dat|Exodus|MetaMask|\.bash_history|/etc/passwd|/etc/shadow|\.docker[/\\]config|"
    r"\.kube[/\\]config|keychain)", re.I)
PERSISTENCE_RE = re.compile(
    r"(\.bashrc|\.zshrc|\.profile|crontab|Start Menu[/\\]+Programs[/\\]+Startup|LaunchAgents|LaunchDaemons|"
    r"CurrentVersion[/\\]+Run|systemd[/\\]system|\.config[/\\]autostart|schtasks)", re.I)
EXFIL_HOST_RE = re.compile(r"(discord(?:app)?\.com/api/webhooks|api\.telegram\.org|pastebin\.com|ngrok|transfer\.sh|"
                           r"requestbin|webhook\.site|pipedream\.net|interact\.sh|oast\.)", re.I)
B64_RE = re.compile(r"^[A-Za-z0-9+/=_-]{60,}$")
HEX_RE = re.compile(r"^(?:[0-9a-fA-F]{2}){40,}$")


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def string_indicators(s: str) -> List[Tuple[str, str]]:
    """[(kind, value)] for one string literal. kinds: url, ip, exfil_host, sensitive_path,
    persistence_path, encoded_blob."""
    out: List[Tuple[str, str]] = []
    for u in URL_RE.findall(s):
        out.append(("url", u[:200]))
    if not out:
        for ip in IP_RE.findall(s):
            if not ip.startswith(("0.", "127.", "255.")) and all(int(x) <= 255 for x in ip.split(":")[0].split(".")):
                out.append(("ip", ip))
    m = EXFIL_HOST_RE.search(s)
    if m:
        out.append(("exfil_host", m.group(0)))
    m = SENSITIVE_RE.search(s)
    if not m:
        for rx in EXTRA_SENSITIVE:                       # fixture rulesets (ensure_ruleset)
            m = rx.search(s)
            if m:
                break
    if m:
        out.append(("sensitive_path", m.group(0)))
    m = PERSISTENCE_RE.search(s)
    if m:
        out.append(("persistence_path", m.group(0)))
    st = s.strip()
    if len(st) > 200_000:                                # entropy of a huge blob: sample its head
        st = st[:200_000]
    if (B64_RE.match(st) and shannon_entropy(st) >= 4.0) or HEX_RE.match(st):
        out.append(("encoded_blob", st[:40] + "..."))
    return out


INDICATOR_BEHAVIORS = {"sensitive_path": ("sensitive_data",), "persistence_path": ("persistence",),
                       "encoded_blob": ("obfuscation",), "exfil_host": ("network_send",)}


# ---- API -> behaviors ----------------------------------------------------------------
_CACHE: Dict[str, Tuple[str, ...]] = {}


def behaviors_of(api: str) -> Tuple[str, ...]:
    """Behaviors for a canonical API name ('requests.post', 'builtins.exec', 'socket.socket.connect')."""
    if api in _CACHE:
        return _CACHE[api]
    out: List[str] = []
    for pat, bs, _origin in API_RULES:
        if fnmatch.fnmatchcase(api, pat):
            for b in bs:
                if b not in out:
                    out.append(b)
    _CACHE[api] = tuple(out)
    return _CACHE[api]


def rule_origin(api: str) -> str:
    origins = sorted({o for pat, _b, o in API_RULES if fnmatch.fnmatchcase(api, pat)})
    return "+".join(origins) if origins else ""


# ---- fixture rulesets ------------------------------------------------------------------
# Synthetic controlled fixtures (data/cases.py) call a fake `mock_runtime` module instead of real I/O.
# Its rules are switched on only for samples whose meta names the ruleset, so real packages never see
# them; the switch is idempotent and also happens inside analysis worker processes.
EXTRA_SENSITIVE: List[re.Pattern] = []
FIXTURE_RULESETS = {
    "mock_runtime": {
        "rules": [("mock_runtime.read_mock_file", ("file_read",), "fixture:mock_runtime"),
                  ("mock_runtime.send_mock_http", ("network_send",), "fixture:mock_runtime")],
        "sensitive": [r"private[/\\]token"],
    },
}
_ACTIVE_RULESETS: set = set()


def ensure_ruleset(name: str) -> bool:
    """Activate a fixture ruleset once per process. Returns True when it was newly activated."""
    if not name or name in _ACTIVE_RULESETS or name not in FIXTURE_RULESETS:
        return False
    rs = FIXTURE_RULESETS[name]
    extend_rules(rs["rules"])
    EXTRA_SENSITIVE.extend(re.compile(p, re.I) for p in rs["sensitive"])
    _ACTIVE_RULESETS.add(name)
    return True


def rules_fingerprint() -> str:
    """Hash of the active API rules + extra patterns (part of the facts-cache key)."""
    import hashlib
    blob = repr(sorted((p, tuple(b), o) for p, b, o in API_RULES)) + repr(sorted(r.pattern for r in EXTRA_SENSITIVE))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def extend_rules(rules: Iterable[Tuple[str, Sequence[str], str]]) -> int:
    """Add rules (e.g. from PyGuard's taxonomy). Unknown behavior names are ignored."""
    n = 0
    for pat, bs, origin in rules:
        bs = tuple(b for b in bs if b in BEHAVIORS)
        if bs:
            API_RULES.append((pat, bs, origin)); n += 1
    _CACHE.clear()
    return n


# Keyword lexicon used to map free-text claims onto behaviors (verification).
CLAIM_LEXICON: Dict[str, Tuple[str, ...]] = {
    "data_exfiltration": ("exfiltrat", "steal", "leak", "harvest", "send.*(credential|token|environment|data|info)",
                          "upload.*(credential|token|environment|data|file)"),
    "download_exec": ("download.*(execut|run|eval)", "fetch.*(execut|run|eval)", "remote (code|payload)", "dropper",
                      "second.stage"),
    "network_send": ("send", "post request", "upload", "transmit", "webhook", "beacon", "exfil", "http post"),
    "network_download": ("download", "fetch", "retriev", "http get", "urlopen", "urlretrieve"),
    "network_connect": ("socket", "connect to", "reverse shell", "c2", "command.and.control"),
    "file_read": ("read.*file", "reads? (the )?(local )?files?", "open.*file", "enumerat", "list.*director"),
    "file_write": ("write.*file", "writes", "drop.*file", "create.*file", "save.*disk"),
    "file_delete": ("delet", "remove.*file", "wipe", "rmtree"),
    "sensitive_data": ("credential", "password", "ssh key", "private key", "token", "cookie", "wallet", "browser data",
                       "aws", "secret"),
    "env_access": ("environment variable", "os.environ", "getenv", "env var"),
    "system_info": ("hostname", "username", "system info", "host info", "platform", "mac address", "fingerprint"),
    "process_exec": ("subprocess", "shell command", "os.system", "spawn", "execute.*command", "run.*command", "popen"),
    "code_exec": ("exec\\(", "eval\\(", "\\bexec\\b", "\\beval\\b", "dynamic(ally)? execut", "arbitrary code",
                  "execute.*code", "marshal", "pickle"),
    "obfuscation": ("obfuscat", "base64", "encod", "decod", "zlib", "compress", "hex", "rot13", "encrypt"),
    "dynamic_import": ("__import__", "import_module", "dynamic import"),
    "native_code": ("ctypes", "native", "dll", "shellcode"),
    "persistence": ("persist", "autostart", "startup", "crontab", "registry", "bashrc", "scheduled task"),
    "install_hook": ("setup.py", "install.time", "during install", "on install", "cmdclass", "post.install", ".pth",
                     "import time", "at import"),
    "surveillance": ("clipboard", "keylog", "keyboard", "screenshot", "screen capture"),
}
_LEX = {b: [re.compile(p, re.I) for p in pats] for b, pats in CLAIM_LEXICON.items()}


def behaviors_from_text(text: str) -> List[str]:
    out = []
    for b, pats in _LEX.items():
        if any(p.search(text) for p in pats):
            out.append(b)
    return out
