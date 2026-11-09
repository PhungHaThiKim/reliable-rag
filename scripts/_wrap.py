import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pypi_rag.cli import main  # noqa: E402


def run(sub, argv=None, rename=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for old, new in (rename or {}).items():
        argv = [new if a == old else a for a in argv]
    main([sub] + argv)
