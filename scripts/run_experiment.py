"""Plan entry point: `python scripts/run_experiment.py --method e0 --split test ...`
-> python -m pypi_rag.cli run --methods e0 ... (several --method values allowed)."""
from _wrap import run

if __name__ == "__main__":
    run("run", rename={"--method": "--methods"})
