"""Extrapolate GPU time and cost of a full run from a measured PILOT run (50-100 items).

Measure first, then estimate: run the pipeline on a small subset (e.g. `--max-items 50` for Task A,
`--max-items 15` per group for Task B) with the real reader, then

    python scripts/estimate_cost.py --pilot <pilot results dir> --items 500 --methods 7 --readers 2 \
        --runs 3 --price 1.79

The pilot gives measured GPU seconds per generated prompt (sum of each prompt's share of its generation batch,
over unique prompts; E4 variants reuse E3's prompts). Prompts of a full run = items x generated methods x readers
x runs (validation, test, ...). Prompt length differs between datasets: use a pilot of the same dataset.
"""
import argparse
import json
import os
import sys


def seconds_per_prompt(results_dir: str) -> float:
    R = json.load(open(os.path.join(results_dir, "results.json"), encoding="utf-8"))
    c = R.get("cost")
    if c and c.get("unique_prompts"):
        return c["gpu_seconds_total"] / c["unique_prompts"]
    # older runs: rebuild from the LLM cache (one line per unique generation)
    p = os.path.join(results_dir, "llm_cache.jsonl")
    lat = [json.loads(x).get("latency_ms", 0.0) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []
    if not lat:
        sys.exit("no cost block and no llm_cache.jsonl in the pilot directory")
    return sum(lat) / 1000.0 / len(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True, help="results dir of the pilot run")
    ap.add_argument("--items", type=int, required=True, help="items (packages or questions) per run")
    ap.add_argument("--methods", type=int, required=True, help="methods that generate (E4 variants do not)")
    ap.add_argument("--readers", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1, help="number of runs of this size (val, test, ...)")
    ap.add_argument("--price", type=float, default=None, help="USD per GPU hour")
    ap.add_argument("--gpus", type=int, default=1, help="GPUs working in parallel (one reader each)")
    a = ap.parse_args()
    spp = seconds_per_prompt(a.pilot)
    prompts = a.items * a.methods * a.readers * a.runs
    hours = spp * prompts / 3600
    wall = hours / max(a.gpus, 1)
    print(f"measured: {spp:.3f} GPU s per prompt (pilot {a.pilot})")
    print(f"full run: {prompts} prompts -> {hours:.2f} GPU hours, about {wall:.2f} h wall clock on {a.gpus} GPU(s)")
    if a.price:
        print(f"cost: about {hours * a.price:.2f} USD at {a.price} USD per GPU hour (generation only; add setup and data steps)")


if __name__ == "__main__":
    main()
