"""
run_sweep.py

Run all implementations across a grid of epsilon values and seeds.
Results are saved under results/eps_sweep/<dataset_name>/ so multiple
datasets can coexist without overwriting each other.
"""

import argparse
import json
import os
from pathlib import Path
from run_experiment import run_experiment

DEFAULT_EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_SEEDS    = [0, 1, 2, 3, 4]


def _dataset_name_from_schema(schema_path: str) -> str:
    """Read dataset name from schema JSON, fall back to filename stem."""
    try:
        s = json.loads(Path(schema_path).read_text())
        name = s.get("dataset", "")
        if isinstance(name, str) and name.strip():
            # Sanitise for use as directory name
            return name.strip().replace(" ", "_").replace("/", "_")
    except Exception:
        pass
    return Path(schema_path).stem


def run_sweep(schema_path, data_path, epsilons=None, seeds=None,
              implementations=None, output_dir="results/eps_sweep"):
    epsilons = epsilons or DEFAULT_EPSILONS
    seeds    = seeds    or DEFAULT_SEEDS
    impls    = implementations or ["crn", "dpmm", "synthcity"]

    # ── Namespace by dataset ──────────────────────────────────────
    dataset_name = _dataset_name_from_schema(schema_path)
    output_dir   = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset    : {dataset_name}")
    print(f"Output dir : {output_dir}")
    print(f"Epsilons   : {epsilons}")
    print(f"Seeds      : {seeds}  ({len(seeds)} seeds)")
    print(f"Impls      : {impls}")
    print(f"Total runs : {len(epsilons) * len(seeds)} "
          f"({len(epsilons)} eps × {len(seeds)} seeds)\n")

    total = len(epsilons) * len(seeds)
    done  = 0

    for eps in epsilons:
        for seed in seeds:
            done += 1
            print(f"\n[{done}/{total}]  eps={eps}  seed={seed}")
            run_experiment(
                schema_path=schema_path,
                data_path=data_path,
                epsilon=eps,
                seed=seed,
                output_dir=output_dir,
                implementations=impls,
            )

    print(f"\nSweep complete. {total} runs saved to {output_dir}/")
    return output_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--schema",          required=True)
    p.add_argument("--data",            required=True)
    p.add_argument("--epsilons",        nargs="+", type=float,
                   default=DEFAULT_EPSILONS)
    p.add_argument("--seeds",           nargs="+", type=int,
                   default=DEFAULT_SEEDS)
    p.add_argument("--implementations", nargs="+",
                   default=["crn", "dpmm", "synthcity"])
    p.add_argument("--output-dir",      default="results/eps_sweep",
                   help="Base results dir — dataset name is appended automatically")
    args = p.parse_args()

    run_sweep(
        schema_path=args.schema,
        data_path=args.data,
        epsilons=args.epsilons,
        seeds=args.seeds,
        implementations=args.implementations,
        output_dir=args.output_dir,
    )
