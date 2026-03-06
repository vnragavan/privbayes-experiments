"""
run_experiment.py

Main entry point. Runs one or more PrivBayes implementations
at a given epsilon and seed, computes all metrics, saves results.

Each implementation is run in its own process so that peak memory is measured
in isolation (no cross-contamination from other implementations or GC).
"""

import argparse
import json
import os
import traceback
from multiprocessing import Process, Queue

import pandas as pd
from sklearn.model_selection import train_test_split

from implementations.crn_wrapper import CRNWrapper
from implementations.dpmm_wrapper import DPMMWrapper
from implementations.synthcity_wrapper import SynthCityWrapper
from metrics.performance.tracker import PerformanceTracker
from metrics.report import compute_metrics

WRAPPERS = {
    "crn":       CRNWrapper,
    "dpmm":      DPMMWrapper,
    "synthcity": SynthCityWrapper,
}


def run_one(name, wrapper, train_df, test_df, holdout_df,
            schema, epsilon, seed, n_samples, output_dir):
    tracker = PerformanceTracker()

    with tracker.track("fit"):
        wrapper.fit(train_df, schema=schema)

    with tracker.track("sample"):
        synth_df = wrapper.sample(n_samples)

    out_path = os.path.join(
        output_dir, f"{name}_eps{epsilon}_seed{seed}.csv")
    synth_df.to_csv(out_path, index=False)

    if "age" in synth_df.columns:
        print(f"  [debug] age range at constraint check: {synth_df['age'].min()} - {synth_df['age'].max()}")

    metrics = compute_metrics(
        real_df=train_df,
        synth_df=synth_df,
        schema=schema,
        privacy_report=wrapper.privacy_report(),
        implementation=name,
        performance=tracker.summary(),
        test_real_df=test_df,
        train_df=train_df,
        holdout_df=holdout_df,
    )
    metrics["output_path"] = out_path
    metrics["status"] = "ok"
    return metrics


def _run_one_impl_worker(
    q, name, train_df, test_df, holdout_df, schema,
    epsilon, seed, n, output_dir,
):
    """Run a single implementation in this process; put (name, result) on queue."""
    try:
        cls = WRAPPERS.get(name)
        if cls is None:
            q.put((name, {"status": "error", "error": f"unknown impl: {name}"}))
            return
        wrapper = cls(epsilon=epsilon, seed=seed)
        result = run_one(
            name, wrapper, train_df, test_df, holdout_df,
            schema, epsilon, seed, n, output_dir)
        q.put((name, result))
    except Exception as e:
        q.put((name, {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }))


def run_experiment(schema_path, data_path, epsilon=1.0,
                   seed=0, n_samples=None, output_dir="outputs",
                   implementations=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(schema_path) as f:
        schema = json.load(f)

    df = pd.read_csv(data_path)

    # 60/20/20 split: train / test / holdout(MIA)
    train_df, temp_df = train_test_split(
        df, test_size=0.4, random_state=seed)
    test_df, holdout_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed)

    n = n_samples or len(train_df)
    impls = implementations or list(WRAPPERS.keys())

    print(f"\nDataset : {data_path}")
    print(f"Schema  : {schema_path}")
    print(f"Rows    : {len(df)} total | "
          f"train={len(train_df)} test={len(test_df)} holdout={len(holdout_df)}")
    print(f"Epsilon : {epsilon}  Seed: {seed}  n_samples: {n}")
    print(f"Running : {impls} (one process per implementation for clean peak memory)\n")

    all_results = {}

    for name in impls:
        print(f"{'=' * 55}")
        print(f"  {name.upper()}")
        print(f"{'=' * 55}")
        cls = WRAPPERS.get(name)
        if cls is None:
            print(f"  UNKNOWN implementation: {name}")
            all_results[name] = {"status": "error",
                                  "error": f"unknown impl: {name}"}
            continue

        q = Queue()
        p = Process(
            target=_run_one_impl_worker,
            args=(q, name, train_df, test_df, holdout_df, schema,
                  epsilon, seed, n, output_dir),
        )
        p.start()
        try:
            name_back, result = q.get(timeout=3600)  # 1 h timeout per impl
        except Exception as e:
            result = {"status": "error", "error": str(e)}
            name_back = name
        p.join(timeout=5)

        all_results[name_back] = result
        if result.get("status") == "ok":
            perf = result.get("performance", {})
            surv = result.get("survival", {})
            print(f"  fit    : {perf.get('fit_time_sec', '?'):.2f}s  "
                  f"sample: {perf.get('sample_time_sec', '?'):.2f}s  "
                  f"peak_mem: {perf.get('peak_memory_max_mb', '?'):.1f} MB")
            print(f"  km_l1  : {surv.get('km_l1', '?')}")
            print(f"  output : {result.get('output_path', '?')}")
        else:
            print(f"  ERROR  : {result.get('error', '?')}")
            if result.get("traceback"):
                print(result["traceback"])

    results_path = os.path.join(
        output_dir, f"results_eps{epsilon}_seed{seed}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved: {results_path}")
    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run PrivBayes compliance experiment")
    p.add_argument("--schema",           required=True,
                   help="Path to schema-generator JSON")
    p.add_argument("--data",             required=True,
                   help="Path to dataset CSV")
    p.add_argument("--epsilon",          type=float, default=1.0)
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--n-samples",        type=int,   default=None,
                   help="Number of synthetic rows (default: same as train set)")
    p.add_argument("--output-dir",       default="outputs")
    p.add_argument("--implementations",  nargs="+",
                   default=["crn", "dpmm", "synthcity"],
                   help="Which implementations to run")
    args = p.parse_args()

    run_experiment(
        schema_path=args.schema,
        data_path=args.data,
        epsilon=args.epsilon,
        seed=args.seed,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        implementations=args.implementations,
    )
