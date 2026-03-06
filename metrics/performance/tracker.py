"""
metrics/performance/tracker.py

Context-manager-based performance tracker.
Measures wall-clock time and peak RSS memory per labelled phase.

Peak memory
-----------
- We sample process RSS at ~20 Hz during each phase (fit / sample) and report
  the maximum RSS (MB) observed in that phase. This is a true peak: short-lived
  allocations are captured as long as they are resident at a sample time.
- Each implementation should be run in its own process (see run_experiment)
  so that the reported peak is the memory footprint of that implementation
  in isolation, not contaminated by prior implementations or GC from others.
"""

import threading
import time
import os
import psutil

_SAMPLE_INTERVAL_SEC = 0.05  # 20 Hz


class PerformanceTracker:

    def __init__(self):
        self._results = {}
        self._process = psutil.Process(os.getpid())

    class _Block:
        def __init__(self, tracker, label):
            self._tracker = tracker
            self._label = label
            self._peak_rss_mb = [0.0]  # mutable so sampler thread can update
            self._stop = [False]
            self._sampler = None

        def _sample_loop(self):
            while not self._stop[0]:
                try:
                    rss_mb = self._tracker._process.memory_info().rss / 1e6
                    self._peak_rss_mb[0] = max(self._peak_rss_mb[0], rss_mb)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(_SAMPLE_INTERVAL_SEC)

        def __enter__(self):
            self._mem_before = self._tracker._process.memory_info().rss / 1e6
            self._peak_rss_mb[0] = self._mem_before
            self._stop[0] = False
            self._t0 = time.perf_counter()
            self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
            self._sampler.start()
            return self

        def __exit__(self, *_):
            self._stop[0] = True
            if self._sampler is not None:
                self._sampler.join(timeout=1.0)
            elapsed = time.perf_counter() - self._t0
            mem_after = self._tracker._process.memory_info().rss / 1e6
            # True peak during phase: max of sampled RSS and final RSS
            peak_mb = max(self._peak_rss_mb[0], mem_after)
            self._tracker._results[self._label] = {
                "time_sec": round(elapsed, 4),
                "peak_memory_mb": round(peak_mb, 2),
            }

    def track(self, label: str) -> "_Block":
        return self._Block(self, label)

    def summary(self) -> dict:
        fit = self._results.get("fit", {})
        sample = self._results.get("sample", {})
        return {
            "fit_time_sec": fit.get("time_sec"),
            "sample_time_sec": sample.get("time_sec"),
            "total_time_sec": round(
                (fit.get("time_sec") or 0.0) + (sample.get("time_sec") or 0.0), 4),
            "peak_memory_fit_mb": fit.get("peak_memory_mb"),
            "peak_memory_sample_mb": sample.get("peak_memory_mb"),
            "peak_memory_max_mb": max(
                fit.get("peak_memory_mb") or 0.0,
                sample.get("peak_memory_mb") or 0.0),
        }
