"""Measure peak RSS of qiu_2026 dataloader construction, with and without `release_movies`.

CPU-only: no model, no GPU, no training. One arm per process invocation -- running both arms in one
process would let the first arm's resident memory pollute the second arm's high-water mark.

    python scratch_qiu_measure_peak_rss.py --release true
    python scratch_qiu_measure_peak_rss.py --release false

Reports the process high-water mark from getrusage (exact, unlike sacct's 30 s sampling) plus an RSS
timeline sampled in a background thread, so the growth during the per-session loop is visible.
"""

import argparse
import os
import resource
import threading
import time

import hydra

from openretina.data_io.base import compute_data_info

REPO = "/weka/bethge/bkr578/projects/open-retina"
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def rss_gb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE_SIZE / 1e9


def peak_rss_gb() -> float:
    # ru_maxrss is KiB on Linux, so scale by 1024 to get bytes before converting to decimal GB --
    # otherwise the peak reads ~2.4% BELOW the concurrent statm sample, which looks like a bug.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9


class Sampler(threading.Thread):
    """Sample RSS so we see the shape of the curve, not just its maximum."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        # NOT self._stop: threading.Thread has its own private _stop() method, and shadowing it with an
        # Event makes join() blow up with "'Event' object is not callable".
        self.interval, self.samples, self._stop_event = interval, [], threading.Event()

    def run(self) -> None:
        t0 = time.perf_counter()
        while not self._stop_event.is_set():
            self.samples.append((time.perf_counter() - t0, rss_gb()))
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2)

    def max_gb(self) -> float:
        return max((r for _, r in self.samples), default=0.0)

    def timeline(self, every: float = 20.0) -> list[tuple[float, float]]:
        out, next_t = [], 0.0
        for t, r in self.samples:
            if t >= next_t:
                out.append((t, r))
                next_t += every
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=["true", "false"], required=True)
    args = parser.parse_args()
    release = args.release == "true"

    sampler = Sampler()
    sampler.start()

    def mark(label: str) -> None:
        print(f"  {label:<34s} rss {rss_gb():6.2f} GB   peak {peak_rss_gb():6.2f} GB", flush=True)

    print(f"=== release_movies={release} ===", flush=True)
    mark("baseline (interpreter + imports)")

    with hydra.initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = hydra.compose(config_name="qiu_2026_core_readout")

    t0 = time.perf_counter()
    movies_dict = hydra.utils.call(cfg.data_io.stimuli)
    neuron_data_dict = hydra.utils.call(cfg.data_io.responses)
    pupil_dict = hydra.utils.call(cfg.data_io.pupil)
    load_s = time.perf_counter() - t0
    mark(f"after loading {len(movies_dict)} sessions")

    # Hoisted exactly as in cli/train.py: must precede construction, since release empties the dict.
    data_info = compute_data_info(neuron_data_dict, movies_dict, partial_data_info=cfg.data_io.get("data_info"))
    assert data_info["input_shape"], "data_info must be computable before the movies are released"

    t1 = time.perf_counter()
    build_dataloaders = hydra.utils.instantiate(cfg.dataloader, _partial_=True)
    dataloaders = build_dataloaders(
        neuron_data_dictionary=neuron_data_dict,
        movies_dictionary=movies_dict,
        pupil_dictionary=pupil_dict,
        release_movies=release,
    )
    build_s = time.perf_counter() - t1
    mark("after building dataloaders")

    sampler.stop()
    n_test_splits = len(dataloaders) - 2
    # getrusage lags a fast-growing process (it is updated at kernel accounting points, so it can read
    # a little BELOW a concurrent /proc sample), while the sampler can miss a spike between ticks.
    # Neither is authoritative alone, so report both and take the larger as the peak.
    peak = max(peak_rss_gb(), sampler.max_gb())
    print(
        f"\n  sessions            : {len(neuron_data_dict)}"
        f"\n  movies dict left    : {len(movies_dict)} entries"
        f"\n  splits built        : train + validation + {n_test_splits} test conditions"
        f"\n  load time           : {load_s:6.1f} s"
        f"\n  build time          : {build_s:6.1f} s"
        f"\n  PEAK RSS            : {peak:6.2f} GB  (getrusage {peak_rss_gb():.2f}, sampled {sampler.max_gb():.2f})"
        f"\n  steady RSS after    : {rss_gb():6.2f} GB",
        flush=True,
    )
    print("\n  rss timeline (20 s):", " ".join(f"{t:.0f}s={r:.0f}G" for t, r in sampler.timeline()), flush=True)


if __name__ == "__main__":
    main()
