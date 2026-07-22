# qiu_2026 — Real Dataset Location (machine-specific)

> Split out of `qiu_2026_integration_plan.md` because this is specific to individual machines, not
> part of the portable integration plan. **Run the identification checklist below at the start of every
> session, before touching `paths.data_dir` / `OPENRETINA_CACHE_DIRECTORY` or assuming the data is
> present** — which of the three cases you're in determines where the data pointer should go.

## Step 0 — identify which machine you're on

Run these checks in order; stop at the first match.

### 1. Bethgelab cluster (any compute node on the shared weka mount)

```bash
ls -d /weka/bethge/bkr578/openretina_cache/franke_lab/qiu_2026/dynamic*/ | wc -l   # == 10 → this machine
```

If this prints `10`, you're on the cluster. **The specific compute node does not matter** — data lives on
the shared weka mount (home is a weka mount), so it's present and complete on *every* node in the
cluster, not just the one that downloaded it.

- **Data pointer:** no action needed. This is the default `~/openretina_cache` for user `bkr578`, so
  leave `paths.data_dir` at the HF URL (default) or `OPENRETINA_CACHE_DIRECTORY` unset —
  `get_local_file_path` finds the extracted folders and skips downloading. Do **not** re-download or
  re-verify session-by-session once the check above passes.
- **Verified state (2026-07-22):** 132 GB, all 10 session folders + all 10
  `data-quality/*neurons_fluor_good.npy` masks present (per-session sizes range 4.1 GB `28188-16-5` → 24
  GB `29163-5-8`). Source confirmed at
  `https://huggingface.co/datasets/open-retina/open-retina/tree/main/franke_lab/qiu_2026` (via
  `HfApi.list_repo_files`).
- **Hardware fingerprint:** set up on `mlcbm004` — Intel Xeon Platinum 8468, ~2 TiB RAM, 1× idle NVIDIA
  H100 80 GB — but any node of this class works; the RAM OOM seen on the old 34 GB laptop is a non-issue
  anywhere on this cluster.
- **Caveat — CPU affinity (per-node, re-check each session):** a shell may be pinned to only a few CPUs
  even on a 192-CPU box (`nproc` returned **4**; affinity `2,3,98,99` on `mlcbm004`). Check `nproc` /
  `os.sched_getaffinity(0)` before setting dataloader `num_workers`.

### 2. Local Mac laptop (`lhoefling`, 34 GB RAM) — deprecated, abandoned for real runs

```bash
[ "$(hostname)" ] && sw_vers >/dev/null 2>&1 && echo "macOS — check /Users/lhoefling/data next"
ls /Users/lhoefling/data/franke_lab/qiu_2026/dynamic*/ 2>/dev/null | wc -l   # partial (2 sessions only)
```

- **Data pointer:** `OPENRETINA_CACHE_DIRECTORY=/Users/lhoefling/data`, `paths.data_dir` →
  `/Users/lhoefling/data/franke_lab/qiu_2026`.
- **Only ever had 2 of 10 sessions** (`dynamic28188-16-5`, `dynamic28188-16-3`, ~2.8 GB combined) —
  never the full dataset. **Abandoned as a training machine**: this box has only 34.4 GB RAM, which
  OOMs/thrashes even at 2 sessions and `batch_size=4` (see the hardware-blocker history in the main
  plan). Only usable for tiny smoke tests, not for producing a real checkpoint.

### 3. Any other machine — treat as naive (unverified, nothing downloaded)

If neither check above matches, assume **no data is present** and this machine has not been
characterized:
- **Data pointer:** point `paths.data_dir` (or `OPENRETINA_CACHE_DIRECTORY`) at a scratch/cache
  directory with room for the full dataset, and let it download fresh from
  `https://huggingface.co/datasets/open-retina/open-retina/tree/main/franke_lab/qiu_2026` (~132 GB) —
  don't guess a path from either case above, they're specific to those machines.
- **Budget for the full download** before assuming any failure is a code bug — verify with the same
  session-count check pattern (`ls -d <cache>/franke_lab/qiu_2026/dynamic*/ | wc -l` should be `10`, plus
  10 `data-quality/*neurons_fluor_good.npy` masks) once it completes.
- **RAM/GPU are unknown** — check `free -h` / `nvidia-smi` (or platform equivalent) before picking
  `batch_size` or session count; don't assume cluster-class hardware. Re-run the hardware-blocker probes
  from the main plan (all 10 sessions → 5 largest → 2 smallest → reduced `batch_size`) if you hit OOM,
  rather than assuming it's the same bug already fixed in commit `923200c`.
