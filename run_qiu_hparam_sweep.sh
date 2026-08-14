#!/bin/bash
#SBATCH --job-name=qiu_hparam
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --array=0-29%4
#SBATCH --output=/weka/bethge/bkr578/projects/open-retina/openretina_assets/slurm/qiu_hparam_%A_%a.log
#
# Stage A of the qiu_2026 hyperparameter plan: 30-trial Optuna TPE sweep, 15 epochs per trial.
# Config: configs/qiu_2026_core_readout_hyperparams_search.yaml (6 search dimensions).
#
# ONE TRIAL PER SLURM TASK. This used to be a single job running all 30 trials in one process via
# `hydra.sweeper.n_trials=30`. Every attempt died the same way, always inside data loading:
#
#   job 425715,   --mem=96G  -> OUT_OF_MEMORY loading trial 2 (the 3rd), at "responses 7/10"
#   job 425886,   --mem=128G -> OUT_OF_MEMORY loading trial 6 (the 4th), at "dataloaders 9/10"
#   job 426256_0, --mem=80G  -> OUT_OF_MEMORY on its ONLY trial,         at "dataloaders 8/10"
#
# Two separate costs were at work, and the third data point is what separated them:
#
#  1. A LARGE PEAK INSIDE ONE TRIAL, during `Creating qiu movie dataloaders`. Since measured directly
#     (scratch_qiu_measure_peak_rss.py, job 427774): loading all 10 sessions costs 43.4 GB, and
#     building the dataloaders took it to a 86.7 GB peak, because `movie_val` + `movie_train_subset`
#     together re-materialise every session's train movie while the source is still held. Per-trial,
#     so process isolation cannot help -- this is what killed the 80G tasks.
#  2. CROSS-TRIAL GROWTH. Hydra --multirun keeps every trial in the SAME process, and RSS does not
#     return to baseline between trials (job 425886 reached MaxRSS 133.9 GB over 3 trials). That
#     steadily ate the headroom item 1 needs, which is why the FIRST trial always succeeded and a
#     later one always died -- and why +32 GB bought exactly one more trial.
#
# One trial per process removes cost 2 and makes the requirement flat: every task needs the same peak,
# whether it is the 1st trial or the 30th. Cost 1 was then fixed in the loader itself -- the qiu
# builder releases each session's source movie as soon as its splits are taken (`release_movies` in
# configs/dataloader/qiu_2026.yaml), which measures at a 58.9 GB peak instead of 86.7 GB.
#
# Hence --mem=80G: ~21 GB of headroom over the measured 58.9 GB construction peak, which covers the
# CUDA host context and pinned batches that the CPU-only measurement excludes. Do not go below this
# without re-running the measurement script -- 80G is exactly what OOM'd BEFORE the loader fix, so the
# margin depends on that fix being in place. --time is 03:00:00 (from 1 day) for one ~35 min trial;
# smaller and shorter allocations also schedule sooner.
#
# Careful reading MaxRSS: JobAcctGatherFrequency is 30 s here, so sacct SAMPLES rather than tracks.
# Task 426256_0 was killed for exceeding 80 GB but reports MaxRSS 52 GB -- the allocation burst
# happened between polls. Treat sacct MaxRSS as a lower bound on the true peak.
#
# The array tasks share one study through the persistent SQLite storage below (the sweeper calls
# optuna.create_study(..., load_if_exists=True)), so TPE still conditions each new suggestion on
# every trial completed so far by any task. %4 in the --array spec caps 4 tasks (= 4 H100s) in
# flight at once, so the wall clock is ~30/4 x 35 min ~ 4.5 h instead of 22.5 h.
#
# TRIAL BUDGET. There is no arithmetic to do on resume any more: every task re-reads the study first
# and exits immediately (rc 0) once TARGET_TRIALS trials are COMPLETE-or-RUNNING, so a plain
# resubmission of the full 0-29 array tops the study up to 30 and no further. Over-provisioning the
# array is free -- surplus tasks no-op in seconds. Because RUNNING trials count, up to (concurrency-1)
# extra trials can slip in around the boundary; that is intentional slack, not a bug.
#
# Submit / resume:            sbatch run_qiu_hparam_sweep.sh
# Fewer GPUs at once:         sbatch --array=0-29%2 run_qiu_hparam_sweep.sh
# Strictly serial (1 writer): sbatch --array=0-29%1 run_qiu_hparam_sweep.sh
# Progress:                   sqlite3 openretina_assets/optuna/qiu_hparams.db \
#                               'select state, count(*) from trials group by state;'
#
# SQLITE CONCURRENCY. Writes are tiny and rare (create trial / set params / complete trial: a few per
# 35 min), so 4 concurrent writers on weka is low-risk, and a lock timeout would fail one task while
# the study and every other task survive. If you ever see "database is locked", drop to %1 or move to
# a real RDB (postgres/mysql) -- do not raise the concurrency.
#
# STALE RUNNING TRIALS. A task killed mid-trial (OOM, wall clock, scancel) leaves its trial stuck in
# RUNNING forever: Optuna never gets to mark it, and it then counts against TARGET_TRIALS here. Reset
# those to FAIL before resubmitting -- but *** NAME THE TRIAL NUMBERS EXPLICITLY ***. A blanket "reset
# every RUNNING trial" also wrecks the bookkeeping of tasks that are training right now, and with %4
# concurrency there usually are some. Find the orphans first (cross-check `sacct -j <array>` for which
# tasks actually died), then reset only those numbers:
#   sqlite3 openretina_assets/optuna/qiu_hparams.db \
#     "select number, datetime_start from trials where state='RUNNING';"
#   uv run python -c "
#   import optuna; from optuna.trial import TrialState
#   ORPHANS = {7}   # <-- edit: trial NUMBERS confirmed dead
#   s = optuna.storages.RDBStorage('sqlite:////weka/bethge/bkr578/projects/open-retina/openretina_assets/optuna/qiu_hparams.db')
#   st = optuna.load_study(study_name='qiu_2026_core_readout_hparams_search', storage=s)
#   [s.set_trial_state(t._trial_id, TrialState.FAIL) for t in st.get_trials(deepcopy=False)
#    if t.number in ORPHANS and t.state == TrialState.RUNNING]"
#
# SAMPLER SEED -- THE SEED MUST BE UNIQUE PER PROCESS, ACROSS SUBMISSIONS, NOT JUST PER TASK.
# The config pins sampler.seed=42, and TPESampler's first n_startup_trials (default 10) suggestions
# come from a seeded RANDOM sampler whose RNG is constructed fresh in each process and does NOT look
# at the study history. So every process that starts with seed 42 replays the same draw sequence from
# the top, however many trials the study already contains. That already happened for real: job 425886
# resumed the study with seed 42 and re-drew job 425715's points, so
#
#   trial 3 == trial 0  and  trial 4 == trial 1   (byte-identical params; values differ only by
#                                                  training nondeterminism, ~0.004)
#
# i.e. 5 COMPLETE trials but only 3 distinct configurations. Per-task offsetting alone does not fix
# this -- 42+0 is still 42, so array task 0 would draw trial 0's point a third time. The seed is
# therefore derived from the ARRAY JOB ID, which SLURM never reuses, giving every submission a fresh
# block of 100 seeds and every task within it a distinct one. Recorded in the log for reproducibility.
# The training seed (`seed: 42`) is deliberately left alone -- that one has to stay fixed for trials
# to be comparable.
#
# Duplicate trials are not silently harmful, but they do cost: TPE over-weights a repeated
# observation, and each duplicate eats one of TARGET_TRIALS. Check for them with:
#   sqlite3 openretina_assets/optuna/qiu_hparams.db "select param_value, count(distinct trial_id)
#     from trial_params where param_name='model.learning_rate' group by param_value having count(*)>1;"
#
# SMOKE TEST. QIU_SWEEP_SMOKE=1 runs 2 trials in ~5 min: 1 epoch, 10 train / 2 val / 1 test batch,
# batch_size 4, 1 session. It uses an in-memory study and a separate output tree, so smoke trials
# cannot pollute the real study DB or results, and it skips the trial-budget guard. This is the only
# check that proves the SAMPLED overrides actually build a model and that objective_target comes back
# as a float. Run it inside an existing GPU allocation (no srun nesting needed):
#   QIU_SWEEP_SMOKE=1 bash run_qiu_hparam_sweep.sh

set -euo pipefail

REPO=/weka/bethge/bkr578/projects/open-retina
CONFIG=qiu_2026_core_readout_hyperparams_search
EXP=qiu_2026_core_readout_hparams_search   # exp_name in the config; note hparams != hyperparams
DB_PATH=$REPO/openretina_assets/optuna/qiu_hparams.db
TARGET_TRIALS="${QIU_SWEEP_TARGET:-30}"

# Identity of this task. Defaults keep the script runnable by hand, outside an array.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

# Optuna sampler seed: one fresh block of 100 per submission (see SAMPLER SEED in the header -- a
# reused seed re-draws points the study already has). SLURM job ids are unique and monotonic; the
# PID fallback covers by-hand runs where ARRAY_JOB is non-numeric.
if [[ "$ARRAY_JOB" =~ ^[0-9]+$ ]]; then
  SEED_BLOCK=$(( ARRAY_JOB % 100000 ))
else
  SEED_BLOCK=$(( $$ % 100000 ))
fi
SAMPLER_SEED=$(( SEED_BLOCK * 100 + TASK_ID ))   # TASK_ID < 100, i.e. arrays up to 0-99

# Thread pools capped to the ALLOCATION, not hardcoded -- exporting more threads than the allocation
# has cores is what froze the node previously (see qiu_2026_history.md). Derived so this script is
# safe both under sbatch (--cpus-per-task=16) and by hand in a smaller interactive allocation.
NTHREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"

mkdir -p "$REPO/openretina_assets/slurm" "$REPO/openretina_assets/optuna"
cd "$REPO"

EXTRA=()
if [[ "${QIU_SWEEP_SMOKE:-0}" == "1" ]]; then
  S='["dynamic28188-16-3-Fluorescence-7b721b-v4a"]'
  EXTRA=(
    hydra.sweeper.n_trials=2
    hydra.sweeper.storage=null                 # in-memory: do not touch the real study DB
    exp_name="${EXP}_smoke"                    # separate output tree
    trainer.max_epochs=1
    +trainer.limit_train_batches=10
    +trainer.limit_val_batches=2
    +trainer.limit_test_batches=1              # the 98-dataloader test loop dominates otherwise
    dataloader.batch_size=4
    "+data_io.stimuli.sessions=$S"
    "+data_io.responses.sessions=$S"
    "+data_io.pupil.sessions=$S"
  )
  MODE="SMOKE (2 trials, 1 epoch, 1 session)"
else
  # Budget guard. COMPLETE + RUNNING, so tasks already training are not double-counted and a
  # resubmitted array tops the study up instead of blowing past TARGET_TRIALS. .timeout makes the
  # read wait out another task's write instead of erroring under `set -e`.
  if [[ -f "$DB_PATH" ]]; then
    INFLIGHT=$(sqlite3 -cmd ".timeout 30000" "$DB_PATH" \
      "select count(*) from trials where state in ('COMPLETE','RUNNING');" 2>/dev/null || echo 0)
  else
    INFLIGHT=0
  fi
  if (( INFLIGHT >= TARGET_TRIALS )); then
    echo "=== task $TASK_ID: $INFLIGHT/$TARGET_TRIALS trials already complete-or-running; nothing to do."
    exit 0
  fi

  # sqlite:/// + an absolute path (which starts with /) = 4 slashes total. Do NOT write
  # sqlite:////$REPO -- $REPO already leads with /, giving 5 slashes and a bogus host component.
  DB="sqlite:///$DB_PATH"
  EXTRA=(
    "hydra.sweeper.storage=$DB"
    hydra.sweeper.n_trials=1                   # one trial per process; see the header
    "hydra.sweeper.sampler.seed=$SAMPLER_SEED"
    # Per-task output tree. The config's default sweep.dir is timestamped to the second, and array
    # tasks start in the same second, so two of them would share one output_dir and fight over
    # checkpoints and loggers.
    "hydra.sweep.dir=$REPO/openretina_assets/runs/$EXP/array_${ARRAY_JOB}/task_${TASK_ID}"
  )
  MODE="FULL (1 trial in this task, ${INFLIGHT}/${TARGET_TRIALS} already in flight, 15 epochs, 10 sessions)"
fi

echo "=== qiu_2026 Stage A hyperparameter sweep ==="
echo "=== task        : array ${ARRAY_JOB} task ${TASK_ID}"
echo "=== sampler seed: ${SAMPLER_SEED}"
echo "=== mode        : $MODE"
echo "=== thread cap  : $NTHREADS"
echo "=== overrides   : ${EXTRA[*]}"

# --multirun is REQUIRED. Without it Hydra ignores the sweeper block entirely and silently runs a
# single training job with the default parameters.
uv run openretina train --config-name "$CONFIG" --multirun "${EXTRA[@]}"
