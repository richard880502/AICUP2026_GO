#!/usr/bin/env bash
# Memory-polite ten-profile extraction for one shard set on the shared w77 host.
# Usage: SET=holdout3 PER_WORKER_GB=6.5 START_GB=8 ./lowmem_extract.sh
#
# Extract one shard set (SET) with a low-memory KataGo config. Workers
#   start only when MemAvailable stays above START_GB for three checks, their
#   count scales with free memory (about 6.2 GB each at steady state; a 40-game
#   smoke test under-states this as 3.7 GB, so pass PER_WORKER_GB=6.5), and every
#   worker is stopped if MemAvailable drops below KILL_GB. Extraction resumes
#   by shard, so a stop only loses the shards in progress. A cooldown follows
#   every stop so the driver does not oscillate.
#
# Processes are tracked by PID only; nothing here matches on command strings.
set -uo pipefail

R=/tmp2/richard/AICUP2026_GO/runtime/competition2_research
L=/tmp2/richard/claude_scratch/lowmem_root
OUT=/mnt/share_data_77/richard/AICUP2026_GO/ten_profile_production
SET=${SET:?SET is required (e.g. holdout3)}
FEAT=${FEAT:-$OUT/feat10_$SET}
LOGS=$OUT/logs
PROFILES=rank_11k,rank_8k,rank_5k,rank_2k,rank_1d,rank_2d,rank_3d,rank_4d,rank_5d,rank_6d
START_GB=${START_GB:-6}
KILL_GB=${KILL_GB:-1.5}
PER_WORKER_GB=${PER_WORKER_GB:-4}
MAX_WORKERS=${MAX_WORKERS:-4}
COOLDOWN_S=${COOLDOWN_S:-600}
TRAIN_START_GB=${TRAIN_START_GB:-5}
EMERGENCY_GB=${EMERGENCY_GB:-1}
mkdir -p "$FEAT" "$LOGS"
LOG=$LOGS/lowmem_extract_$SET.log

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
avail_kb() { awk '/^MemAvailable/{print $2}' /proc/meminfo; }
gb_to_kb() { awk -v g="$1" 'BEGIN{printf "%d", g*1048576}'; }
cache_complete() {
  (cd "$R" && c1/.venv/bin/python -m experiments.ten_profile_production.check_cache \
    --features "$FEAT" --shard-dir "features/shards/$SET" >/dev/null 2>&1)
}

workers=()
stop_workers() {
  for pid in "${workers[@]}"; do
    pkill -TERM -P "$pid" 2>/dev/null
    kill -TERM "$pid" 2>/dev/null
  done
  sleep 5
  for pid in "${workers[@]}"; do
    pkill -KILL -P "$pid" 2>/dev/null
    kill -KILL "$pid" 2>/dev/null
  done
  workers=()
  # Shards are written to <name>.tmp.npz and renamed, so a stopped worker can
  # leave only a temporary file; it would still match feat_*.npz globs.
  rm -f "$FEAT"/*.tmp.npz
}
trap 'say "driver interrupted"; stop_workers; exit 130' INT TERM

say "driver start: START_GB=$START_GB KILL_GB=$KILL_GB per-worker=$PER_WORKER_GB max=$MAX_WORKERS"
last_stop=0
good=0
while ! cache_complete; do
  if ((${#workers[@]} == 0)); then
    now=$(date +%s)
    if ((now - last_stop < COOLDOWN_S)); then sleep 30; continue; fi
    a=$(avail_kb)
    if ((a >= $(gb_to_kb "$START_GB"))); then good=$((good + 1)); else good=0; fi
    if ((good < 3)); then sleep 30; continue; fi
    n=$(awk -v a="$a" -v p="$PER_WORKER_GB" -v m="$MAX_WORKERS" \
      'BEGIN{n=int((a/1048576-2)/p); if(n>m)n=m; if(n<1)n=1; print n}')
    say "starting $n worker(s), MemAvailable $((a / 1024)) MB"
    for ((w = 0; w < n; w++)); do
      (cd "$R" && CUDA_VISIBLE_DEVICES=1 exec c1/.venv/bin/python "$L/extract_rank_features.py" \
        --shard-dir "features/shards/$SET" --output "$FEAT" --samples-per-game 96 \
        --profiles "$PROFILES" --device 0 --worker "$w" --num-workers "$n" --depth 3 \
        >> "$LOGS/lowmem_extract_${SET}_w${w}.log" 2>&1) &
      workers+=($!)
    done
    good=0
    continue
  fi
  a=$(avail_kb)
  if ((a < $(gb_to_kb "$KILL_GB"))); then
    say "MemAvailable $((a / 1024)) MB < ${KILL_GB} GB: stopping ${#workers[@]} worker(s)"
    stop_workers
    last_stop=$(date +%s)
    continue
  fi
  alive=()
  for pid in "${workers[@]}"; do kill -0 "$pid" 2>/dev/null && alive+=("$pid"); done
  if ((${#alive[@]} == 0)); then
    say "all workers exited; re-checking cache"
    workers=()
    continue
  fi
  workers=("${alive[@]}")
  sleep 15
done
stop_workers
say "$SET cache complete: $(ls "$FEAT"/feat_*.npz | wc -l) shards"

say "extraction for $SET complete"
