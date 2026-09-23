#!/usr/bin/env bash
# Sample GPU utilization/VRAM on a JarvisLabs instance during a run.
# Usage (run from repo root):
#   scripts/jl_gpu_monitor.sh <machine_id> <csv_path> [interval_sec] [max_seconds]
# Writes rows: unix_ts,util_pct,mem_used_mib,mem_total_mib. Exits after
# max_seconds (default 7200) or when the remote query stops succeeding.
set -euo pipefail
MACHINE_ID="${1:?machine id required}"
CSV="${2:?csv path required}"
INTERVAL="${3:-20}"
MAX_SECONDS="${4:-7200}"

mkdir -p "$(dirname "$CSV")"
: > "$CSV"
echo "ts,util_pct,mem_used_mib,mem_total_mib" > "$CSV"

QUERY="nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"

START=$(date +%s)
while :; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))
  if [ "$ELAPSED" -ge "$MAX_SECONDS" ]; then break; fi

  LINE=$(jl exec "$MACHINE_ID" --json -- sh -lc "$QUERY" 2>/dev/null \
    | python3 -c "
import sys, json
try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)
out = (payload.get('stdout') or '').strip()
if payload.get('exit_code') != 0 or not out:
    sys.exit(0)
fields = [f.strip() for f in out.split(',')]
if len(fields) >= 3:
    print(f\"{fields[0]},{fields[1]},{fields[2]}\")
" 2>/dev/null || true)

  if [ -n "$LINE" ]; then
    echo "$NOW,$LINE" >> "$CSV"
  else
    echo "ts,$((ELAPSED/60))m: query failed (instance busy?)" >&2
  fi
  sleep "$INTERVAL"
done

# Summary line to stderr for the orchestrator.
python3 - "$CSV" <<'PY' >&2
import sys, statistics, csv
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
if not rows:
    print("no GPU samples collected")
    sys.exit(0)
util = [int(r["util_pct"]) for r in rows if r["util_pct"].strip()]
mem = [int(r["mem_used_mib"]) for r in rows if r["mem_used_mib"].strip()]
n = len(rows)
print(f"gpu_monitor: {n} samples over {n * 20}s-ish")
if util:
    print(f"  util%  mean={statistics.mean(util):.0f} median={statistics.median(util):.0f} "
          f"p90={sorted(util)[int(0.9*len(util))]:.0f} max={max(util)}")
if mem:
    print(f"  vram   peak={max(mem)} MiB mean={statistics.mean(mem):.0f} MiB")
    print(f"  vram   peak_pct_of_total={100*max(mem)/int(rows[0]['mem_total_mib']):.1f}%")
PY
echo "gpu_monitor: done -> $CSV" >&2