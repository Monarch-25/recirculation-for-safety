#!/usr/bin/env bash
# JarvisLabs orchestrator for the recirculation eval harness (A100-40GB).
#
# Creates/ reuses one persistent instance, syncs the repo, and runs
# scripts/evaluate.py on it via a managed `jl run` (background-safe: the
# job survives this process dying, unlike the Modal CLI path).
#
# Usage (from repo root):
#   scripts/jl_run.sh <config.yaml> \
#       [--limit N] [--batch-size N] [--resume-from REL_PATH] \
#       [--monitor [S]] [--pause] [--no-download]
#
#   --monitor          sample nvidia-smi every 20s during the run (local CSV)
#   --monitor S        sample every S seconds
#   --pause            pause the instance after the run (stops GPU billing)
#   --no-download      skip pulling /home/recirc_results.tar.gz
#
# Examples:
#   scripts/jl_run.sh configs/gsm8k_gemma3_4b_pt_recirc_a010_s18_d7_jl.yaml \
#       --limit 5 --monitor 5 --pause
#   scripts/jl_run.sh configs/gsm8k_gemma3_4b_pt_recirc_a010_s18_d7_jl.yaml \
#       --batch-size 64 --monitor 20
set -euo pipefail

CONFIG="${1:?usage: jl_run.sh <config.yaml> [opts]}"
shift

LIMIT=""
BATCH=""
RESUME=""
MONITOR=""
MONITOR_IV=20
PAUSE=0
DOWNLOAD=1

while [ $# -gt 0 ]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --batch-size) BATCH="$2"; shift 2 ;;
    --resume-from) RESUME="$2"; shift 2 ;;
    --monitor)
      MONITOR=1
      if [[ "${2:-}" =~ ^[0-9]+$ ]]; then MONITOR_IV="$2"; shift; fi
      shift ;;
    --pause) PAUSE=1; shift ;;
    --no-download) DOWNLOAD=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 2; }

INSTANCE_NAME="jl-recirc-a100"
STATE_FILE=".jl_state"
OUT_DIR="results_jl"
mkdir -p "$OUT_DIR"

echo "== frame: config=$CONFIG limit=${LIMIT:-full} batch=${BATCH:-cfg} resume=${RESUME:-fresh}"

# ---- 1. instance (create once, reuse across stages) -----------------------
machine_id=""
if [ -f "$STATE_FILE" ]; then
  machine_id="$(cat "$STATE_FILE")"
fi
if [ -n "$machine_id" ] \
     && jl get "$machine_id" >/dev/null 2>&1; then
  echo "== reuse instance $machine_id"
  STATE="$(jl get "$machine_id" --json 2>/dev/null | python3 -c \
    "import sys,json;d=json.load(sys.stdin);print(d.get('status') or d.get('state') or 'unknown')" 2>/dev/null || echo unknown)"
  echo "   instance state: $STATE"
  STATE_LOW="$(echo "$STATE" | tr 'A-Z' 'a-z')"
  if [[ "$STATE_LOW" == *paused* || "$STATE_LOW" == *stopped* ]]; then
    echo "== resuming instance $machine_id (warm model cache)"
    RESUME_JSON="$(jl resume "$machine_id" --gpu A100 --yes --json)"
    echo "== resumed"
    # Resume may assign a NEW machine id — always use the returned one.
    NEW_ID="$(echo "$RESUME_JSON" | python3 -c \
      "import sys,json; print(json.load(sys.stdin).get('machine_id',''))" 2>/dev/null || true)"
    if [ -n "$NEW_ID" ] && [ "$NEW_ID" != "$machine_id" ]; then
      echo "== resume reassigned machine id: $machine_id -> $NEW_ID"
      machine_id="$NEW_ID"
      echo "$machine_id" > "$STATE_FILE"
    fi
  fi
else
  echo "== creating A100-40GB instance $INSTANCE_NAME (this may take a couple minutes)"
  CREATE_JSON="$(jl create --gpu A100 --storage 100 --name "$INSTANCE_NAME" \
                 --template pytorch --yes --json)"
  machine_id="$(echo "$CREATE_JSON" | python3 -c "import sys,json;print(json.load(sys.stdin)['machine_id'])")"
  echo "$machine_id" > "$STATE_FILE"
  echo "== instance $machine_id ready"
fi

# ---- 2. secrets env (HF gated model + wandb) --------------------------------
# Written as JSON; scripts/jl_eval_entry.py loads it into os.environ.
SCRIPTS_DIR="scripts"
SECRETS="$SCRIPTS_DIR/.jl_secrets.env"
TOKENS="access_tokens.txt"
python3 - "$SECRETS" "$TOKENS" <<'PY'
import json, os, sys
path, tokens_file = sys.argv[1], sys.argv[2]
vals = {}
for line in open(tokens_file):
    line = line.rstrip("\n")
    for key in ("hf_access_token", "wandb_login"):
        if line.startswith(key):
            _, _, v = line.partition("=")
            vals[key] = v.strip()
            break
payload = {
    "HF_TOKEN": vals.get("hf_access_token", ""),
    "WANDB_API_KEY": vals.get("wandb_login", ""),
    "WANDB_ENTITY": "prefrontier-poor-research",
    "WANDB_PROJECT": "recirc-gsm8k",
}
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "w") as fh:
    json.dump(payload, fh, indent=2)
os.chmod(path, 0o600)
print(f"== wrote {path}")
PY

# ---- 3. launch managed run ---------------------------------------------------
EVAL_ARGS=(--config "$CONFIG")
[ -n "$LIMIT" ] && EVAL_ARGS+=(--limit "$LIMIT")
[ -n "$BATCH" ] && EVAL_ARGS+=(--batch-size "$BATCH")
[ -n "$RESUME" ] && EVAL_ARGS+=(--resume-from "$RESUME")

echo "== launching: jl run . --script $SCRIPTS_DIR/jl_eval_entry.py --on $machine_id"
echo "   eval args: ${EVAL_ARGS[*]}"
LAUNCH_JSON="$(jl run . --script "$SCRIPTS_DIR/jl_eval_entry.py" \
  --on "$machine_id" --requirements "$SCRIPTS_DIR/jl_requirements.txt" \
  --no-follow --yes --json -- "${EVAL_ARGS[@]}")"

print_json_field() { echo "$LAUNCH_JSON" | python3 -c \
  "import sys,json; print(json.load(sys.stdin).get('$1',''))"; }
RUN_ID="$(print_json_field run_id)"
[ -n "$RUN_ID" ] || { echo "failed to parse run_id from:" >&2; echo "$LAUNCH_JSON" >&2; exit 3; }
echo "== run $RUN_ID started (wandb run name = <experiment.name>-<run_id>, has _jl_lab suffix)"

# ---- 4. optional GPU monitor --------------------------------------------------
MON_PID=""
if [ -n "$MONITOR" ]; then
  GPU_CSV="$OUT_DIR/gpu_${RUN_ID}.csv"
  bash "$SCRIPTS_DIR/jl_gpu_monitor.sh" "$machine_id" "$GPU_CSV" \
      "$MONITOR_IV" 7200 &
  MON_PID=$!
  echo "== gpu monitor pid $MON_PID -> $GPU_CSV"
fi

# ---- 5. poll to completion -----------------------------------------------------
STATUS=""
while :; do
  sleep 20
  STAT_JSON="$(jl run status "$RUN_ID" --json 2>/dev/null || echo '{}')"
  STATUS="$(echo "$STAT_JSON" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('state','unknown'))" 2>/dev/null || echo unknown)"
  echo "   [$STATUS]"
  case "$STATUS" in
    succeeded|failed|instance-paused|instance-missing|instance-failed) break ;;
    *) : ;;
  esac
done

if [ -n "$MON_PID" ]; then
  echo "== stopping gpu monitor"
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  # Final summary already printed to stderr by the monitor; show the CSV head.
  [ -f "$GPU_CSV" ] && { echo "== gpu samples:"; tail -n 3 "$GPU_CSV"; }
fi

echo "== final status: $STATUS"
echo "--- run log tail ---"
jl run logs "$RUN_ID" --tail 30 || true

# ---- 6. download results ---------------------------------------------------------
if [ "$DOWNLOAD" -eq 1 ] && [ "$STATUS" = "succeeded" ]; then
  # Entry prints `[jl] results_package=<abs>`; fall back to /home path.
  LOGS="$(jl run logs "$RUN_ID" 2>/dev/null || true)"
  PACKAGE="$(echo "$LOGS" | grep -oE 'results_package=[^ ]+' | tail -n 1 | cut -d= -f2)"
  PACKAGE="${PACKAGE:-/home/recirc_results.tar.gz}"
  echo "== downloading $PACKAGE"
  jl download "$machine_id" "$PACKAGE" "$OUT_DIR/${RUN_ID}.tar.gz" \
    || echo "download failed (instance may not be reachable)"
fi

# ---- 7. lifecycle ------------------------------------------------------------------
if [ "$PAUSE" -eq 1 ]; then
  echo "== pausing instance $machine_id (storage-only cost from now)"
  jl pause "$machine_id" --yes --json >/dev/null && echo "== paused"
  echo "resume later with: jl resume $machine_id --gpu A100"
else
  echo "== instance $machine_id left RUNNING (GPU billing active). "
  echo "   pause when done: jl pause $machine_id --yes"
fi

echo "== done. run_id=$RUN_ID state=$STATUS"
exit 0