#!/usr/bin/env bash
# Spins up N processes (ranks 0..N-1) of the ring-allreduce binary.
# Usage: ./run_ring.sh [--gpu] <world_size> [path/to/binary]
set -euo pipefail

GPU_FLAG=""
POSITIONAL_ARGS=()

for arg in "$@"; do
  if [[ "$arg" == "--gpu" ]]; then
    GPU_FLAG="--gpu"
  else
    POSITIONAL_ARGS+=("$arg")
  fi
done

if [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]]; then
  echo "Usage: $0 [--gpu] <world_size> [binary]" >&2
  exit 1
fi

WORLD_SIZE="${POSITIONAL_ARGS[0]}"
BIN="${POSITIONAL_ARGS[1]:-./build/myapp}"
LOG_DIR="./logs"

if [[ ! -x "$BIN" ]]; then
  echo "error: binary not found or not executable: $BIN" >&2
  exit 1
fi

rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

pids=()
echo "Launching $WORLD_SIZE ranks from $BIN ${GPU_FLAG:+with GPU }..."
for (( rank=0; rank<WORLD_SIZE; rank++ )); do
  if [[ -n "$GPU_FLAG" ]]; then
    "$BIN" "$GPU_FLAG" "$rank" "$WORLD_SIZE" > "$LOG_DIR/rank_${rank}.log" 2>&1 &
  else
    "$BIN" "$rank" "$WORLD_SIZE" > "$LOG_DIR/rank_${rank}.log" 2>&1 &
  fi
  pids+=("$!")
done

status=0
for rank in "${!pids[@]}"; do
  if ! wait "${pids[$rank]}"; then
    echo "rank $rank (pid ${pids[$rank]}) exited with a non-zero status" >&2
    status=1
  fi
done

echo
echo "=== Output ==="
for (( rank=0; rank<WORLD_SIZE; rank++ )); do
  echo "--- rank $rank ---"
  cat "$LOG_DIR/rank_${rank}.log"
done

exit "$status"