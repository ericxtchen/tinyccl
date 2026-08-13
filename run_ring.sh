#!/usr/bin/env bash
# Spins up N processes (ranks 0..N-1) of the ring-allreduce binary.
# Usage: ./run_ring.sh <world_size> [path/to/binary]
set -euo pipefail

WORLD_SIZE="${1:?Usage: $0 <world_size> [binary]}"
BIN="${2:-./build/myapp}"
LOG_DIR="./logs"

if [[ ! -x "$BIN" ]]; then
  echo "error: binary not found or not executable: $BIN" >&2
  exit 1
fi

rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

pids=()
echo "Launching $WORLD_SIZE ranks from $BIN ..."
for (( rank=0; rank<WORLD_SIZE; rank++ )); do
  "$BIN" "$rank" "$WORLD_SIZE" > "$LOG_DIR/rank_${rank}.log" 2>&1 &
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