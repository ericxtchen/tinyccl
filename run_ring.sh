#!/usr/bin/env bash
# Spins up N processes (ranks 0..N-1) of the ring-allreduce binary.
# Usage: ./run_ring.sh [--gpu] [--size <N>] <world_size> [path/to/binary]
set -euo pipefail

EXTRA_FLAGS=()
POSITIONAL_ARGS=()

# Parse arguments dynamically using a while loop
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      EXTRA_FLAGS+=("--gpu")
      shift
      ;;
    --size)
      if [[ -z "${2:-}" ]]; then
        echo "error: --size requires a value" >&2
        exit 1
      fi
      EXTRA_FLAGS+=("--size" "$2")
      shift 2
      ;;
    --size=*)
      EXTRA_FLAGS+=("--size" "${1#*=}")
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Ensure we have at least the world_size argument
if [[ ${#POSITIONAL_ARGS[@]} -lt 1 ]]; then
  echo "Usage: $0 [--gpu] [--size <N>] <world_size> [binary]" >&2
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
echo "Launching $WORLD_SIZE ranks from $BIN (flags: ${EXTRA_FLAGS[*]:-none})..."

# Launch each rank, passing along all parsed extra flags
for (( rank=0; rank<WORLD_SIZE; rank++ )); do
  "$BIN" "${EXTRA_FLAGS[@]}" "$rank" "$WORLD_SIZE" > "$LOG_DIR/rank_${rank}.log" 2>&1 &
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