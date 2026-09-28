#!/usr/bin/env bash
#
# local_stop.sh — stop and remove the standalone A3S container started by
# local_setup.sh.
#
# Usage:
#   ./docker/local_stop.sh            # stop & remove the container (default)
#   ./docker/local_stop.sh --pause    # just stop it, keep it (resume: docker start <container>)
#   ./docker/local_stop.sh --help
#
# Overridable via environment:
#   A3S_CONTAINER (caba3s-local)

set -euo pipefail

A3S_CONTAINER="${A3S_CONTAINER:-caba3s-local}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m    ✓ %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m    ✗ %s\033[0m\n' "$*" >&2; exit 1; }

case "${1:-}" in
  "")             MSG="Stopping and removing $A3S_CONTAINER"; CMD="rm -f" ;;
  --pause|--stop) MSG="Pausing $A3S_CONTAINER (resume with: docker start $A3S_CONTAINER)"; CMD="stop" ;;
  -h|--help)      sed -n '3,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
  *)              die "unknown argument: ${1} (see --help)" ;;
esac

log "$MSG"
docker "$CMD" "$A3S_CONTAINER" >/dev/null 2>&1 || true
ok "done"
