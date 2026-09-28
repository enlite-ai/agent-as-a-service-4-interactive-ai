#!/usr/bin/env bash
#
# local_setup.sh — build and run the A3S container. Standalone: it knows nothing
# about InteractiveAI, and InteractiveAI does not start it.
#
# Usage (from a3s-service/):
#   ./docker/local_setup.sh              # build (cache-aware) and start on port 5010
#   ./docker/local_setup.sh --port 6000  # publish on another host port
#   ./docker/local_setup.sh --agent xd   # serve PowerGrid with the xd agent (default: t2.1)
#   ./docker/local_setup.sh --rebuild    # build from scratch, ignoring the cache
#   ./docker/local_setup.sh --help
#
# It prints the recommendation URL to pass to InteractiveAI when it is done.
#
# Environment (flags win):
#   A3S_PORT (5010)  A3S_IMAGE (caba3s-local)  A3S_CONTAINER (caba3s-local)
#   A3S_POWERGRID_AGENT (t2.1)  — which policy serves PowerGrid: "t2.1" or "xd"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A3S_PORT="${A3S_PORT:-5010}"
A3S_IMAGE="${A3S_IMAGE:-caba3s-local}"
A3S_CONTAINER="${A3S_CONTAINER:-caba3s-local}"
A3S_POWERGRID_AGENT="${A3S_POWERGRID_AGENT:-t2.1}"

# Same checks the user can run by hand with ./docker/check_agents.sh.
# shellcheck source=check_agents.sh
source "$REPO_ROOT/docker/check_agents.sh"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    ✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m    ! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m    ✗ %s\033[0m\n' "$*" >&2; exit 1; }

BUILD_ARGS=()
while (( $# )); do
  case "$1" in
    -h|--help)            sed -n '3,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --build)              ;;  # accepted for compatibility, it is the default
    --rebuild|--no-cache) BUILD_ARGS+=(--no-cache) ;;
    --port)               [[ "${2:-}" =~ ^[0-9]+$ ]] || die "--port needs a port number"
                          A3S_PORT="$2"; shift ;;
    --port=*)             A3S_PORT="${1#*=}"
                          [[ "$A3S_PORT" =~ ^[0-9]+$ ]] || die "--port needs a port number" ;;
    --agent)              [[ -n "${2:-}" ]] || die "--agent needs an agent name (t2.1 or xd)"
                          A3S_POWERGRID_AGENT="$2"; shift ;;
    --agent=*)            A3S_POWERGRID_AGENT="${1#*=}" ;;
    *)                    die "unknown option '$1' — see --help" ;;
  esac
  shift
done

require() { command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed."; }
require docker
require curl

# Block until an HTTP endpoint answers with the wanted status, or time out.
wait_for_http() {
  local url="$1" want="${2:-200}" tries="${3:-60}" i=1 code
  while (( i <= tries )); do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$url" || true)"
    [[ "$code" == "$want" ]] && return 0
    printf '    waiting for %s (%s/%s, last=%s)\r' "$url" "$i" "$tries" "$code"
    sleep 2; (( i++ ))
  done
  printf '\n'; return 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
log "Checking the '$A3S_POWERGRID_AGENT' agent is configured"
a3s_require_agent "$A3S_POWERGRID_AGENT" || exit 1

if docker ps -a --format '{{.Names}}' | grep -qx "$A3S_CONTAINER"; then
  log "Removing the existing $A3S_CONTAINER container"
  docker rm -f "$A3S_CONTAINER" >/dev/null
  ok "removed"
fi

if (exec 3<>"/dev/tcp/127.0.0.1/${A3S_PORT}") 2>/dev/null; then
  die "port $A3S_PORT is already in use — free it, or run with A3S_PORT=<port>"
fi

# Always build: the build is cache-aware, so unchanged sources return in
# seconds, while reusing a stale image silently runs code from another branch.
log "Building $A3S_IMAGE (the first build pulls torch / Grid2Op / LightSim2Grid and takes a while)"
docker build "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}" \
  --build-arg "A3S_POWERGRID_AGENT=$A3S_POWERGRID_AGENT" \
  -f "$REPO_ROOT/docker/Dockerfile" \
  -t "$A3S_IMAGE" "$REPO_ROOT"
ok "image built"

log "Starting $A3S_CONTAINER on port $A3S_PORT"
docker run -d --name "$A3S_CONTAINER" -p "${A3S_PORT}:5010" \
  -e FLASK_APP="app:create_app('dev')" \
  -e "A3S_POWERGRID_AGENT=$A3S_POWERGRID_AGENT" \
  "$A3S_IMAGE" >/dev/null
ok "container started (PowerGrid served by the '$A3S_POWERGRID_AGENT' agent)"

log "Waiting for the service"
wait_for_http "http://localhost:${A3S_PORT}/api/v1/health" 200 \
  || die "A3S did not come up — check: docker logs $A3S_CONTAINER"
ok "healthy: $(curl -s "http://localhost:${A3S_PORT}/api/v1/health")"

# The URL InteractiveAI has to call is the host.docker.internal one, not
# localhost: its recommendation service runs in a container of its own, where
# localhost is that container, not this host.
A3S_URL="http://host.docker.internal:${A3S_PORT}/api/v1/recommendation"

printf '\n\033[1;32mA3S is running.\033[0m\n\n'
printf '  Health      http://localhost:%s/api/v1/health\n'              "$A3S_PORT"
printf '  Recommend   POST http://localhost:%s/api/v1/recommendation\n' "$A3S_PORT"
printf '  Stop        ./docker/local_stop.sh\n'
printf '\n  For InteractiveAI:  %s\n' "$A3S_URL"
printf '\nPass that URL to --a3s, from the InteractiveAI repo root:\n\n'
if [[ "$A3S_PORT" == "5010" ]]; then
  printf '  ./local_setup.sh --a3s                 # 5010 is the default, no URL needed\n\n'
else
  printf '  ./local_setup.sh --a3s %s\n\n' "$A3S_URL"
fi
