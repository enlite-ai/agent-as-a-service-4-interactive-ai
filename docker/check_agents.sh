#!/usr/bin/env bash
#
# check_agents.sh — is resources/PowerGrid/ laid out so an agent can actually run?
#
# resources/PowerGrid/ is not committed (README step 1). Anything missing there
# only surfaces as a crash on the first recommendation request, so check it up
# front. Run it on its own, or let docker/local_setup.sh call it for you.
#
# Usage (from a3s-service/):
#   ./docker/check_agents.sh          # report on every agent; fails if none is usable
#   ./docker/check_agents.sh t2.1     # check just that one

A3S_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A3S_RES="$A3S_ROOT/resources/PowerGrid"
A3S_SIM_RES="$A3S_RES/PowerGridgrid2op_poc_simulator"

AGENTS=(t2.1 xd)

# Files each agent needs, on top of the Grid2Op env data every agent needs.
agent_requires() {
  case "$1" in
    t2.1) printf '%s\n' \
            "$A3S_RES/T2.1_deep_expert/PPO_SB3/model/PPO_SB3.zip" \
            "$A3S_RES/T2.1_deep_expert/ExpertAgent/assets/whole_action_space.npz" ;;
    xd)   printf '%s\n' "$A3S_SIM_RES/XD_silly_repo/submission" ;;
    *)    return 1 ;;
  esac
}

# The Grid2Op environment itself — without it no agent can run.
env_missing() {
  [[ -f "$A3S_SIM_RES/CONFIG_POWERGRID.toml"   ]] || printf '%s\n' "$A3S_SIM_RES/CONFIG_POWERGRID.toml"
  [[ -d "$A3S_SIM_RES/env_ICAPS_input_data_test" ]] || printf '%s\n' "$A3S_SIM_RES/env_ICAPS_input_data_test/"
}

# Paths $1 needs that are not there. Empty output = the agent is ready.
agent_missing() {
  local path
  env_missing
  while read -r path; do
    [[ -e "$path" ]] || printf '%s\n' "$path"
  done < <(agent_requires "$1")
}

# Fail with the list of what to fetch unless $1 is ready. Used by local_setup.sh.
a3s_require_agent() {
  local agent="$1" missing
  agent_requires "$agent" >/dev/null 2>&1 \
    || { printf '\033[1;31m    ✗ unknown agent '\''%s'\'' — use %s\033[0m\n' "$agent" "${AGENTS[*]}" >&2; return 1; }

  missing="$(agent_missing "$agent")"
  if [[ -z "$missing" ]]; then
    printf '\033[1;32m    ✓ the '\''%s'\'' agent is configured\033[0m\n' "$agent"
    return 0
  fi

  printf '\033[1;33m    ! the '\''%s'\'' agent is not configured — these are missing:\033[0m\n' "$agent" >&2
  printf '\033[1;33m      %s\033[0m\n' $missing >&2
  printf '\033[1;31m    ✗ fetch them first — see step 1 of "Run it" in a3s-service/README.md\033[0m\n' >&2
  return 1
}

# Report on every agent. Exit 0 as long as one of them can serve.
a3s_report_agents() {
  local agent missing usable=0
  printf '\nAgents in %s:\n\n' "$A3S_RES"
  for agent in "${AGENTS[@]}"; do
    missing="$(agent_missing "$agent")"
    if [[ -z "$missing" ]]; then
      printf '  \033[1;32m✓ %-5s ready\033[0m\n' "$agent"
      usable=1
    else
      printf '  \033[1;33m✗ %-5s missing:\033[0m\n' "$agent"
      printf '        %s\n' $missing
    fi
  done
  printf '\n'
  if (( usable )); then
    printf '\033[1;32mAt least one agent is configured — ./docker/local_setup.sh can start.\033[0m\n'
    printf 'Select one with --agent <name> (default t2.1).\n\n'
    return 0
  fi
  printf '\033[1;31mNo agent is configured.\033[0m Fetch the files listed above —\n'
  printf 'see step 1 of "Run it" in a3s-service/README.md.\n\n'
  return 1
}

# Sourced by local_setup.sh for the functions above; run directly, it reports.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if (( $# )); then
    a3s_require_agent "$1"
  else
    a3s_report_agents
  fi
fi
