#!/usr/bin/env bash
set -euo pipefail

retry_failures=false
if [[ "${1:-}" == "--retry-failures" ]]; then
  retry_failures=true
  shift
fi

run_tests() {
  local report="$1"
  shift
  uv run --frozen pytest "$@" --asyncio-mode=auto -n auto -rs \
    --junitxml="$RUNNER_TEMP/provider-evidence-$report.xml"
}

status=0
run_tests initial "$@" || status=$?
if [[ "$status" == 5 ]]; then
  echo "::warning::No tests collected; provider is untested."
  exit 0
fi
if [[ "$status" == 1 && "$retry_failures" == true ]]; then
  echo "Retrying only failed live-provider tests once."
  run_tests retry "$@" --last-failed
else
  exit "$status"
fi
