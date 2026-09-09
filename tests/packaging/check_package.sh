#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

# The default build creates an sdist, then builds the wheel from that sdist.
uv build --default-index https://pypi.org/simple --out-dir "$tmp_dir/dist" "$repo_root"
wheels=("$tmp_dir"/dist/*.whl)
uv venv --python "${PACKAGE_PYTHON:-3.11}" "$tmp_dir/.venv"
constraints=()
if [ "${PACKAGE_SDK:-latest}" = minimum ]; then
    # Pin the declared SDK/model floors, allowing their transitive dependencies
    # (notably pydantic-core) to resolve to compatible versions.
    constraints=("openai==2.0.0" "pydantic==2.8.0")
fi
uv pip install --default-index https://pypi.org/simple \
    --python "$tmp_dir/.venv/bin/python" \
    "${wheels[0]}${PACKAGE_EXTRAS:+[$PACKAGE_EXTRAS]}" "${constraints[@]}"
uv pip check --python "$tmp_dir/.venv/bin/python"
cp "$repo_root/tests/packaging/contract.py" "$tmp_dir/contract.py"
cd "$tmp_dir"
unset PYTHONPATH
"$tmp_dir/.venv/bin/python" -I contract.py
# Help must not need credentials or contact a provider.
unset OPENAI_API_KEY OPENAI_ADMIN_KEY OPENAI_ORG_ID OPENAI_PROJECT_ID
for command in "" files jobs; do
    if [ -n "$command" ]; then
        "$tmp_dir/.venv/bin/instructor" "$command" --help >/dev/null
    else
        "$tmp_dir/.venv/bin/instructor" --help >/dev/null
    fi
done
