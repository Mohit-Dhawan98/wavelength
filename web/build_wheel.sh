#!/bin/sh
# Build the engine wheel into web/wheel/ so the page can load it. Run from the repo root.
set -e
cd "$(dirname "$0")/.."
mkdir -p web/wheel
find web/wheel -name '*.whl' -delete
uv build --wheel -o web/wheel -q
file=$(basename "$(ls web/wheel/*.whl)")
printf '{"file": "%s"}\n' "$file" > web/wheel/index.json
echo "web/wheel/$file"
