#!/usr/bin/env bash
# `claude plugin update` compares version strings, not content: shipping a changed
# skill under an unchanged version leaves every existing install on the old copy.
set -euo pipefail

base="${1:?usage: check-plugin-version.sh <base-ref>}"
plugin_paths=(.claude/skills .claude/agents .claude/.claude-plugin)

if git diff --quiet "$base"...HEAD -- "${plugin_paths[@]}"; then
    exit 0
fi

read_version() { sed -n 's/^version = "\(.*\)"/\1/p' | head -1; }
before=$(git show "$base:pyproject.toml" | read_version)
after=$(read_version < pyproject.toml)

if [ "$before" = "$after" ]; then
    echo "Plugin content changed but pyproject version is still $after."
    echo "Bump it, or installed plugins stay on the old copy:"
    git diff --name-only "$base"...HEAD -- "${plugin_paths[@]}" | sed 's/^/  /'
    exit 1
fi

echo "Plugin content changed and version moved $before -> $after."
