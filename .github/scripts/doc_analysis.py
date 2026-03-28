"""Analyze changed files for documentation impact.

Reads CHANGED_FILES and GITHUB_OUTPUT from environment variables.
Writes suggestions in GitHub Actions heredoc output format.
"""

import os
import sys
from pathlib import Path

changed_files_str = os.environ.get("CHANGED_FILES", "")
changed = [f.strip() for f in changed_files_str.strip().split("\n") if f.strip()]
suggestions: list[dict[str, str]] = []

for filepath in changed:
    if not all(c.isalnum() or c in "._-/" for c in filepath):
        print(
            f"Warning: Skipping file with unexpected characters (length={len(filepath)})",
            file=sys.stderr,
        )
        continue

    path = Path(filepath)

    if path.suffix == ".py" and path.parent.name in [
        "agents",
        "orchestrator",
        "github",
        "models",
    ]:
        suggestions.append(
            {
                "file": f".ai/rules/{path.parent.name}.md",
                "reason": f"File `{filepath}` changed - may need documentation update",
                "priority": "normal",
            }
        )

    if "config" in filepath.lower():
        suggestions.append(
            {
                "file": "config.example.yaml",
                "reason": "Configuration changes may need example updates",
                "priority": "normal",
            }
        )
        suggestions.append(
            {
                "file": "README.md",
                "reason": "Configuration changes may affect README docs",
                "priority": "normal",
            }
        )

    if "cli" in filepath.lower():
        suggestions.append(
            {
                "file": "README.md",
                "reason": "CLI changes may need README command documentation update",
                "priority": "normal",
            }
        )

seen_files: set[str] = set()
unique: list[dict[str, str]] = []
for s in suggestions:
    if s["file"] not in seen_files:
        seen_files.add(s["file"])
        unique.append(s)

output_file = os.environ.get("GITHUB_OUTPUT", "/dev/stdout")
with open(output_file, "a") as f:
    f.write("SUGGESTIONS<<EOF\n")
    if unique:
        f.write("<!-- AI-CODE-REVIEWER-DOC-BOT -->\n")
        f.write("## 📚 Documentation Check\n\n")
        f.write("The following documentation files may need updates based on code changes:\n\n")
        for s in unique:
            f.write(f"- **{s['file']}**: {s['reason']} [{s['priority']}]\n")
        f.write("\n*Please review these suggestions and update documentation if needed.*\n")
    f.write("EOF\n")
