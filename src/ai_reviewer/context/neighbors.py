"""Heuristics for selecting non-changed files that clarify the diff.

Strategy:
- Siblings of each changed file (same directory).
- One-hop import graph: files imported by the changed file + files
  that import the changed file.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Callable
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)


def parse_imports_python(source: str, current_module: str | None = None) -> set[str]:
    """Return dotted module names imported by the source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level and current_module:
                parts = current_module.split(".")
                parts = parts[: -node.level] if node.level <= len(parts) else []
                prefix = ".".join(parts)
                base = f"{prefix}.{base}" if base else prefix
            for alias in node.names:
                name = alias.name
                full = f"{base}.{name}" if base else name
                imports.add(full)
    return imports


_TS_IMPORT_RE = re.compile(
    r"""(?mx)
    ^\s*(?:
        import\s+[^;]*?from\s+["']([^"']+)["']
      | import\s+["']([^"']+)["']
      | (?:const|let|var)\s+\S+\s*=\s*require\(\s*["']([^"']+)["']\s*\)
    )
    """,
)


def parse_imports_regex_ts(source: str) -> set[str]:
    """Match TS/JS import specifiers (paths or module names)."""
    out: set[str] = set()
    for m in _TS_IMPORT_RE.finditer(source):
        for group in m.groups():
            if group:
                out.add(group)
    return out


_GO_SINGLE_RE = re.compile(r'^\s*import\s+"([^"]+)"\s*$', re.MULTILINE)
_GO_BLOCK_RE = re.compile(r"import\s*\((.*?)\)", re.DOTALL)
_GO_BLOCK_ITEM_RE = re.compile(r'"([^"]+)"')


def parse_imports_regex_go(source: str) -> set[str]:
    out: set[str] = set()
    out.update(m.group(1) for m in _GO_SINGLE_RE.finditer(source))
    for block in _GO_BLOCK_RE.finditer(source):
        out.update(m.group(1) for m in _GO_BLOCK_ITEM_RE.finditer(block.group(1)))
    return out


_RUST_USE_RE = re.compile(r"^\s*use\s+([a-zA-Z0-9_:]+)", re.MULTILINE)


def parse_imports_regex_rust(source: str) -> set[str]:
    return {m.group(1) for m in _RUST_USE_RE.finditer(source)}


_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([a-zA-Z0-9_.]+)\s*;", re.MULTILINE)


def parse_imports_regex_java(source: str) -> set[str]:
    return {m.group(1) for m in _JAVA_IMPORT_RE.finditer(source)}


def _path_to_module(path: str) -> str | None:
    """Derive a dotted module name from a repo-relative .py path."""
    p = PurePosixPath(path)
    if p.suffix != ".py":
        return None
    parts = list(p.with_suffix("").parts)
    # Strip leading 'src/' if present (common Python layout)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def parse_imports_by_path(path: str, source: str) -> set[str]:
    """Dispatch by file extension."""
    p = PurePosixPath(path)
    ext = p.suffix.lower()
    if ext == ".py":
        return parse_imports_python(source, current_module=_path_to_module(path))
    if ext in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
        return parse_imports_regex_ts(source)
    if ext == ".go":
        return parse_imports_regex_go(source)
    if ext == ".rs":
        return parse_imports_regex_rust(source)
    if ext == ".java":
        return parse_imports_regex_java(source)
    return set()


def _module_to_possible_paths(module: str) -> list[str]:
    """Heuristic: turn 'ai_reviewer.models.findings' into possible path stems."""
    parts = module.split(".")
    stems = [
        "/".join(parts),
        "/".join(parts[:-1]) if len(parts) > 1 else parts[0],
    ]
    candidates: list[str] = []
    for stem in stems:
        for ext in (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java"):
            candidates.append(f"{stem}{ext}")
        candidates.append(f"{stem}/__init__.py")
        candidates.append(f"{stem}/index.ts")
        candidates.append(f"{stem}/index.js")
    return candidates


def select_neighbors(
    changed_files: dict[str, str],
    repo_paths: list[str],
    read_file: Callable[[str], str] | None = None,  # noqa: ARG001 — reserved for inbound-edge use
    max_siblings: int = 5,
    max_total: int = 20,
) -> list[str]:
    """Return paths of neighbor files (not already changed)."""
    changed_set = set(changed_files)
    repo_set = set(repo_paths)
    picks: list[str] = []

    def add(path: str) -> None:
        if path not in changed_set and path not in picks and path in repo_set:
            picks.append(path)

    for path in changed_files:
        parent = str(PurePosixPath(path).parent)
        siblings = [p for p in repo_paths if str(PurePosixPath(p).parent) == parent and p != path]
        for s in siblings[:max_siblings]:
            add(s)
            if len(picks) >= max_total:
                return picks

    for path, src in changed_files.items():
        imports = parse_imports_by_path(path, src)
        for module in imports:
            for cand in _module_to_possible_paths(module):
                if cand in repo_set:
                    add(cand)
                    if len(picks) >= max_total:
                        return picks

    return picks
