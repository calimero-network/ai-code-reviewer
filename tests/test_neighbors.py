from ai_reviewer.context.neighbors import (
    parse_imports_python,
    parse_imports_regex_go,
    parse_imports_regex_ts,
    select_neighbors,
)


def test_python_absolute_imports():
    src = "import os\nimport ai_reviewer.config\nfrom ai_reviewer.models import findings\n"
    imports = parse_imports_python(src)
    assert "os" in imports
    assert "ai_reviewer.config" in imports
    assert "ai_reviewer.models.findings" in imports


def test_python_relative_imports_resolved():
    src = "from . import sibling\nfrom .utils import helper\n"
    imports = parse_imports_python(src, current_module="ai_reviewer.agents.base")
    assert "ai_reviewer.agents.sibling" in imports
    assert "ai_reviewer.agents.utils.helper" in imports


def test_python_malformed_returns_empty():
    assert parse_imports_python("def f(:\n  ") == set()


def test_ts_imports():
    src = """
    import { foo } from "./utils";
    import bar from '../types';
    const q = require("./legacy");
    """
    out = parse_imports_regex_ts(src)
    assert "./utils" in out
    assert "../types" in out
    assert "./legacy" in out


def test_go_imports_single_and_block():
    src = """
package a
import "fmt"
import (
    "strings"
    "my/pkg"
)
"""
    out = parse_imports_regex_go(src)
    assert out == {"fmt", "strings", "my/pkg"}


def test_select_neighbors_siblings_and_outbound_imports():
    changed = {"src/app/user.py": "from .auth import verify\n"}
    repo_paths = [
        "src/app/user.py",
        "src/app/auth.py",
        "src/app/profile.py",
        "src/models/__init__.py",
        "src/unrelated/other.py",
    ]

    def read(path: str) -> str:
        return changed.get(path, "")

    neighbors = select_neighbors(
        changed_files=changed,
        repo_paths=repo_paths,
        read_file=read,
        max_siblings=2,
        max_total=6,
    )
    assert "src/app/auth.py" in neighbors
    assert "src/app/profile.py" in neighbors
    assert "src/unrelated/other.py" not in neighbors
