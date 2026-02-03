"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path

# Sample diffs for testing
SAMPLE_SECURE_DIFF = """\
diff --git a/auth/login.py b/auth/login.py
index 1234567..abcdefg 100644
--- a/auth/login.py
+++ b/auth/login.py
@@ -10,6 +10,12 @@ def authenticate(username: str, password: str) -> bool:
     hashed = hash_password(password)
     return db.verify_user(username, hashed)
+
+def get_user(user_id: int) -> dict:
+    \"\"\"Fetch user by ID using parameterized query.\"\"\"
+    query = "SELECT * FROM users WHERE id = %s"
+    return db.execute(query, (user_id,))
"""

SAMPLE_VULNERABLE_DIFF = """\
diff --git a/auth/login.py b/auth/login.py
index 1234567..abcdefg 100644
--- a/auth/login.py
+++ b/auth/login.py
@@ -10,6 +10,12 @@ def authenticate(username: str, password: str) -> bool:
     hashed = hash_password(password)
     return db.verify_user(username, hashed)
+
+def get_user(username: str) -> dict:
+    \"\"\"Fetch user by username.\"\"\"
+    query = f"SELECT * FROM users WHERE username = '{username}'"
+    return db.execute(query)
"""

SAMPLE_PERFORMANCE_DIFF = """\
diff --git a/utils/processor.py b/utils/processor.py
index 1234567..abcdefg 100644
--- a/utils/processor.py
+++ b/utils/processor.py
@@ -5,6 +5,15 @@ def process_items(items: list) -> list:
     return [transform(item) for item in items]
+
+def find_duplicates(items: list) -> list:
+    \"\"\"Find duplicate items in list.\"\"\"
+    duplicates = []
+    for i in range(len(items)):
+        for j in range(len(items)):
+            if i != j and items[i] == items[j] and items[i] not in duplicates:
+                duplicates.append(items[i])
+    return duplicates
"""


@pytest.fixture
def sample_secure_diff() -> str:
    """A diff with no security issues."""
    return SAMPLE_SECURE_DIFF


@pytest.fixture
def sample_vulnerable_diff() -> str:
    """A diff with SQL injection vulnerability."""
    return SAMPLE_VULNERABLE_DIFF


@pytest.fixture
def sample_performance_diff() -> str:
    """A diff with O(n²) performance issue."""
    return SAMPLE_PERFORMANCE_DIFF


@pytest.fixture
def sample_file_contents() -> dict[str, str]:
    """Sample file contents for context."""
    return {
        "auth/login.py": """\
import hashlib
from database import db

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username: str, password: str) -> bool:
    \"\"\"Authenticate a user.\"\"\"
    hashed = hash_password(password)
    return db.verify_user(username, hashed)
""",
        "utils/processor.py": """\
from typing import Any

def transform(item: Any) -> Any:
    return item

def process_items(items: list) -> list:
    return [transform(item) for item in items]
""",
    }


@pytest.fixture
def mock_review_context() -> dict:
    """Mock review context for testing."""
    return {
        "repo_name": "test-org/test-repo",
        "pr_number": 42,
        "pr_title": "Add user authentication",
        "pr_description": "This PR adds basic user authentication.",
        "base_branch": "main",
        "head_branch": "feature/auth",
        "author": "testuser",
        "changed_files_count": 2,
        "additions": 15,
        "deletions": 0,
        "labels": ["enhancement"],
        "repo_languages": ["python"],
    }


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"
