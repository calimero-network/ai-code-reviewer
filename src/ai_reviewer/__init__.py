"""AI Code Reviewer - Multi-agent code review system."""

from importlib.metadata import version

# Read from the installed distribution so it cannot drift from pyproject.
__version__ = version("ai-code-reviewer")
