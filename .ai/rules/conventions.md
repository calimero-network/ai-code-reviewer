# Coding Conventions

## Purpose
Consistent conventions that all code in this repository should follow.

## Python Version
- **Python 3.11+** required
- Use modern type hints: `list[str]` not `List[str]`, `str | None` not `Optional[str]`

## Type Hints

### Always Type
```python
# ✅ Correct
def process_review(diff: str, files: dict[str, str]) -> AgentReview:
    ...

# ❌ Wrong
def process_review(diff, files):
    ...
```

### Use Type Aliases for Complex Types
```python
# ✅ Good - clear intent
FileContents = dict[str, str]
FindingList = list[ReviewFinding]

def review(files: FileContents) -> FindingList:
    ...
```

## Async/Await

### All I/O Operations Are Async
```python
# ✅ Correct
async def fetch_pr(self, pr_number: int) -> dict:
    response = await self.client.get(...)
    return response.json()

# ❌ Wrong - blocking call
def fetch_pr(self, pr_number: int) -> dict:
    response = requests.get(...)  # Blocks!
    return response.json()
```

### Use asyncio.gather for Parallel
```python
# ✅ Correct - parallel execution
results = await asyncio.gather(
    agent1.review(diff),
    agent2.review(diff),
    agent3.review(diff),
)

# ❌ Wrong - sequential
results = [
    await agent1.review(diff),
    await agent2.review(diff),
    await agent3.review(diff),
]
```

## Dataclasses

### Use for All Data Structures
```python
from dataclasses import dataclass, field

@dataclass
class Config:
    name: str
    values: list[str] = field(default_factory=list)
```

### Avoid Mutable Default Arguments
```python
# ✅ Correct
@dataclass
class Finding:
    tags: list[str] = field(default_factory=list)

# ❌ Wrong - shared mutable state
@dataclass
class Finding:
    tags: list[str] = []  # Bug! Shared across instances
```

## Logging

### Use Module-Level Logger
```python
import logging

logger = logging.getLogger(__name__)

class MyAgent:
    def review(self, diff: str):
        logger.info(f"Starting review of {len(diff)} chars")
        try:
            ...
        except Exception as e:
            logger.error(f"Review failed: {e}")
            raise
```

### Log Levels
| Level | Use Case |
|-------|----------|
| DEBUG | Detailed debugging info |
| INFO | Normal operation milestones |
| WARNING | Recoverable issues |
| ERROR | Failures that affect output |

## Error Handling

### Specific Exceptions
```python
# ✅ Correct - specific exception
class InsufficientAgentsError(Exception):
    """Raised when too few agents succeed."""
    pass

# ❌ Wrong - generic exception
raise Exception("Not enough agents")
```

### Don't Swallow Exceptions
```python
# ✅ Correct - log and re-raise or handle
try:
    result = await agent.review(diff)
except Exception as e:
    logger.error(f"Agent failed: {e}")
    raise  # Or handle appropriately

# ❌ Wrong - silent failure
try:
    result = await agent.review(diff)
except Exception:
    pass  # Bug hidden!
```

## Imports

### Order
1. Standard library
2. Third-party packages
3. Local imports

```python
# Standard library
import asyncio
import logging
from dataclasses import dataclass

# Third party
import httpx
from pydantic import BaseModel

# Local
from ai_reviewer.models import ReviewFinding
from ai_reviewer.agents.base import ReviewAgent
```

### Use Absolute Imports
```python
# ✅ Correct
from ai_reviewer.models.findings import ReviewFinding

# ❌ Avoid relative imports
from ..models.findings import ReviewFinding
```

## Naming

### Classes
- PascalCase: `ReviewAgent`, `ConsolidatedFinding`

### Functions/Methods
- snake_case: `aggregate_findings`, `get_pr_diff`

### Constants
- UPPER_SNAKE: `DEFAULT_TIMEOUT`, `MAX_RETRIES`

### Private Members
- Single underscore prefix: `_parse_response`, `self._client`

## Documentation

### Docstrings for Public API
```python
def aggregate(self, reviews: list[AgentReview]) -> ConsolidatedReview:
    """Combine multiple agent reviews into a unified review.
    
    Args:
        reviews: List of completed agent reviews
        
    Returns:
        Consolidated review with deduplicated, scored findings
        
    Raises:
        ValueError: If reviews list is empty
    """
```

### Comments for Why, Not What
```python
# ✅ Good - explains why
# Use embeddings for clustering because text matching
# fails on semantically similar but differently worded findings
clusters = self._cluster_by_embeddings(findings)

# ❌ Bad - states the obvious
# Loop through findings
for finding in findings:
```

## Testing

### Test File Naming
- Test file: `test_<module>.py`
- Test class: `Test<ClassName>`
- Test method: `test_<behavior>`

```python
# tests/test_orchestrator.py
class TestAgentOrchestrator:
    async def test_returns_results_when_all_agents_succeed(self):
        ...
    
    async def test_raises_when_below_min_threshold(self):
        ...
```

### Use Fixtures
```python
@pytest.fixture
def mock_cursor_client():
    client = Mock(spec=CursorClient)
    client.complete_json = AsyncMock(return_value={"findings": []})
    return client
```

## Tools

### Linting: Ruff
```bash
ruff check .
ruff format .
```

### Type Checking: MyPy
```bash
mypy src/
```

### Testing: Pytest
```bash
pytest -v
pytest --cov=ai_reviewer
```
