# Models Module Rules

## Purpose
The `models/` module contains all data structures used across the system. These are pure data containers with no business logic.

## File Structure

```
models/
├── __init__.py       # Re-exports public types
├── context.py        # ReviewContext - input context
├── findings.py       # ReviewFinding, Severity, Category
└── review.py         # AgentReview, ConsolidatedReview
```

## Core Types

### ReviewContext (Input)
```python
@dataclass
class ReviewContext:
    """Context provided to agents."""
    repo_name: str
    pr_number: int
    pr_title: str
    pr_description: str
    base_branch: str
    head_branch: str
    author: str
    changed_files_count: int
    additions: int
    deletions: int
    labels: list[str]
    repo_languages: list[str]
    custom_instructions: str | None = None
```

### ReviewFinding (Agent Output)
```python
@dataclass
class ReviewFinding:
    """Single finding from an agent."""
    file_path: str
    line_start: int
    line_end: int | None
    severity: Severity
    category: Category
    title: str
    description: str
    suggested_fix: str | None
    confidence: float  # 0.0-1.0
```

### Enums
```python
class Severity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    NITPICK = "nitpick"

class Category(Enum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    LOGIC = "logic"
    STYLE = "style"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
```

## Invariants

### M1: Dataclasses Only
All models are `@dataclass` or `Enum`. No class methods beyond serialization.

### M2: No Business Logic
Models don't contain validation or transformation logic. That belongs in services.

### M3: Immutable by Convention
Treat all models as immutable after creation. Don't mutate instances.

### M4: Type-Safe Enums
Always use `Severity` and `Category` enums, never raw strings.

### M5: Optional Fields Explicit
Use `field_name: Type | None = None` for optional fields, never implicit.

## Creating New Models

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class NewModel:
    """Describe what this model represents."""
    
    # Required fields first
    required_field: str
    another_required: int
    
    # Optional fields with defaults last
    optional_field: str | None = None
    list_field: list[str] = field(default_factory=list)
```

## Serialization Pattern

Models should be JSON-serializable for API responses:

```python
@dataclass
class MyModel:
    value: str
    severity: Severity
    
    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "severity": self.severity.value,  # Enum to string
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MyModel":
        return cls(
            value=data["value"],
            severity=Severity(data["severity"]),  # String to enum
        )
```

## Type Relationships

```
ReviewContext ──────────────────────────────────┐
                                                │
     ┌──────────────────────────────────────────┼───────┐
     │           AgentOrchestrator              │       │
     │                  │                       ▼       │
     │    ┌─────────────┼─────────────┐    uses for    │
     │    ▼             ▼             ▼    review      │
     │ Agent1        Agent2        Agent3              │
     │    │             │             │                │
     │    ▼             ▼             ▼                │
     │ AgentReview  AgentReview  AgentReview           │
     │    │             │             │                │
     │    └─────────────┼─────────────┘                │
     │                  ▼                              │
     │           ReviewAggregator                      │
     │                  │                              │
     │                  ▼                              │
     │         ConsolidatedReview                      │
     │         (with ConsolidatedFinding[])            │
     └─────────────────────────────────────────────────┘
```

## Anti-Patterns

1. **Don't add methods to dataclasses** - Keep them as pure data
2. **Don't use dicts for structured data** - Create typed models
3. **Don't use strings for enums** - Use `Severity`/`Category` types
4. **Don't nest too deeply** - Flatten when possible
5. **Don't use mutable defaults** - Use `field(default_factory=...)`
