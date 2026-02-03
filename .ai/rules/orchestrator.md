# Orchestrator Module Rules

## Purpose
The `orchestrator/` module handles parallel agent execution and result aggregation. It's the coordination layer between raw agents and final output.

## Key Types

```python
@dataclass
class OrchestratorConfig:
    timeout_seconds: int = 120
    min_agents_required: int = 2
    max_parallel_agents: int = 5
    retry_on_failure: bool = True
    max_retries: int = 2

class AgentOrchestrator:
    """Runs multiple agents in parallel, handles failures."""
    
class ReviewAggregator:
    """Combines agent results via clustering and consensus scoring."""
```

## File Structure

```
orchestrator/
├── __init__.py        # Public exports
├── orchestrator.py    # AgentOrchestrator class
└── aggregator.py      # ReviewAggregator class
```

## Orchestrator Invariants

### O1: Parallel Execution
All agents run concurrently via `asyncio.gather()`. Never sequential.

### O2: Timeout Per Agent
Each agent has individual timeout. One slow agent doesn't block others.

### O3: Minimum Agent Threshold
Review fails only if fewer than `min_agents_required` succeed.

### O4: No Agent Coupling
Orchestrator doesn't know agent internals. Treats all as `ReviewAgent`.

## Aggregator Invariants

### O5: Semantic Clustering
Similar findings clustered by meaning, not exact text match.

### O6: Consensus Score Calculation
```python
consensus_score = len(agreeing_agents) / total_agents
```

### O7: Priority Ordering
Final findings sorted by: `severity_weight × consensus_score`

### O8: Deduplication
Same finding from multiple agents = single output with higher confidence.

## Orchestration Flow

```python
async def review(self, diff, files, context):
    # 1. Create parallel tasks for all agents
    tasks = [
        asyncio.create_task(
            self._run_agent_with_timeout(agent, diff, files, context)
        )
        for agent in self.agents
    ]
    
    # 2. Wait for all, collecting exceptions
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 3. Filter successful reviews
    successful = [r for r in results if isinstance(r, AgentReview)]
    
    # 4. Check threshold
    if len(successful) < self.config.min_agents_required:
        raise InsufficientAgentsError(...)
    
    return successful
```

## Aggregation Algorithm

```python
def aggregate(self, reviews: list[AgentReview]) -> ConsolidatedReview:
    # 1. Extract all findings from all agents
    all_findings = self._extract_all_findings(reviews)
    
    # 2. Cluster by semantic similarity (embeddings or text)
    clusters = self._cluster_similar_findings(all_findings)
    
    # 3. For each cluster, create consolidated finding
    consolidated = []
    for cluster in clusters:
        merged = self._merge_cluster(cluster)
        merged.consensus_score = len(cluster) / len(reviews)
        consolidated.append(merged)
    
    # 4. Sort by priority
    consolidated.sort(key=self._priority_score, reverse=True)
    
    return ConsolidatedReview(findings=consolidated, ...)
```

## Error Handling

| Error | Behavior |
|-------|----------|
| Agent timeout | Log warning, continue with others |
| Agent exception | Log error, continue with others |
| Below min threshold | Raise `InsufficientAgentsError` |
| All agents fail | Raise with all error details |

## Configuration Tuning

```yaml
orchestrator:
  timeout_seconds: 120      # Increase for large PRs
  min_agents_required: 2    # Lower = more tolerant of failures
  max_parallel_agents: 5    # Limit concurrent API calls

aggregator:
  similarity_threshold: 0.85  # Higher = stricter clustering
  min_consensus_for_critical: 0.5  # % agreement for critical findings
```

## Anti-Patterns

1. **Don't run agents sequentially** - Always parallel
2. **Don't fail fast on first error** - Collect all results first
3. **Don't couple to specific agent types** - Generic `ReviewAgent` interface
4. **Don't ignore partial results** - Some findings better than none
5. **Don't hardcode thresholds** - Use configuration
