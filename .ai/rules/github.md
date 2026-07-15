# GitHub Module Rules

## Purpose
The `github/` module handles all GitHub API integration: fetching PR data, posting reviews, and webhook handling.

## File Structure

```
github/
├── __init__.py      # Public exports
├── client.py        # GitHub API wrapper (reviews, delta, doc-bot comments, repo probing)
├── webhook.py       # FastAPI webhook handlers
└── formatter.py     # Review output formatting
```

## Key Types

```python
class GitHubClient:
    """Wrapper around PyGitHub for PR operations."""
    
    async def get_pr_diff(self, repo: str, pr_number: int) -> str: ...
    async def get_changed_files(self, repo: str, pr_number: int) -> dict[str, str]: ...
    async def post_review(self, repo: str, pr_number: int, review: ConsolidatedReview): ...
    async def post_inline_comments(self, repo: str, pr_number: int, findings: list): ...

    # Documentation review support
    def probe_repo_paths(self, repo_name: str, ref: str, paths: list[str]) -> set[str]: ...
    def find_doc_bot_comment(self, pr: PullRequest, marker: str) -> int | None: ...
    def post_or_update_doc_comment(self, pr: PullRequest, body: str, marker: str) -> None: ...

class ReviewFormatter:
    """Formats ConsolidatedReview for different outputs."""
    
    def to_github_markdown(self, review: ConsolidatedReview) -> str: ...
    def to_json(self, review: ConsolidatedReview) -> dict: ...
    def to_cli(self, review: ConsolidatedReview) -> str: ...
    def format_all_agents_failed(self) -> str: ...
```

## Invariants

### G1: Async GitHub Operations
All GitHub API calls are async. Use `httpx` or async PyGitHub wrapper.

### G2: Rate Limiting Awareness
Respect GitHub API rate limits. Use exponential backoff on 429 errors.

### G3: Minimal Data Fetching
Only fetch what's needed. Don't pull entire repo for single PR review.

### G4: Webhook Security
Always validate webhook signatures using `X-Hub-Signature-256` header.

### G5: No Agent Logic in GitHub Module
This module fetches data and posts results. Review logic is in orchestrator.

### G6: Doc-Bot Comment Deduplication
`post_or_update_doc_comment` uses an HTML comment marker (`<!-- AI-CODE-REVIEWER-DOC-BOT -->`) to find and update an existing comment instead of creating duplicates. `find_doc_bot_comment` searches issue comments for the marker. `probe_repo_paths` checks whether convention files and architecture directories exist in the repo (short list, ~6-8 items, one `get_contents` call each).

### G7: Concurrency Gating
An `asyncio.Semaphore` (`MAX_CONCURRENT_REVIEWS`, default 2) gates all in-process reviews in the webhook handler. This semaphore is shared by both the full review and any LGTM re-check operations to prevent resource exhaustion.

### G8: Never Silent All-Agents-Failure
When all review agents fail, a visible `⚠️ Review could not complete` comment is posted (action `COMMENT`, never `APPROVE`). This is a terminal outcome that prevents silent failures.

### G9: Raise on Credential Missing
The `run_review()` function raises `RuntimeError` instead of logging and returning when credentials (GitHub token or Anthropic API key) are missing. This ensures proper retry semantics for queued jobs and explicit failure in inline mode.
FIND>>>
<<<REPLACE
### G6: Doc-Bot Comment Deduplication
`post_or_update_doc_comment` uses an HTML comment marker (`<!-- AI-CODE-REVIEWER-DOC-BOT -->`) to find and update an existing comment instead of creating duplicates. `find_doc_bot_comment` searches issue comments for the marker. `probe_repo_paths` checks whether convention files and architecture directories exist in the repo (short list, ~6-8 items, one `get_contents` call each).

### G7: Concurrency Gating
An `asyncio.Semaphore` (`MAX_CONCURRENT_REVIEWS`, default 2) gates all in-process reviews in the webhook handler. This semaphore is shared by both the full review and any LGTM re-check operations to prevent resource exhaustion.

### G8: Never Silent All-Agents-Failure
When all review agents fail, a visible `⚠️ Review could not complete` comment is posted (action `COMMENT`, never `APPROVE`). This is a terminal outcome that prevents silent failures.

### G9: Raise on Credential Missing
The `run_review()` function raises `RuntimeError` instead of logging and returning when credentials (GitHub token or Anthropic API key) are missing. This ensures proper retry semantics for queued jobs and explicit failure in inline mode.

## Webhook Handling

```python
import hashlib
import hmac

def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature using timing-safe comparison.
    
    IMPORTANT: Must use hmac.compare_digest() to prevent timing attacks.
    """
    if not signature or not signature.startswith("sha256="):
        return False
    
    expected = "sha256=" + hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    # Timing-safe comparison prevents timing attacks
    return hmac.compare_digest(expected, signature)

async def run_review(repo: str, pr_number: int) -> ConsolidatedReview:
    """Execute review for a PR.
    
    Raises RuntimeError if credentials are missing, enabling proper
    retry semantics for queued jobs and explicit failure in inline mode.
    """
    # Validation and execution; no silent fallback
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN not configured")
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")
    
    # ... perform review ...

@app.post("/webhook")
async def github_webhook(request: Request):
    # 1. Validate signature (timing-safe)
    signature = request.headers.get("X-Hub-Signature-256")
    body = await request.body()
    if not verify_webhook_signature(body, signature, WEBHOOK_SECRET):
        raise HTTPException(401, "Invalid signature")
    
    # 2. Parse event
    event_type = request.headers.get("X-GitHub-Event")
    payload = await request.json()
    
    # 3. Handle relevant events
    if event_type == "pull_request":
        action = payload["action"]
        if action in ("opened", "synchronize", "reopened"):
            # Enqueue or run review with concurrency gating
            asyncio.create_task(handle_pr_event(payload))
    
    return {"status": "ok"}

@app.post("/process-review")
async def process_review(request: Request):
    """Cloud Tasks endpoint for deferred review execution.
    
    Validates X-Task-Auth header, deduplicates on head_sha, invokes
    run_review, and posts failure notice on exhausted retries.
    """
    auth = request.headers.get("X-Task-Auth")
    if not auth or auth != TASK_AUTH_TOKEN:
        raise HTTPException(401, "Unauthorized")
    
    payload = await request.json()
    repo = payload["repo"]
    pr_number = payload["pr_number"]
    head_sha = payload["head_sha"]
    
    if _already_reviewed(repo, pr_number, head_sha):
        return {"status": "ok", "deduped": True}
    
    try:
        await run_review(repo, pr_number)
        return {"status": "ok"}
    except Exception as e:
        retry_count = int(request.headers.get("X-CloudTasks-TaskRetryCount", 0))
        if retry_count >= TASK_MAX_ATTEMPTS - 1:
            # Final attempt: post visible failure notice
            await _post_review_failed_notice(repo, pr_number, str(e))
            logger.info(f"review-job-dead repo={repo} pr={pr_number}")
            return {"status": "ok"}
        # Intermediate attempt: return 500 to trigger retry
        raise HTTPException(500, "Review failed, will retry")
```

## Review Actions

| Scenario | GitHub Review Action |
|----------|---------------------|
| Critical findings | `REQUEST_CHANGES` |
| Warnings/suggestions only | `COMMENT` |
| No findings | `APPROVE` (if configured) |
| Review failed | Comment with error, no action |

## Output Format (GitHub Markdown)

```markdown
## 🤖 AI Code Review

**Reviewed by {N} agents** | Consensus score: {X}%

### 🔴 Critical Issues ({count})
[findings...]

### 🟡 Warnings ({count})
<details><summary>Click to expand</summary>
[findings...]
</details>

### 💡 Suggestions ({count})
<details><summary>Click to expand</summary>
[findings...]
</details>

---
<sub>Generated by AI Code Reviewer</sub>
```

## Error Handling

```python
async def post_review_safely(repo: str, pr: int, review: ConsolidatedReview):
    try:
        await self.post_review(repo, pr, review)
    except RateLimitExceeded:
        # Wait and retry
        await asyncio.sleep(60)
        await self.post_review(repo, pr, review)
    except PermissionDenied:
        # Log but don't crash
        logger.error(f"No permission to post review to {repo}#{pr}")
    except Exception as e:
        # Post error comment as fallback
        await self.post_comment(
            repo, pr, 
            f"⚠️ AI Review failed: {e}\n\nPlease check logs."
        )
```

## Configuration

```yaml
github:
  token: ${GITHUB_TOKEN}
  webhook_secret: ${GITHUB_WEBHOOK_SECRET}
  
  # Optional: GitHub App auth (for higher rate limits)
  app_id: ${GITHUB_APP_ID}
  private_key_path: ./github-app.pem

concurrency:
  max_concurrent_reviews: ${MAX_CONCURRENT_REVIEWS:-2}

task_queue:
  # Enable Cloud Tasks durable queue mode (optional)
  queue_path: ${TASK_QUEUE_PATH}
  target_url: ${TASK_TARGET_URL}
  auth_token: ${TASK_AUTH_TOKEN}
  max_attempts: ${TASK_MAX_ATTEMPTS:-4}
```

## Anti-Patterns

1. **Don't embed review logic** - Only fetch/post, delegate to orchestrator
2. **Don't ignore rate limits** - Implement proper backoff
3. **Don't skip signature validation** - Security critical
4. **Don't block webhook handler** - Return quickly, process async
5. **Don't hardcode repos/permissions** - Use configuration
6. **Don't create duplicate doc-bot comments** - Always search for the marker first via `find_doc_bot_comment`
7. **Don't silently fail when all agents error** - Post `format_all_agents_failed()` comment and log
8. **Don't log and return on missing credentials** - Raise `RuntimeError` to enable retry semantics
9. **Don't exceed concurrency limits** - Always acquire `_get_review_semaphore()` before review execution
10. **Don't fail silently in queued mode** - Post visible failure notice on final attempt exhaustion
