# CI Setup Guide

## How the doc auto-update works

When a PR merges to `main`, a CI job runs `ai-reviewer update-docs`. It detects stale documentation via the `source_to_docs_mapping` in each repo's `.ai-reviewer.yaml`, generates fully updated file content using Claude Sonnet, commits it to a new branch (`docs/auto-<sha>`), and opens a PR assigned to the original author. Nothing auto-merges — a human always reviews and merges the doc PR.

There are two ways to wire this up. The webhook server approach is better because it requires zero files in each repository.

---

## Option A — Webhook server (recommended, zero files per repo)

Deploy the `ai-reviewer serve` webhook server once, register a single org-level GitHub webhook, and every repo in the org gets automatic doc updates automatically. Repos that don't want this just leave `doc_generation.enabled: false` in their `.ai-reviewer.yaml`.

### 1. Deploy the webhook server

The server runs as a standard Python process. The simplest deployment is a single GCP Cloud Run service or a fly.io app — anything with a public HTTPS URL works.

**Using Docker:**

```bash
docker build -t ai-reviewer .
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e GITHUB_TOKEN=ghp_... \
  -e GITHUB_WEBHOOK_SECRET=your-secret \
  ai-reviewer \
  ai-reviewer serve --port 8080
```

The server exposes `POST /webhook` for GitHub events and `GET /health` for health checks.

**GCP Cloud Run (one command after building):**

```bash
gcloud run deploy ai-reviewer \
  --image gcr.io/YOUR_PROJECT/ai-reviewer \
  --platform managed \
  --region europe-west1 \
  --set-env-vars ANTHROPIC_API_KEY=sk-ant-...,GITHUB_TOKEN=ghp_...,GITHUB_WEBHOOK_SECRET=your-secret \
  --allow-unauthenticated
```

Note the public URL — you need it in step 2.

### 2. Register the org-level GitHub webhook

Go to: `https://github.com/organizations/calimero-network/settings/hooks` → **Add webhook**

| Field | Value |
|---|---|
| Payload URL | `https://your-server-url/webhook` |
| Content type | `application/json` |
| Secret | the same value you set as `GITHUB_WEBHOOK_SECRET` |
| Events | Select **Pull requests** and **Pushes** |

That's it. Every repo in the org now sends events to the server.

### 3. Enable per-repo in `.ai-reviewer.yaml`

In each repo where you want doc auto-updates, add:

```yaml
documentation:
  enabled: true
  source_to_docs_mapping:
    "src/ai_reviewer/cli.py": [README.md]
    "src/ai_reviewer/agents/**": [.ai/rules/agents.md]
    # add more mappings as needed

doc_generation:
  enabled: true
  model: claude-sonnet-4-6
  max_files: 5
```

Repos without `doc_generation.enabled: true` are silently skipped — no changes, no cost.

### What the server needs to handle push events

The existing webhook server (`src/ai_reviewer/github/webhook.py`) currently handles `pull_request` events. To handle `push` events for doc updates, you need to add a push handler. Open an issue or PR against this repo to track that work — or run Option B below in the meantime.

---

## Option B — Reusable GitHub Actions workflow (one small file per repo)

No server to deploy. Each repo adds a single workflow file that calls the shared reusable workflow hosted in this repo. All the logic lives here; the per-repo file is just a trigger and credential pass-through.

### 1. Add secrets to each repo (or at org level)

At org level (`github.com/organizations/calimero-network/settings/secrets/actions`):

| Secret | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `GH_PAT` | Classic PAT with `repo` scope (needed to open PRs across repos) |

### 2. Add one workflow file per repo

Create `.github/workflows/doc-update.yaml` in each target repo:

```yaml
name: Doc Auto-Update
on:
  push:
    branches: [main, master]
jobs:
  doc-update:
    uses: calimero-network/ai-code-reviewer/.github/workflows/doc-update.yaml@main
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      GH_PAT: ${{ secrets.GH_PAT }}
```

That's the entire file. All logic is in the shared workflow.

### 3. Enable per-repo in `.ai-reviewer.yaml`

Same as Option A step 3 above.

---

## Testing locally before wiring up CI

You can run `update-docs` against any real merged PR to verify it works end-to-end before setting up CI.

```bash
# Install
pip install -e ".[dev]"

# Set credentials
export ANTHROPIC_API_KEY=sk-ant-...
export GITHUB_TOKEN=ghp_...   # needs repo scope

# Dry run: see what would be generated, no branch created, no PR opened
ai-reviewer update-docs calimero-network/auth-frontend 42 --dry-run

# Real run: creates branch docs/auto-<sha> and opens a PR
ai-reviewer update-docs calimero-network/auth-frontend 42
```

`--dry-run` prints the first 60 lines of each generated file to stdout. Use this to sanity-check the AI output before letting it open real PRs.

---

## Running the test suite

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage report
pytest --cov=ai_reviewer --cov-report=term-missing

# Run only the doc-related tests
pytest tests/test_doc_analyzer.py -v

# Lint
ruff check .

# Type check
mypy src/
```

All three commands must pass before merging anything. The CI workflow (`.github/workflows/ci.yaml`) runs the same checks automatically on every PR.

---

## Secrets reference

| Secret | Where used | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude API calls | Required everywhere |
| `GITHUB_TOKEN` | Read PR data, post comments | Provided automatically by GitHub Actions |
| `GH_PAT` | Open PRs, resolve threads | Classic PAT with `repo` scope — fine-grained PATs don't support `resolveReviewThread` GraphQL mutation |
| `GITHUB_WEBHOOK_SECRET` | Webhook server HMAC verification | Only needed for Option A |
