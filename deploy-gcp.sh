#!/bin/bash
# Deploy AI Code Reviewer to Google Cloud Run
#
# Prerequisites:
#   1. gcloud CLI installed: https://cloud.google.com/sdk/docs/install
#   2. Logged in: gcloud auth login
#   3. Project set: gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   ./deploy-gcp.sh                    # Interactive setup
#   ./deploy-gcp.sh --quick            # Skip prompts, use defaults/env vars

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration (override with env vars)
SERVICE_NAME="${SERVICE_NAME:-ai-code-reviewer}"
REGION="${REGION:-us-central1}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-5}"
MEMORY="${MEMORY:-1Gi}"
TIMEOUT="${TIMEOUT:-300}"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🤖 AI Code Reviewer - GCP Cloud Run Deployment${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Check gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get current project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: No GCP project set${NC}"
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo -e "${GREEN}✓${NC} Project: ${PROJECT_ID}"
echo -e "${GREEN}✓${NC} Service: ${SERVICE_NAME}"
echo -e "${GREEN}✓${NC} Region:  ${REGION}"
echo

# Check for quick mode
QUICK_MODE=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK_MODE=true
fi

# ============== Step 1: Enable APIs ==============
echo -e "${YELLOW}Step 1: Enabling required APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com \
    --quiet

echo -e "${GREEN}✓${NC} APIs enabled"
echo

# ============== Step 2: Set up Secrets ==============
echo -e "${YELLOW}Step 2: Setting up secrets in Secret Manager...${NC}"

create_or_update_secret() {
    local secret_name=$1
    local prompt_text=$2
    local env_var_name=$3
    
    # Check if secret exists
    if gcloud secrets describe "$secret_name" &>/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Secret '$secret_name' exists"
        return 0
    fi
    
    # Get value from env var or prompt
    local secret_value="${!env_var_name:-}"
    
    if [ -z "$secret_value" ] && [ "$QUICK_MODE" = false ]; then
        echo -e "  ${BLUE}→${NC} $prompt_text"
        read -s -p "    Enter value (hidden): " secret_value
        echo
    fi
    
    if [ -z "$secret_value" ]; then
        echo -e "  ${YELLOW}⚠${NC} Skipping '$secret_name' (no value provided)"
        echo "     Set it later: echo 'YOUR_VALUE' | gcloud secrets create $secret_name --data-file=-"
        return 1
    fi
    
    # Create secret
    echo "$secret_value" | gcloud secrets create "$secret_name" --data-file=- --quiet
    echo -e "  ${GREEN}✓${NC} Created secret '$secret_name'"
}

create_or_update_secret "CURSOR_API_KEY" "Cursor API Key (required)" "CURSOR_API_KEY"
create_or_update_secret "GITHUB_TOKEN" "GitHub Token (required)" "GITHUB_TOKEN"
create_or_update_secret "GITHUB_WEBHOOK_SECRET" "GitHub Webhook Secret (recommended)" "GITHUB_WEBHOOK_SECRET"

echo

# ============== Step 3: Build & Push Image ==============
echo -e "${YELLOW}Step 3: Building and pushing Docker image...${NC}"
echo "  This may take a few minutes on first build..."

IMAGE_URL="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

gcloud builds submit \
    --tag "${IMAGE_URL}:latest" \
    --quiet \
    .

echo -e "${GREEN}✓${NC} Image built: ${IMAGE_URL}:latest"
echo

# ============== Step 4: Deploy to Cloud Run ==============
echo -e "${YELLOW}Step 4: Deploying to Cloud Run...${NC}"

# Build secrets flag (only include secrets that exist)
SECRETS_FLAG=""
for secret in CURSOR_API_KEY GITHUB_TOKEN GITHUB_WEBHOOK_SECRET; do
    if gcloud secrets describe "$secret" &>/dev/null 2>&1; then
        if [ -n "$SECRETS_FLAG" ]; then
            SECRETS_FLAG="${SECRETS_FLAG},"
        fi
        SECRETS_FLAG="${SECRETS_FLAG}${secret}=${secret}:latest"
    fi
done

gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_URL}:latest" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --memory "${MEMORY}" \
    --timeout "${TIMEOUT}s" \
    --min-instances "${MIN_INSTANCES}" \
    --max-instances "${MAX_INSTANCES}" \
    --set-secrets "${SECRETS_FLAG}" \
    --quiet

echo -e "${GREEN}✓${NC} Deployed to Cloud Run"
echo

# ============== Step 5: Get Service URL ==============
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --format 'value(status.url)')

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo -e "Service URL: ${GREEN}${SERVICE_URL}${NC}"
echo -e "Webhook URL: ${GREEN}${SERVICE_URL}/webhook${NC}"
echo -e "Health URL:  ${SERVICE_URL}/health"
echo
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Configure GitHub webhook:"
echo "   • Go to: https://github.com/YOUR_ORG/YOUR_REPO/settings/hooks/new"
echo "   • Payload URL: ${SERVICE_URL}/webhook"
echo "   • Content type: application/json"
echo "   • Secret: (use the GITHUB_WEBHOOK_SECRET you configured)"
echo "   • Events: Select 'Pull requests'"
echo
echo "2. Test the deployment:"
echo "   curl ${SERVICE_URL}/health"
echo
echo "3. View logs:"
echo "   gcloud run logs read ${SERVICE_NAME} --region ${REGION}"
