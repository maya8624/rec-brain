# Azure Deployment Guide — rec-brain

## What We've Done

### 1. Fixed Docker Build Structure
- Moved `requirements.txt` install out of `Dockerfile.base` into `Dockerfile`
- `Dockerfile.base` now only installs heavy ML libs (`requirements.base.txt`)
- `Dockerfile` now installs lighter packages (`requirements.txt`) + app code
- Result: adding/changing Python packages only requires `docker compose up --build`, not a base rebuild

### 2. Created Azure Resources

| Resource | Name |
|---|---|
| Resource Group | `rg-nexus-dev` |
| Container Registry (ACR) | `recnexusdevacr` (`recnexusdevacr.azurecr.io`) |
| Container Apps Environment | `nexus-dev-env` |
| Container App | `rec-brain` |
| Service Principal | `rec-brain-github` |

### 3. Configured GitHub Secrets
Added the following secrets to the GitHub repo under the `dev` environment:

| Secret | Purpose |
|---|---|
| `AZURE_CREDENTIALS` | Service principal JSON for Azure login |
| `ACR_LOGIN_SERVER` | `recnexusdevacr.azurecr.io` |
| `ACR_USERNAME` | ACR admin username |
| `ACR_PASSWORD` | ACR admin password |
| `AZURE_RESOURCE_GROUP` | `rg-nexus-dev` |
| `CONTAINER_APP_NAME` | `rec-brain` |

### 4. Created GitHub Actions Workflow
File: `.github/workflows/deploy.yml`

- **`build-base` job** — triggers only when `Dockerfile.base` or `requirements.base.txt` changes, pushes `rec-brain-base` to ACR
- **`deploy` job** — triggers on every push to `main`, builds app image tagged with commit SHA, deploys to Container Apps

---

### Step 1 — Push base image to ACR (one-time manual step) ✓ Done
```bash
az acr login --name recnexusdevacr
docker tag rec-brain-base recnexusdevacr.azurecr.io/rec-brain-base:latest
docker push recnexusdevacr.azurecr.io/rec-brain-base:latest
```

## Completed Steps

### Step 2 — Set environment variables on the Container App ✓ Done
Set via `az containerapp update` using OpenAI as the LLM provider:
- `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, `OPENAI_MODEL_NAME=gpt-4o-mini`
- `POSTGRES_URL` → Azure PostgreSQL Flexible Server (`nexus-db-dev01.postgres.database.azure.com`) with `?sslmode=require`
- `BACKEND_BASE_URL` → `https://nexus-api-byegb2gjfzh3hqbq.australiaeast-01.azurewebsites.net`
- `ENVIRONMENT=development`, `LOG_LEVEL=INFO`, `LLM_TEMPERATURE=0.0`, `LLM_MAX_TOKENS=4096`

### Step 3 — Grant Container App access to ACR ✓ Done
```bash
az containerapp registry set \
  --name rec-brain \
  --resource-group rg-nexus-dev \
  --server recnexusdevacr.azurecr.io \
  --username recnexusdevacr \
  --password <acr-password>
```
ACR password stored as secret `recnexusdevacrazurecrio-recnexusdevacr` on the Container App.

### Step 4 — Trigger first deployment ✓ Done
Pushed to `main` — GitHub Actions built the app image, pushed to ACR, and deployed to the Container App.

### Step 5 — Verify ✓ Done
App is live at:
```
https://rec-brain.victoriousisland-e32ac3b0.australiaeast.azurecontainerapps.io
```

---

## Architecture Summary

```
Local machine
  └── docker build → rec-brain-base (heavy ML deps)
                   → rec-brain (app)

GitHub Actions (on push to main)
  └── build app image → push to ACR → deploy to Container Apps

Azure
  ├── ACR (recnexusdevacr)     — image storage
  ├── Container Apps Env       — networking/infrastructure
  └── Container App (rec-brain) — running service
```

## Future — Production Environment
- Create a new GitHub Environment `prd` with separate Azure credentials
- Point to a separate resource group, ACR, and Container App
- Add approval gates on the `prd` environment in GitHub settings
