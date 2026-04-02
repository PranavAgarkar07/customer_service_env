---
title: Customer Service Env
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
tags:
  - openenv
pinned: false
---
# Customer Service Agent Environment

A multi-step reinforcement learning environment for training and evaluating AI agents on realistic customer service interactions. Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) platform for the **Meta PyTorch OpenEnv Hackathon**.

## 🎯 Overview

The agent receives customer complaints (defective products, duplicate charges, cancellation requests) and must resolve them using a set of **6 deterministic tools**. Rewards are partial, sequence-aware, and include penalties for wrong actions — making this a challenging, real-world RL benchmark.

## 🏗️ Architecture

```
inference.py  (LLM Agent — Qwen/Llama)
    │ OpenAI Client
    ▼
client.py  (WebSocket EnvClient)
    │
    ▼
server/app.py  (FastAPI — REST + WebSocket)
    ├── customer_service_env_environment.py  (Reward Engine)
    ├── tools.py   (6 tools + Mock DBs: 5 users, 9 orders)
    └── scenarios.py  (6 scenarios across 3 difficulties)
```

## 🔧 Tools

| Tool | Description | Parameters |
|:-----|:------------|:-----------|
| `verify_user` | Verify a customer's identity and account status | `user_id` |
| `check_order` | Look up order details (product, price, tracking, eligibility) | `order_id` |
| `issue_refund` | Process a refund for eligible orders | `order_id`, `reason` |
| `check_policy` | Look up company policy (refund, shipping, returns, warranty, cancellation) | `topic` |
| `escalate_to_human` | Escalate complex cases to a human agent | `reason` |
| `route_to_regional_team` | Route the case to a specialized regional team when the customer speaks a foreign language | `language`, `reason` |

## 📋 Scenarios (6 Tasks)

### Easy

| ID | Description | Tools Required | Max Score |
|:---|:------------|:---------------|:----------|
| `easy_order_status` | Customer asks for tracking info | `check_order` → message | 1.0 |
| `easy_order_cancel` | Customer cancels unshipped order | `check_order` → `check_policy` → message | 1.0 |

### Medium

| ID | Description | Tools Required | Max Score |
|:---|:------------|:---------------|:----------|
| `medium_refund_request` | Defective product refund | `verify_user` → `check_order` → `issue_refund` → message | 1.0 |

### Hard

| ID | Description | Tools Required | Max Score |
|:---|:------------|:---------------|:----------|
| `hard_fraud_detection` | Duplicate charges investigation | `verify_user` → `check_order` ×2 → `check_policy` → `issue_refund` → message | 1.0 |
| `hard_non_refundable` | Digital download refund (denied) + escalation | `verify_user` → `check_order` → `check_policy` → `issue_refund` (fails) → `escalate_to_human` | 1.0 |
| `hard_multilingual` | Customer complains in Spanish | `verify_user` → `check_order` → `route_to_regional_team` → message | 1.0 |

## ⚖️ Reward System

### Positive Rewards
- **Tool rewards** (0.1–0.4): Earned for each correct tool call in the expected sequence
- **Sequence bonuses** (0.1–0.2): Earned for following the correct order (e.g., verify before refund)
- **Message rewards** (0.15–0.4): Earned when the agent's response contains resolution keywords
- **Special rewards** (0.05–0.15): Earned for recognizing edge cases (e.g., non-refundable items)

### Negative Rewards (Penalties)
- **Wrong tool** (-0.05): Calling a tool not in the scenario's correct sequence
- **Duplicate call** (-0.03): Calling the exact same tool with the same arguments twice
- **No action** (-0.05): Taking a step without calling a tool or sending a message

## 🧪 Benchmarking Validation Suite

Judges/Evaluators can mathematically verify the RL reward function determinism using the isolated test suite:
```bash
uv run python test_benchmark.py
```
This script acts as a static "Oracle Agent" demonstrating the perfect sequence of fractional rewards leading up to `1.00` across all 6 scenarios.

## 🚀 Quick Start

### Install
```bash
uv install
```

### Run Tests
```bash
uv run python /tmp/test_env.py
```

### Start Server
```bash
uv run server
```

### Run Inference
```bash
export HF_TOKEN=your_token_here
uv run python inference.py
```

### Docker
```bash
docker build -t customer_service_env:latest .
docker run -p 8000:8000 customer_service_env:latest
```

## 📊 Benchmark Results

| Scenario | Score | Steps |
|:---------|:------|:------|
| `easy_order_status` | 0.80 | 2 |
| `easy_order_cancel` | 0.80 | 3 |
| `medium_refund_request` | 1.00 | 4 |
| `hard_fraud_detection` | 1.00 | 6 |
| `hard_non_refundable` | 0.85 | 5 |
| `hard_multilingual` | 0.95 | 4 |
| **Average** | **0.90** | |

## 📁 File Structure

```
customer_service_env/
├── __init__.py                # Package exports
├── models.py                  # Pydantic Action/Observation/State types
├── client.py                  # WebSocket EnvClient
├── inference.py               # LLM agent with structured logging
├── openenv.yaml               # OpenEnv manifest (6 tasks)
├── Dockerfile                 # Production Docker build
├── pyproject.toml             # Dependencies
├── .env                       # API keys (gitignored)
└── server/
    ├── __init__.py
    ├── app.py                 # FastAPI HTTP + WebSocket server
    ├── customer_service_env_environment.py  # Core reward engine
    ├── tools.py               # 6 tools + mock databases
    └── scenarios.py           # 6 scenarios across 3 difficulty levels
```

## 🔑 Environment Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `HF_TOKEN` | — | Hugging Face API token (required for inference) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use for inference |
| `IMAGE_NAME` | `customer_service_env:latest` | Docker image name |
| `ENABLE_WEB_INTERFACE` | `true` | Enable Gradio web playground |
