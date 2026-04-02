"""
Inference Script — Customer Service Agent Environment
=====================================================

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from customer_service_env import CustomerServiceAction, CustomerServiceEnv

# =============================================================================
# Configuration
# =============================================================================

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME = os.getenv("IMAGE_NAME", "customer_service_env:latest")

BENCHMARK = "customer_service_env"
MAX_STEPS = 12
TEMPERATURE = 0.3
MAX_TOKENS = 300

SCENARIOS = [
    "easy_order_status",
    "easy_order_cancel",
    "medium_refund_request",
    "hard_fraud_detection",
    "hard_non_refundable",
    "hard_multilingual",
]

SYSTEM_PROMPT = textwrap.dedent("""
You are a helpful and professional customer service agent. You interact with a simulated environment.
Your goal is to resolve customer queries exactly as requested.

To succeed:
1. Always verify the customer's identity with 'verify_user' if requested or if performing sensitive actions like refunds.
2. Always check order details with 'check_order' to confirm status and eligibility.
3. Check company policy with 'check_policy' if there is any ambiguity about what is allowed.
4. When you resolve an issue, always confirm details like Tracking IDs, Refund IDs, or escalation status in your final message to the customer.
5. BE POLITE: Use the customer's name (once verified) and be professional.
6. If the customer speaks a language other than English, route them to the appropriate regional team using 'route_to_regional_team'.

Available Tools:
- verify_user(user_id: str)
- check_order(order_id: str)
- issue_refund(order_id: str, reason: str)
- check_policy(topic: str)
- escalate_to_human(reason: str)
- route_to_regional_team(language: str, reason: str)

Rules for Tool Usage:
- Output your response in raw JSON format with three keys: 'tool_name', 'tool_args', and 'message'.
- If no tool is needed for the current step, set 'tool_name' and 'tool_args' to null.
- Ensure 'tool_args' is a dictionary.
- Your 'message' should be what you say to the customer.

IMPORTANT: Always include relevant IDs (TRK-xxx, REF-xxx, etc.) in your messages to earn full points.
""").strip()


# =============================================================================
# Logging Helpers
# =============================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# =============================================================================
# LLM Interaction
# =============================================================================

def get_agent_action(
    client: OpenAI,
    conversation: List[Dict[str, str]],
    tool_result: Optional[Dict[str, Any]],
    feedback: str,
) -> Dict[str, Any]:
    """Ask the LLM to decide the next action."""

    user_content = f"Environment feedback: {feedback}\n"
    if tool_result:
        user_content += f"Last tool result: {json.dumps(tool_result)}\n"
    user_content += "\nConversation so far:\n"
    for msg in conversation[-6:]:  # Last 6 messages for context window
        user_content += f"  [{msg['role']}]: {msg['content']}\n"
    user_content += "\nDecide your next action (JSON):"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Parse JSON from the response
        # Try to extract JSON if wrapped in markdown code block
        if "```" in raw:
            lines = raw.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            raw = "\n".join(json_lines)

        return json.loads(raw)

    except (json.JSONDecodeError, Exception) as e:
        print(f"[DEBUG] LLM parse error: {e}", flush=True)
        return {"tool_name": None, "tool_args": {}, "message": "I apologize, let me help you."}


# =============================================================================
# Main Inference Loop
# =============================================================================

async def run_scenario(client: OpenAI, env: CustomerServiceEnv, scenario_id: str) -> float:
    """Run a single scenario and return the total score."""
    task_name = scenario_id
    rewards: List[float] = []
    steps_taken = 0
    total_score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(scenario_id=scenario_id)
        obs = result.observation
        tool_result = None

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Get LLM decision
            action_dict = get_agent_action(
                client,
                obs.conversation_history,
                tool_result,
                obs.feedback,
            )

            # Build action
            action = CustomerServiceAction(
                tool_name=action_dict.get("tool_name"),
                tool_args=action_dict.get("tool_args", {}),
                message=action_dict.get("message", ""),
            )

            # Execute step
            result = await env.step(action)
            obs = result.observation
            tool_result = obs.tool_result

            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step

            # Build action string for logging
            action_str = ""
            if action.tool_name:
                action_str = f"{action.tool_name}({json.dumps(action.tool_args)})"
            if action.message:
                short_msg = action.message[:50].replace("\n", " ")
                action_str += f" msg='{short_msg}'"
            action_str = action_str.strip() or "noop"

            log_step(step=step, action=action_str, reward=reward, done=result.done, error=None)

            if result.done:
                break

        total_score = min(max(sum(rewards), 0.0), 1.0)
        success = total_score >= 0.3

    except Exception as e:
        print(f"[DEBUG] Scenario error: {e}", flush=True)
        success = False
        total_score = 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=total_score, rewards=rewards)

    return total_score


async def main() -> None:
    """Run inference across all three scenarios."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to the environment
    env = await CustomerServiceEnv.from_docker_image(IMAGE_NAME)

    scores = {}
    try:
        for scenario_id in SCENARIOS:
            print(f"\n{'='*60}", flush=True)
            print(f"Running scenario: {scenario_id}", flush=True)
            print(f"{'='*60}", flush=True)

            score = await run_scenario(client, env, scenario_id)
            scores[scenario_id] = score

        # Print summary
        print(f"\n{'='*60}", flush=True)
        print("FINAL SUMMARY", flush=True)
        print(f"{'='*60}", flush=True)
        for sid, score in scores.items():
            print(f"  {sid}: {score:.3f}", flush=True)
        avg = sum(scores.values()) / len(scores) if scores else 0.0
        print(f"  Average: {avg:.3f}", flush=True)

    finally:
        try:
            # Ignore non-fatal cleanup errors (e.g. docker stop timeout)
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())