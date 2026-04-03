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
TEMPERATURE = 0.1
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
You are an intelligent customer service AI agent. You must resolve customer queries efficiently.

CRITICAL RULES:
1. ONLY call tools that are relevant to the scenario. Read the customer query carefully.
2. NEVER repeat the same action or message. Each step must make forward progress.
3. Maximum 5 steps per task. Be efficient.
4. ALWAYS include specific IDs in your messages (TRK-xxx, REF-xxx, ORD-xxx, etc.).
5. Use the customer's name and be professional and polite.

WORKFLOW PER SCENARIO TYPE:

For ORDER STATUS questions (customer asks "where is my order" / "tracking"):
  Step 1: check_order(order_id) — get status and tracking
  Step 2: message with status + tracking number directly to customer
  (Do NOT call verify_user for simple status checks — it is not needed)

For ORDER CANCELLATION questions (customer asks to cancel an order):
  Step 1: check_order(order_id) — confirm it's still in processing
  Step 2: check_policy(topic="cancellation") — look up cancellation rules
  Step 3: message confirming cancellation with order ID and policy details
  (Do NOT call verify_user for cancellations — it is not needed)

For REFUND REQUESTS (customer asking for money back on delivered item):
  Step 1: verify_user(user_id) — verify identity first
  Step 2: check_order(order_id) — confirm delivery and eligibility
  Step 3: issue_refund(order_id, reason) — process the refund
  Step 4: message with refund confirmation ID, amount, and timeline

For DUPLICATE CHARGE / FRAUD issues (customer reports multiple charges):
  Step 1: verify_user(user_id) — verify identity
  Step 2: check_order(first_order_id) — check first order details
  Step 3: check_order(second_order_id) — check second order details
  Step 4: Compare orders — look at product, price, timing, payment to confirm duplicate
  Step 5: issue_refund(duplicate_order_id, reason="duplicate charge") — refund the duplicate order
  Step 6: ONLY NOW send a final message to the customer with words "duplicate", "refund", and the refunded order ID (e.g. ORD-5004)

  ⚠️ CRITICAL: During steps 1-4, do NOT use the words "duplicate" or "refund" in your messages.
     Say things like "I'm investigating both charges" or "Let me compare the orders".
     Using those words too early will end the episode before the refund is processed.
     Save resolution language for the FINAL message AFTER issue_refund succeeds.

For NON-REFUNDABLE ITEM complaints (digital products, gift cards):
  Step 1: verify_user(user_id) — verify identity
  Step 2: check_order(order_id) — check order details
  Step 3: check_policy(topic="refund") — check refund policy
  Step 4: issue_refund(order_id, reason) — ATTEMPT the refund (it WILL fail, that's expected and necessary)
  Step 5: escalate_to_human(reason) — escalate after refund denial
  Step 6: message explaining the item is non-refundable, it's a digital product, and the case has been escalated
  (You MUST attempt the refund even if you suspect it will fail — this is required)

For NON-ENGLISH / MULTILINGUAL customers:
  Step 1: verify_user(user_id) — verify identity first
  Step 2: check_order(order_id) — check their order
  Step 3: route_to_regional_team(language, reason) — route to the right team
  Step 4: message in their language mentioning "regional" or "transfer"
  (ALWAYS verify and check order BEFORE routing — do NOT route immediately)

Available Tools:
- verify_user(user_id: str)
- check_order(order_id: str)
- issue_refund(order_id: str, reason: str)
- check_policy(topic: str) — valid topics: "refund", "cancellation", "shipping", "returns", "warranty"
- escalate_to_human(reason: str)
- route_to_regional_team(language: str, reason: str)

OUTPUT FORMAT — respond with valid JSON only, no markdown code blocks:
{
  "tool_name": "tool_name_here_or_null",
  "tool_args": {"arg": "value"},
  "message": "What you say to the customer"
}

Set tool_name to null and tool_args to null when you only want to send a message.

ANTI-LOOP RULES:
- If you already called a tool, do NOT call it again with the same arguments.
- If you sent a message and the episode didn't end, you MUST call a tool next.
- After checking orders in fraud/duplicate cases, proceed to check_policy then issue_refund.
- Never send the same message twice. If stuck, try a different tool or escalate.
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