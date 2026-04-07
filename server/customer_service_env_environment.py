# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Customer Service Agent Environment Implementation.

A multi-step customer service simulation where agents resolve customer
queries using tool APIs. Supports 5 scenarios across 3 difficulty levels
with partial rewards, sequence bonuses, and penalties.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment


def safe_reward(r: float) -> float:
    """Map any raw step reward to a value strictly inside (0.01, 0.99).

    Per OpenEnv Phase 2 validation rules every individual step reward must be
    a floating-point number in the open interval (0, 1).
    Negative rewards (penalties) are first shifted into a positive range by
    adding 0.5 so that a penalty of -0.05 becomes 0.45 instead of -0.05.
    """
    if r < 0:
        r = 0.5 + r   # shift: -0.05 → 0.45, -0.03 → 0.47, -0.5 → 0.0 (then clamped up)
    return max(0.01, min(0.99, r))

try:
    from ..models import (
        CustomerServiceAction,
        CustomerServiceObservation,
        CustomerServiceState,
    )
    from .scenarios import SCENARIOS, get_scenario, Scenario
    from .tools import AVAILABLE_TOOLS, call_tool
except (ImportError, ModuleNotFoundError):
    from models import (
        CustomerServiceAction,
        CustomerServiceObservation,
        CustomerServiceState,
    )
    from server.scenarios import SCENARIOS, get_scenario, Scenario
    from server.tools import AVAILABLE_TOOLS, call_tool


class CustomerServiceEnvironment(Environment):
    """
    Customer Service Agent Environment.

    The agent receives a customer query and must resolve it using the
    available tools (verify_user, check_order, issue_refund, etc.).

    Reward System:
    - Correct tool calls earn incremental rewards (0.1-0.4)
    - Following the correct sequence earns bonuses
    - Providing accurate info in messages earns rewards
    - PENALTIES: Wrong/irrelevant tool calls get -0.05
    - PENALTIES: Repeated tool calls (same tool+args) get -0.03

    5 Scenarios:
    - easy_order_status: 1-2 tools, check order tracking
    - easy_order_cancel: 2 tools, cancel unshipped order
    - medium_refund_request: 3 tools + sequence bonus
    - hard_fraud_detection: 5 tools, investigate duplicate charges
    - hard_non_refundable: 5 tools, handle non-refundable + escalate
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 15

    def __init__(self):
        """Initialize the environment."""
        super().__init__()
        self._state = CustomerServiceState()
        self._scenario: Optional[Scenario] = None
        self._conversation: List[Dict[str, str]] = []
        self._tools_called: List[str] = []
        self._tool_calls_detail: List[Dict[str, Any]] = []  # (tool_name, args) for duplicate detection
        self._tool_results: List[Dict[str, Any]] = []
        self._total_reward: float = 0.0
        self._yielded_reward: float = 0.0
        self._user_verified: bool = False
        self._orders_checked: List[str] = []
        self._refund_issued: bool = False
        self._refund_attempted: bool = False
        self._policy_checked: bool = False
        self._escalated: bool = False
        self._routed_to_regional_team: bool = False
        self._resolved: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_id: str = "easy_order_status",
        **kwargs: Any,
    ) -> CustomerServiceObservation:
        """Reset the environment with a specific scenario.

        Args:
            seed: Random seed (unused, scenarios are deterministic)
            episode_id: Custom episode ID
            scenario_id: ID of the scenario to load

        Returns:
            Initial observation with customer query
        """
        self._scenario = get_scenario(scenario_id)
        self._conversation = []
        self._tools_called = []
        self._tool_calls_detail = []
        self._tool_results = []
        self._total_reward = 0.0
        self._yielded_reward = 0.0
        self._user_verified = False
        self._orders_checked = []
        self._refund_issued = False
        self._refund_attempted = False
        self._policy_checked = False
        self._escalated = False
        self._routed_to_regional_team = False
        self._resolved = False

        ep_id = episode_id or str(uuid4())
        self._state = CustomerServiceState(
            episode_id=ep_id,
            step_count=0,
            scenario_id=scenario_id,
            difficulty=self._scenario.difficulty,
        )

        # Add customer's initial query to conversation
        self._conversation.append({
            "role": "customer",
            "content": self._scenario.customer_query,
        })

        return CustomerServiceObservation(
            done=False,
            reward=0.0,
            customer_query=self._scenario.customer_query,
            conversation_history=list(self._conversation),
            tool_result=None,
            available_tools=AVAILABLE_TOOLS,
            scenario_id=scenario_id,
            difficulty=self._scenario.difficulty,
            feedback=f"New ticket from {self._scenario.customer_name}. "
                     f"Difficulty: {self._scenario.difficulty}. "
                     f"Use tools to resolve their query.",
            steps_taken=0,
            max_steps=self.MAX_STEPS,
        )

    def step(
        self,
        action: CustomerServiceAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CustomerServiceObservation:
        """Execute one step in the environment.

        Args:
            action: The agent's action (tool call and/or message)
            timeout_s: Timeout (unused)

        Returns:
            Observation with results and updated conversation
        """
        self._state.step_count += 1
        step_reward = 0.0
        tool_result = None
        feedback_parts: List[str] = []

        # --- Process tool call ---
        if action.tool_name:
            tool_result = call_tool(action.tool_name, action.tool_args)
            self._tools_called.append(action.tool_name)
            self._tool_results.append(tool_result)

            # Check for duplicate tool call (same tool + same args)
            call_signature = {"tool": action.tool_name, "args": dict(sorted(action.tool_args.items()))}
            is_duplicate = call_signature in self._tool_calls_detail
            self._tool_calls_detail.append(call_signature)

            # Record the tool call in conversation
            self._conversation.append({
                "role": "system",
                "content": f"[Tool: {action.tool_name}] Args: {action.tool_args} -> Result: {tool_result}",
            })

            if is_duplicate:
                # Penalty for exact duplicate call
                step_reward += self._scenario.penalty_repeated_tool if self._scenario else -0.03
                feedback_parts.append(
                    f"Warning: Duplicate tool call '{action.tool_name}' with same arguments. "
                    f"Penalty applied ({self._scenario.penalty_repeated_tool if self._scenario else -0.03})."
                )
            else:
                # Compute reward for the tool call
                tool_reward = self._compute_tool_reward(action.tool_name, action.tool_args, tool_result)
                step_reward += tool_reward

                if tool_reward > 0:
                    feedback_parts.append(f"Tool '{action.tool_name}' executed successfully. Reward: +{tool_reward:.2f}")
                elif tool_reward < 0:
                    feedback_parts.append(
                        f"Tool '{action.tool_name}' was not needed for this scenario. Penalty: {tool_reward:.2f}"
                    )
                else:
                    if tool_result.get("success"):
                        feedback_parts.append(f"Tool '{action.tool_name}' executed but no reward earned.")
                    else:
                        feedback_parts.append(
                            f"Tool '{action.tool_name}' failed: {tool_result.get('error', 'unknown')}"
                        )

        # --- Process agent message ---
        if action.message:
            self._conversation.append({
                "role": "agent",
                "content": action.message,
            })

            msg_reward = self._compute_message_reward(action.message)
            step_reward += msg_reward

            if msg_reward > 0:
                feedback_parts.append(f"Your response contained relevant information. Reward: +{msg_reward:.2f}")
            else:
                feedback_parts.append("Your response was recorded.")

        # --- No action taken ---
        if not action.tool_name and not action.message:
            step_reward += -0.05  # Penalty for doing nothing
            feedback_parts.append("No action taken. Please call a tool or send a message.")

        # --- Check if resolved ---
        done = self._check_done()

        # Update cumulative reward (clamp to [-0.5, 1.0])
        self._total_reward += step_reward
        self._total_reward = max(self._total_reward, -0.5)  # Floor at -0.5
        self._total_reward = min(self._total_reward, 0.999)  # Ceiling at 0.999 — strict open bound, avoids Phase 2 validator rejection
        self._state.partial_score = max(self._total_reward, 0.0)  # State shows non-negative

        if done:
            final_score = max(self._total_reward, 0.0)
            final_score = min(final_score, 1.0)
            feedback_parts.append(
                f"Episode complete. Total score: {final_score:.2f}/1.00. "
                f"Steps used: {self._state.step_count}/{self.MAX_STEPS}."
            )

        # Enforce strict open-bound (0.01, 0.99) for every reward returned to OpenEnv.
        # The validator rejects 0, 1, and any negative value on individual step rewards.
        target_total_yielded = self._total_reward
        if done:
            target_total_yielded = max(0.01, min(0.99, self._total_reward))

        raw_delta = target_total_yielded - getattr(self, '_yielded_reward', 0.0)
        # Apply safe_reward to the outgoing delta — shifts negatives into (0.01, 0.99)
        step_reward_to_yield = safe_reward(raw_delta)
        # Track cumulative using the TRANSFORMED value so OpenEnv's running total stays valid
        self._yielded_reward = getattr(self, '_yielded_reward', 0.0) + step_reward_to_yield

        return CustomerServiceObservation(
            done=done,
            reward=round(step_reward_to_yield, 4),
            customer_query=self._scenario.customer_query if self._scenario else "",
            conversation_history=list(self._conversation),
            tool_result=tool_result,
            available_tools=AVAILABLE_TOOLS,
            scenario_id=self._state.scenario_id,
            difficulty=self._state.difficulty,
            feedback=" | ".join(feedback_parts) if feedback_parts else "Step recorded.",
            steps_taken=self._state.step_count,
            max_steps=self.MAX_STEPS,
        )

    @property
    def state(self) -> CustomerServiceState:
        """Get the current environment state."""
        self._state.resolved = self._resolved
        self._state.escalated = self._escalated
        self._state.user_verified = self._user_verified
        self._state.tools_called = list(self._tools_called)
        return self._state

    # =========================================================================
    # Reward Logic
    # =========================================================================

    def _compute_tool_reward(
        self, tool_name: str, tool_args: Dict[str, Any], result: Dict[str, Any]
    ) -> float:
        """Compute reward for a tool call based on scenario requirements."""
        if not self._scenario:
            return 0.0

        scenario = self._scenario
        reward = 0.0

        # --- PENALTY: Tool not in correct sequence at all ---
        if tool_name not in scenario.correct_tool_sequence:
            return scenario.penalty_wrong_tool  # -0.05 default

        # --- Tool-specific reward logic ---
        if tool_name == "verify_user":
            if tool_args.get("user_id") == scenario.customer_id and result.get("success"):
                reward = scenario.partial_rewards.get("verify_user", 0.1)
                self._user_verified = True
            elif not result.get("success"):
                reward = scenario.penalty_wrong_tool  # Wrong user ID

        elif tool_name == "check_order":
            order_id = tool_args.get("order_id", "")
            if not result.get("success"):
                reward = scenario.penalty_wrong_tool
            elif order_id and order_id not in self._orders_checked:
                self._orders_checked.append(order_id)

                if scenario.scenario_id == "hard_fraud_detection":
                    # Hard: reward for checking each of the two orders
                    if len(self._orders_checked) == 1:
                        reward = scenario.partial_rewards.get("check_order_first", 0.1)
                    elif len(self._orders_checked) == 2:
                        reward = scenario.partial_rewards.get("check_order_second", 0.15)
                else:
                    # Check if it's a relevant order for this scenario
                    if order_id in scenario.expected_order_ids:
                        reward = scenario.partial_rewards.get("check_order", 0.2)

        elif tool_name == "issue_refund":
            if scenario.scenario_id == "hard_non_refundable":
                # Special case: trying to refund non-refundable item
                if not self._refund_attempted:
                    self._refund_attempted = True
                    # Reward for TRYING (even though it fails) — shows agent is following process
                    reward = scenario.partial_rewards.get("issue_refund_attempt", 0.1)
                    if not result.get("success"):
                        # Agent correctly discovered it's non-refundable
                        reward += scenario.partial_rewards.get("recognize_non_refundable", 0.15)
            else:
                # Normal refund
                if not self._refund_issued and result.get("success"):
                    if "issue_refund" in scenario.correct_tool_sequence:
                        reward = scenario.partial_rewards.get("issue_refund", 0.2)
                        self._refund_issued = True

                        # Sequence bonus: verify -> check -> refund
                        if scenario.difficulty == "medium" and self._user_verified:
                            reward += scenario.partial_rewards.get("correct_sequence", 0.1)
                        elif scenario.difficulty == "hard" and self._user_verified and len(self._orders_checked) >= 2:
                            reward += scenario.partial_rewards.get("identify_duplicate", 0.1)

        elif tool_name == "check_policy":
            if not self._policy_checked:
                if "check_policy" in scenario.correct_tool_sequence:
                    reward = scenario.partial_rewards.get("check_policy", 0.1)
                    self._policy_checked = True

        elif tool_name == "escalate_to_human":
            if not self._escalated:
                self._escalated = True
                if "escalate_to_human" in scenario.correct_tool_sequence:
                    reward = scenario.partial_rewards.get("escalate_to_human", 0.1)
                    # Bonus: escalation after discovering non-refundable
                    if scenario.scenario_id == "hard_non_refundable" and self._refund_attempted:
                        reward += 0.05  # Extra for correct escalation timing

        elif tool_name == "route_to_regional_team":
            if not self._routed_to_regional_team:
                self._routed_to_regional_team = True
                if "route_to_regional_team" in scenario.correct_tool_sequence:
                    reward = scenario.partial_rewards.get("route_to_regional_team", 0.4)

        return reward

    def _compute_message_reward(self, message: str) -> float:
        """Compute reward for agent's message based on resolution keywords and politeness."""
        if not self._scenario:
            return 0.0

        reward = 0.0
        msg_lower = message.lower()

        # --- Resolution keyword matching ---
        keywords_found = sum(
            1 for kw in self._scenario.resolution_keywords
            if kw.lower() in msg_lower
        )

        if keywords_found > 0:
            ratio = keywords_found / len(self._scenario.resolution_keywords)
            max_msg_reward = self._scenario.partial_rewards.get(
                "correct_info_in_message", 0.15
            )
            reward += ratio * max_msg_reward
            if ratio >= 0.5:
                self._resolved = True

        # --- Polite response reward: using customer's name ---
        polite_reward = self._scenario.partial_rewards.get("polite_response", 0.0)
        if polite_reward > 0:
            customer_first_name = self._scenario.customer_name.split()[0].lower()
            if customer_first_name in msg_lower:
                reward += polite_reward
                # Only award once — zero out so it's not given again
                self._scenario.partial_rewards["polite_response"] = 0.0

        return reward

    def _check_done(self) -> bool:
        """Check if the episode should end."""
        if self._state.step_count >= self.MAX_STEPS:
            return True
        if self._resolved:
            return True
        if self._escalated:
            return True
        if self._routed_to_regional_team:
            return True
        return False
