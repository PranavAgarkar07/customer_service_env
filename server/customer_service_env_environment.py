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
    from .scenarios import get_scenario, list_scenarios
    from .scenario_generator import GeneratedScenario
    from .tools import AVAILABLE_TOOLS, TOOL_DESCRIPTIONS, call_tool
    from .reward_engine import RewardEngine
except (ImportError, ModuleNotFoundError):
    from models import (
        CustomerServiceAction,
        CustomerServiceObservation,
        CustomerServiceState,
    )
    from server.scenarios import get_scenario, list_scenarios
    from server.scenario_generator import GeneratedScenario
    from server.tools import AVAILABLE_TOOLS, TOOL_DESCRIPTIONS, call_tool
    from server.reward_engine import RewardEngine


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
        self._scenario: Optional[GeneratedScenario] = None
        self._ctx = None
        self._reward_engine: Optional[RewardEngine] = None
        self._conversation: List[Dict[str, str]] = []
        self._total_reward: float = 0.0
        self._yielded_reward: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_id: str = "order_status",
        **kwargs: Any,
    ) -> CustomerServiceObservation:
        """Reset the environment with a generative scenario."""
        self._scenario = get_scenario(scenario_id, seed=seed)
        self._ctx = self._scenario.tool_context
        self._reward_engine = RewardEngine()

        self._conversation = []
        self._total_reward = 0.0
        self._yielded_reward = 0.0

        ep_id = episode_id or str(uuid4())
        self._state = CustomerServiceState(
            episode_id=ep_id,
            step_count=0,
            scenario_id=self._scenario.scenario_type,
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
            scenario_id=self._scenario.scenario_type,
            difficulty=self._scenario.difficulty,
            feedback=f"New ticket from {self._scenario.customer_name}. Difficulty: {self._scenario.difficulty}.",
            steps_taken=0,
            max_steps=self.MAX_STEPS,
        )

    def step(
        self,
        action: CustomerServiceAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CustomerServiceObservation:
        """Execute one step with outcome-based grading."""
        self._state.step_count += 1
        tool_result = None
        feedback_parts: List[str] = []

        # 1. Process Tool Call
        if action.tool_name:
            tool_result = call_tool(action.tool_name, action.tool_args, ctx=self._ctx)
            self._state.tools_called.append(action.tool_name)
            
            self._conversation.append({
                "role": "system",
                "content": f"[Tool: {action.tool_name}] Args: {action.tool_args} -> Result: {tool_result}",
            })
            
            # Simple state updates needed for escalations
            if action.tool_name == "escalate_to_human":
                self._state.escalated = True
            elif action.tool_name == "verify_user" and tool_result.get("success"):
                self._state.user_verified = True
            
            feedback_parts.append(f"Tool '{action.tool_name}' executed.")

        # 2. Process Agent Message
        if action.message:
            self._conversation.append({"role": "agent", "content": action.message})
            feedback_parts.append("Message recorded.")

        # 3. Compute Shaped Step Reward (pass required_tools for better signal)
        required_tools = self._scenario.required_tools if self._scenario else []
        step_reward = self._reward_engine.compute_step_reward(
            action, tool_result or {}, self._state, required_tools=required_tools
        )

        # 4. Check Terminal State 
        done = self._check_done_state_based()

        # 5. Compute Terminal Reward on episode completion
        if done:
            terminal_reward = self._reward_engine.compute_terminal_reward(self._scenario, self._ctx, self._state)
            step_reward += terminal_reward
            self._state.resolved = terminal_reward > 0
            
            feedback_parts.append(f"Episode complete. Outcome {'ACHIEVED' if self._state.resolved else 'FAILED'}.")
            feedback_parts.append(f"Steps: {self._state.step_count}/{self.MAX_STEPS}")

        # Update cumulative reward (clamp to [-0.5, 1.0])
        self._total_reward += step_reward
        self._total_reward = max(self._total_reward, -0.5)
        self._total_reward = min(self._total_reward, 0.999) 
        self._state.partial_score = max(self._total_reward, 0.0)

        # Enforce strict open-bound (0.01, 0.99)
        target_total_yielded = max(0.01, min(0.99, self._total_reward)) if done else self._total_reward
        raw_delta = target_total_yielded - getattr(self, '_yielded_reward', 0.0)
        
        step_reward_to_yield = safe_reward(raw_delta)
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
            feedback=" | ".join(feedback_parts) if feedback_parts else "No action taken.",
            steps_taken=self._state.step_count,
            max_steps=self.MAX_STEPS,
        )

    @property
    def state(self) -> CustomerServiceState:
        return self._state

    def _check_done_state_based(self) -> bool:
        if self._state.step_count >= self.MAX_STEPS:
            return True
        if self._scenario.terminal_state_check(self._ctx, self._state):
            return True
        return False
