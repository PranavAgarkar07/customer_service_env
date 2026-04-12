# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Customer Service Agent Environment Client."""

from typing import Any, Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import CustomerServiceAction, CustomerServiceObservation, CustomerServiceState


class CustomerServiceEnv(
    EnvClient[CustomerServiceAction, CustomerServiceObservation, CustomerServiceState]
):
    """
    Client for the Customer Service Agent Environment.

    Example:
        >>> with CustomerServiceEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(scenario_id="easy_order_status")
        ...     print(result.observation.customer_query)
        ...
        ...     result = env.step(CustomerServiceAction(
        ...         tool_name="check_order",
        ...         tool_args={"order_id": "ORD-5002"},
        ...     ))
        ...     print(result.observation.tool_result)
    """

    def _step_payload(self, action: CustomerServiceAction) -> Dict[str, Any]:
        """Convert action to JSON payload."""
        payload: Dict[str, Any] = {}
        if action.tool_name:
            payload["tool_name"] = action.tool_name
            payload["tool_args"] = action.tool_args
        if action.message:
            payload["message"] = action.message
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CustomerServiceObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})

        raw_reward = payload.get("reward")
        if raw_reward is None or not isinstance(raw_reward, (int, float)):
            safe_reward = 0.05
        else:
            safe_reward = max(0.01, min(0.99, float(raw_reward)))

        observation = CustomerServiceObservation(
            done=payload.get("done", False),
            reward=safe_reward,
            customer_query=obs_data.get("customer_query", ""),
            conversation_history=obs_data.get("conversation_history", []),
            tool_result=obs_data.get("tool_result"),
            available_tools=obs_data.get("available_tools", []),
            scenario_id=obs_data.get("scenario_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            feedback=obs_data.get("feedback", ""),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 15),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=safe_reward,
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CustomerServiceState:
        """Parse server response into State object."""
        return CustomerServiceState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            scenario_id=payload.get("scenario_id", ""),
            difficulty=payload.get("difficulty", ""),
            resolved=payload.get("resolved", False),
            escalated=payload.get("escalated", False),
            user_verified=payload.get("user_verified", False),
            tools_called=payload.get("tools_called", []),
            partial_score=max(0.05, min(0.99, float(payload.get("partial_score", 0.05)))),
        )
