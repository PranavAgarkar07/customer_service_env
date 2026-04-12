# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Customer Service Agent Environment.

Defines Action, Observation, and State types for a multi-step
customer service simulation where an RL agent resolves queries
using tool APIs (verify_user, check_order, issue_refund, route_to_regional_team, etc.).
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator
import math


class CustomerServiceAction(Action):
    """Action the agent takes each step.

    The agent can:
    - Call a tool by setting `tool_name` + `tool_args`
    - Send a message to the customer by setting `message`
    - Do both in one step
    """

    tool_name: Optional[str] = Field(
        default=None,
        description="Name of the tool to call (e.g., 'verify_user', 'check_order')",
    )
    tool_args: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Arguments for the tool call",
    )
    message: str = Field(
        default="",
        description="Message to send to the customer",
    )

    @field_validator("tool_args", mode="before")
    @classmethod
    def validate_tool_args(cls, v: Any) -> Dict[str, Any]:
        """Convert string JSON or partial JSON to a dictionary."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            v_stripped = v.strip()
            if not v_stripped:
                return {}
            # Try parsing as raw JSON
            try:
                import json
                return json.loads(v_stripped)
            except json.JSONDecodeError:
                # If it looks like key:value but missing {}, try adding them
                if ":" in v_stripped and not v_stripped.startswith("{"):
                    try:
                        return json.loads(f"{{{v_stripped}}}")
                    except json.JSONDecodeError:
                        pass
                raise ValueError(f"Invalid JSON for tool_args: {v}. Expected a dictionary or JSON string like {{'key': 'value'}}")
        return v


class CustomerServiceObservation(Observation):
    """Observation the agent receives after each step.

    Contains the customer query, conversation history, tool results,
    and metadata about the current scenario.
    """

    # done: bool and reward: Optional[float] inherited from Observation
    
    @field_validator("reward", mode="before")
    @classmethod
    def clamp_reward(cls, v: Any) -> float:
        """Ultimate defense-in-depth against OOB task scores.
        
        The OpenEnv platform will fail deep validation if any task score
        is precisely 0.0 or 1.0. This clamping ensures values are mapped
        to strictly non-zero and non-one ranges.
        """
        if v is None:
            return 0.05
        try:
            v_float = float(v)
            if math.isnan(v_float) or math.isinf(v_float):
                return 0.05
            if v_float <= 0.0:
                return 0.01
            if v_float >= 1.0:
                return 0.99
            return v_float
        except (ValueError, TypeError):
            return 0.05

    customer_query: str = Field(default="", description="The customer's current query or reply")
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Full conversation history as list of {role, content} dicts",
    )
    tool_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Result from the last tool call, if any",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="Names of tools the agent can call",
    )
    scenario_id: str = Field(default="", description="ID of the current scenario")
    difficulty: str = Field(default="", description="Difficulty level: easy, medium, hard")
    feedback: str = Field(default="", description="Environment feedback message")
    steps_taken: int = Field(default=0, description="Steps taken so far")
    max_steps: int = Field(default=15, description="Maximum steps allowed")


class CustomerServiceState(State):
    """Internal state tracking for the customer service environment."""

    # episode_id and step_count inherited from State

    scenario_id: str = ""
    difficulty: str = ""
    resolved: bool = False
    escalated: bool = False
    routed: bool = False       # True after route_to_regional_team is called
    user_verified: bool = False
    tools_called: List[str] = Field(default_factory=list)
    partial_score: float = Field(default=0.05)
