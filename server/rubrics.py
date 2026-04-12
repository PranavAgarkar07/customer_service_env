# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Rubric system for the Customer Service Agent Environment.

Mirrors the pattern used by repl_env (OpenEnv RFC 004):
  - Outcome rubrics: verify actual state mutations (e.g. refund_log, routing_log)
  - Process rubrics: reward correct steps, penalise wrong ones
  - Message rubrics: score the quality of the agent's final response

Key design decisions:
  - Rubrics operate on ToolContext (live mutable state), NOT just tool names.
    This means a refund_request scenario only gets full credit if issue_refund
    actually succeeded and ctx.refund_log[order_id] is set — not just if the
    agent *called* issue_refund with any arguments.
  - This matches how the calendar_env verifies SQL state mutations and how
    repl_env checks the final_answer against expected_answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set


# =============================================================================
# Base Rubric Interface (RFC 004)
# =============================================================================

class BaseRubric:
    """Abstract base for all rubrics."""

    def score(
        self,
        ctx: Any,           # ToolContext — live episode state with mutations
        state: Any,         # CustomerServiceState — step counter, tools_called, etc.
        scenario: Any,      # GeneratedScenario — ground truth for this episode
        final_message: str = "",  # Agent's last text message
    ) -> float:
        """Return a score in [0.0, 1.0]."""
        raise NotImplementedError


# =============================================================================
# Outcome Rubrics — verify actual state mutations
# =============================================================================

class RefundMutationRubric(BaseRubric):
    """
    Full credit only if ctx.refund_log contains the correct order_id.

    This is the key difference from a tool-name check: the agent must have
    called issue_refund with the *right* order ID and it must have *succeeded*
    (tools only write to refund_log on success).
    """

    def score(self, ctx: Any, state: Any, scenario: Any, final_message: str = "") -> float:
        expected_order = getattr(scenario, "primary_order_id", None)
        if not expected_order:
            return 0.0
        # Check ctx.refund_log — populated by issue_refund() on success only
        if expected_order in ctx.refund_log:
            return 1.0
        # Partial: refund was issued for *some* order (e.g., fraud picked wrong one)
        if ctx.refund_log:
            return 0.4
        return 0.0


class RoutingMutationRubric(BaseRubric):
    """
    Full credit only if ctx.routing_log shows the customer was routed.

    For multilingual scenarios — the agent must have called route_to_regional_team
    and it must have succeeded (populated routing_log).
    """

    def score(self, ctx: Any, state: Any, scenario: Any, final_message: str = "") -> float:
        routing_log: Dict[str, str] = getattr(ctx, "routing_log", {})
        expected_lang = getattr(scenario, "language", "")

        if not routing_log:
            # Fallback: agent at least checked the order (partial completion)
            if "check_order" in state.tools_called and "verify_user" in state.tools_called:
                return 0.35
            return 0.0

        # Routing happened — check language match
        routed_lang = list(routing_log.values())[0].lower() if routing_log else ""
        if expected_lang and expected_lang.lower() in routed_lang:
            return 1.0
        # Routed to wrong team — still better than not routing
        return 0.6


class EscalationRubric(BaseRubric):
    """
    Full credit if human escalation happened (state.escalated = True).

    Used by non_refundable scenario: agent must reach escalate_to_human
    *after* discovering the item cannot be refunded.
    """

    def score(self, ctx: Any, state: Any, scenario: Any, final_message: str = "") -> float:
        if getattr(state, "escalated", False):
            return 1.0
        # Partial: they tried to refund (correct intent) even if not escalated yet
        if ctx.refund_log is not None and len(ctx.refund_log) > 0:
            return 0.3
        return 0.0


class OrderStatusRubric(BaseRubric):
    """
    Full credit if the agent checked the correct order.

    Verifies the scenario's primary order was looked up in the orders_db
    (check_order writes nothing, so we check the tool call log against the
    expected order ID extracted from the customer query).
    """

    def score(self, ctx: Any, state: Any, scenario: Any, final_message: str = "") -> float:
        # For order_status we can't mutate state, so fall back to tool-call check
        # but we also scan the final message for the correct tracking number
        expected_order = getattr(scenario, "primary_order_id", None)
        order_checked = expected_order and any(
            expected_order in call for call in getattr(state, "tool_args_log", [])
        )

        # Check if the tracking number appears in the final message
        order_data = ctx.orders_db.get(expected_order, {}) if expected_order else {}
        tracking = order_data.get("tracking", "")
        tracking_mentioned = tracking and tracking.lower() in final_message.lower()

        if tracking_mentioned:
            return 1.0
        if order_checked or "check_order" in state.tools_called:
            return 0.7
        return 0.0


# =============================================================================
# Process Rubrics — evaluate step-by-step behaviour
# =============================================================================

class SequenceRubric(BaseRubric):
    """
    Score how closely the agent followed the required tool sequence.

    Unlike a binary outcome check, this gives partial credit for partial
    sequences — useful as a dense training signal for RL.
    """

    def score(self, ctx: Any, state: Any, scenario: Any, final_message: str = "") -> float:
        required: List[str] = getattr(scenario, "required_tools", [])
        if not required:
            return 1.0

        called: List[str] = state.tools_called
        called_set: Set[str] = set(called)
        required_set: Set[str] = set(required)

        # Coverage: what fraction of required tools were called?
        coverage = len(called_set & required_set) / len(required_set)

        # Order bonus: were they called in the right order?
        order_bonus = 0.0
        last_idx = -1
        in_order = 0
        for tool in required:
            try:
                idx = called.index(tool)
                if idx > last_idx:
                    in_order += 1
                    last_idx = idx
            except ValueError:
                pass
        if len(required) > 0:
            order_bonus = (in_order / len(required)) * 0.2

        # Redundancy penalty
        extra_calls = max(0, len(called) - len(required))
        redundancy_penalty = min(0.15, extra_calls * 0.03)

        return min(1.0, coverage * 0.8 + order_bonus - redundancy_penalty)


# =============================================================================
# Message Quality Rubric
# =============================================================================

class MessageQualityRubric(BaseRubric):
    """
    Score the agent's final message for required resolution keywords.

    Instead of checking if the agent called the right tools, this verifies
    that the agent *communicated* the outcome to the customer correctly.
    E.g., a refund resolution message should include the refund amount and
    confirmation ID.
    """

    def score(self, ctx: Any, state: Any, scenario: Any, final_message: str = "") -> float:
        if not final_message:
            return 0.0

        scenario_type = getattr(scenario, "scenario_type", "")
        msg_lower = final_message.lower()
        keywords: List[str] = []
        score = 0.0

        if scenario_type == "refund_request":
            keywords = ["refund", "processed", "confirmation"]
            expected_order = getattr(scenario, "primary_order_id", "")
            if expected_order and expected_order.lower() in msg_lower:
                score += 0.3  # Mentioned the right order

        elif scenario_type == "order_status":
            keywords = ["tracking", "order", "status"]
            order_data = ctx.orders_db.get(getattr(scenario, "primary_order_id", ""), {})
            tracking = order_data.get("tracking", "")
            if tracking and tracking.lower() in msg_lower:
                score += 0.4  # Actually surfaced the tracking number

        elif scenario_type == "order_cancel":
            keywords = ["cancel", "refund", "confirmed"]

        elif scenario_type == "fraud_duplicate":
            keywords = ["duplicate", "refund", "investigation"]

        elif scenario_type == "non_refundable":
            keywords = ["non-refundable", "escalat", "human", "policy"]

        elif scenario_type == "multilingual":
            keywords = ["team", "regional", "transfer", "route"]

        if keywords:
            hits = sum(1 for kw in keywords if kw in msg_lower)
            score += (hits / len(keywords)) * 0.6

        return min(1.0, score)


# =============================================================================
# Composite Rubric — combines all rubrics for a given scenario
# =============================================================================

@dataclass
class CustomerServiceRubric:
    """
    Composite rubric that selects the right outcome + process + message rubrics
    for each scenario type at runtime.

    Usage:
        rubric = CustomerServiceRubric.for_scenario(scenario)
        final_score = rubric.score(ctx, state, scenario, final_message)
    """

    outcome_rubric: BaseRubric
    process_rubric: BaseRubric
    message_rubric: BaseRubric
    # Weights must sum to 1.0
    outcome_weight: float = 0.60
    process_weight: float = 0.25
    message_weight: float = 0.15

    @classmethod
    def for_scenario(cls, scenario: Any) -> "CustomerServiceRubric":
        """Factory: return the right rubric mix for a given scenario type."""
        stype = getattr(scenario, "scenario_type", "")

        outcome_map = {
            "refund_request":  RefundMutationRubric(),
            "fraud_duplicate": RefundMutationRubric(),
            "non_refundable":  EscalationRubric(),
            "multilingual":    RoutingMutationRubric(),
            "order_status":    OrderStatusRubric(),
            "order_cancel":    RefundMutationRubric(),
        }
        return cls(
            outcome_rubric=outcome_map.get(stype, OrderStatusRubric()),
            process_rubric=SequenceRubric(),
            message_rubric=MessageQualityRubric(),
        )

    def score(
        self,
        ctx: Any,
        state: Any,
        scenario: Any,
        final_message: str = "",
    ) -> float:
        """Compute the weighted composite score in [0.0, 1.0]."""
        o = self.outcome_rubric.score(ctx, state, scenario, final_message)
        p = self.process_rubric.score(ctx, state, scenario, final_message)
        m = self.message_rubric.score(ctx, state, scenario, final_message)
        return (
            o * self.outcome_weight
            + p * self.process_weight
            + m * self.message_weight
        )
