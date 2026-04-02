# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scenario definitions for the Customer Service Agent environment.

5 scenarios across 3 difficulty levels with deterministic grading:
  1. Easy:   Order status check
  2. Easy:   Order cancellation
  3. Medium: Refund request
  4. Hard:   Fraud / duplicate order detection
  5. Hard:   Non-refundable item escalation
"""

from typing import Any, Dict, List, Optional


class Scenario:
    """A customer service scenario with grading criteria."""

    def __init__(
        self,
        scenario_id: str,
        difficulty: str,
        customer_name: str,
        customer_id: str,
        order_id: str,
        customer_query: str,
        correct_tool_sequence: List[str],
        resolution_keywords: List[str],
        partial_rewards: Dict[str, float],
        penalty_wrong_tool: float = -0.05,
        penalty_repeated_tool: float = -0.03,
        description: str = "",
        expected_order_ids: Optional[List[str]] = None,
    ):
        self.scenario_id = scenario_id
        self.difficulty = difficulty
        self.customer_name = customer_name
        self.customer_id = customer_id
        self.order_id = order_id
        self.customer_query = customer_query
        self.correct_tool_sequence = correct_tool_sequence
        self.resolution_keywords = resolution_keywords
        self.partial_rewards = partial_rewards
        self.penalty_wrong_tool = penalty_wrong_tool
        self.penalty_repeated_tool = penalty_repeated_tool
        self.description = description
        self.expected_order_ids = expected_order_ids or [order_id]


# =============================================================================
# Scenario 1 — Easy: Order Status Check
# =============================================================================

EASY_ORDER_STATUS = Scenario(
    scenario_id="easy_order_status",
    difficulty="easy",
    customer_name="Alice Johnson",
    customer_id="USR-1001",
    order_id="ORD-5002",
    customer_query=(
        "Hi, I'm Alice Johnson (ID: USR-1001). "
        "I placed an order ORD-5002 for a USB-C cable. "
        "Can you tell me the current status and tracking number?"
    ),
    correct_tool_sequence=["check_order"],
    resolution_keywords=["shipped", "TRK-112233"],
    partial_rewards={
        "check_order": 0.4,
        "correct_info_in_message": 0.4,
        "polite_response": 0.2,
    },
    description="Customer asks for order status. Agent should look up the order and report tracking info.",
)

# =============================================================================
# Scenario 2 — Easy: Order Cancellation
# =============================================================================

EASY_ORDER_CANCEL = Scenario(
    scenario_id="easy_order_cancel",
    difficulty="easy",
    customer_name="Derek Wilson",
    customer_id="USR-1004",
    order_id="ORD-5006",
    customer_query=(
        "Hi, I'm Derek Wilson (ID: USR-1004). "
        "I placed an order ORD-5006 for a Laptop Sleeve but I changed my mind. "
        "It hasn't shipped yet. Can you cancel it?"
    ),
    correct_tool_sequence=["check_order", "check_policy"],
    resolution_keywords=["cancel", "processing", "ORD-5006"],
    partial_rewards={
        "check_order": 0.3,
        "check_policy": 0.2,
        "correct_info_in_message": 0.3,
        "polite_response": 0.2,
    },
    description=(
        "Customer wants to cancel an unshipped order. Agent should check the order "
        "status (processing), check cancellation policy, and confirm the cancellation."
    ),
)

# =============================================================================
# Scenario 3 — Medium: Refund Request
# =============================================================================

MEDIUM_REFUND = Scenario(
    scenario_id="medium_refund_request",
    difficulty="medium",
    customer_name="Alice Johnson",
    customer_id="USR-1001",
    order_id="ORD-5001",
    customer_query=(
        "Hello, I'm Alice Johnson, user ID USR-1001. "
        "I received my order ORD-5001 (Wireless Headphones) but they're defective. "
        "I'd like a refund please."
    ),
    correct_tool_sequence=["verify_user", "check_order", "issue_refund"],
    resolution_keywords=["refund", "REF-5001", "79.99"],
    partial_rewards={
        "verify_user": 0.2,
        "check_order": 0.2,
        "issue_refund": 0.3,
        "correct_sequence": 0.15,
        "correct_info_in_message": 0.15,
    },
    description=(
        "Customer requests a refund for defective item. Agent must: "
        "1) verify user identity, 2) check order eligibility, 3) issue refund."
    ),
)

# =============================================================================
# Scenario 4 — Hard: Fraud Detection / Duplicate Order
# =============================================================================

HARD_FRAUD = Scenario(
    scenario_id="hard_fraud_detection",
    difficulty="hard",
    customer_name="Bob Smith",
    customer_id="USR-1002",
    order_id="ORD-5003",
    customer_query=(
        "Hi, I'm Bob Smith (user USR-1002). I see two charges on my account "
        "for a Mechanical Keyboard - orders ORD-5003 and ORD-5004 both for $149.99. "
        "I only ordered one! Can you investigate and fix this?"
    ),
    correct_tool_sequence=[
        "verify_user",
        "check_order",  # check ORD-5003
        "check_order",  # check ORD-5004
        "check_policy",
        "issue_refund",  # refund duplicate
    ],
    resolution_keywords=["duplicate", "refund", "ORD-5004"],
    partial_rewards={
        "verify_user": 0.1,
        "check_order_first": 0.1,
        "check_order_second": 0.15,
        "identify_duplicate": 0.2,
        "check_policy": 0.1,
        "issue_refund": 0.2,
        "correct_info_in_message": 0.15,
    },
    expected_order_ids=["ORD-5003", "ORD-5004"],
    description=(
        "Customer reports a duplicate charge. Agent must: "
        "1) verify user, 2) check both orders to confirm duplicate, "
        "3) check policy, 4) refund the duplicate, 5) explain to customer."
    ),
)

# =============================================================================
# Scenario 5 — Hard: Non-Refundable Item + Escalation
# =============================================================================

HARD_NON_REFUNDABLE = Scenario(
    scenario_id="hard_non_refundable",
    difficulty="hard",
    customer_name="Elena Martinez",
    customer_id="USR-1005",
    order_id="ORD-5009",
    customer_query=(
        "Hello, I'm Elena Martinez (USR-1005). I purchased a Digital Art Course "
        "(order ORD-5009) for $199.99 but the content quality is terrible. "
        "I want a full refund immediately!"
    ),
    correct_tool_sequence=[
        "verify_user",
        "check_order",
        "check_policy",   # must check refund policy — digital is non-refundable
        "issue_refund",   # will FAIL — tool returns error for digital items
        "escalate_to_human",  # correct action after refund denial
    ],
    resolution_keywords=["non-refundable", "digital", "escalat"],
    partial_rewards={
        "verify_user": 0.1,
        "check_order": 0.1,
        "check_policy": 0.15,
        "issue_refund_attempt": 0.1,  # reward for trying (even though it fails)
        "recognize_non_refundable": 0.15,
        "escalate_to_human": 0.2,
        "correct_info_in_message": 0.2,
    },
    penalty_wrong_tool=-0.05,
    penalty_repeated_tool=-0.03,
    description=(
        "Customer wants refund for a digital download — which is non-refundable. "
        "Agent must: 1) verify user, 2) check order, 3) check refund policy, "
        "4) attempt refund (learns it fails), 5) escalate to human agent, "
        "6) explain the policy to the customer."
    ),
)


# =============================================================================
# Scenario 6 — Hard: Multilingual Handoff
# =============================================================================

HARD_MULTILINGUAL = Scenario(
    scenario_id="hard_multilingual",
    difficulty="hard",
    customer_name="Carlos Mendez",
    customer_id="USR-1002",
    order_id="ORD-5003",
    customer_query=(
        "Hola, soy Carlos Mendez (USR-1002). "
        "Recibí mi teclado mecánico (orden ORD-5003) pero parece estar defectuoso. "
        "¿Me pueden ayudar con esto o transferirme a alguien que hable español?"
    ),
    correct_tool_sequence=[
        "verify_user",
        "check_order",
        "route_to_regional_team",
    ],
    resolution_keywords=["español", "transfer", "spanish", "regional"],
    partial_rewards={
        "verify_user": 0.2,
        "check_order": 0.2,
        "route_to_regional_team": 0.4,
        "correct_info_in_message": 0.2,
    },
    description=(
        "Customer complains in Spanish regarding a defective product. "
        "Agent must: 1) verify user, 2) check order, "
        "3) correctly route the conversation to the Spanish regional team."
    ),
)


# =============================================================================
# Scenario Registry
# =============================================================================

SCENARIOS: Dict[str, Scenario] = {
    "easy_order_status": EASY_ORDER_STATUS,
    "easy_order_cancel": EASY_ORDER_CANCEL,
    "medium_refund_request": MEDIUM_REFUND,
    "hard_fraud_detection": HARD_FRAUD,
    "hard_non_refundable": HARD_NON_REFUNDABLE,
    "hard_multilingual": HARD_MULTILINGUAL,
}

SCENARIO_BY_DIFFICULTY: Dict[str, List[Scenario]] = {
    "easy": [EASY_ORDER_STATUS, EASY_ORDER_CANCEL],
    "medium": [MEDIUM_REFUND],
    "hard": [HARD_FRAUD, HARD_NON_REFUNDABLE, HARD_MULTILINGUAL],
}


def get_scenario(scenario_id: str) -> Scenario:
    """Get a scenario by ID."""
    if scenario_id not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_id}'. Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[scenario_id]


def list_scenarios() -> List[Dict[str, Any]]:
    """List all available scenarios with metadata."""
    return [
        {
            "scenario_id": s.scenario_id,
            "difficulty": s.difficulty,
            "description": s.description,
            "num_required_tools": len(s.correct_tool_sequence),
        }
        for s in SCENARIOS.values()
    ]
