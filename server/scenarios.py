# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scenario factory for the Customer Service Agent environment.

All scenarios are procedurally generated via ScenarioGenerator so that
every episode has unique customer IDs, order IDs, product names, prices,
and query phrasing. The agent must generalize — not memorize.

Supported scenario types:
  order_status    — easy:   agent checks order tracking
  order_cancel    — easy:   agent checks then cancels a processing order
  refund_request  — medium: agent verifies user, checks order, issues refund
  fraud_duplicate — hard:   agent investigates and refunds a duplicate charge
  non_refundable  — hard:   agent attempts refund, discovers it fails, escalates
  multilingual    — hard:   agent routes non-English customer to regional team
"""

from typing import Any, Dict, List, Optional

try:
    from .scenario_generator import ScenarioGenerator, GeneratedScenario
except ImportError:
    from server.scenario_generator import ScenarioGenerator, GeneratedScenario


# --- Legacy name → canonical type mapping ---
_LEGACY_MAP: Dict[str, str] = {
    "easy_order_status": "order_status",
    "easy_order_cancel": "order_cancel",
    "medium_refund_request": "refund_request",
    "hard_fraud_detection": "fraud_duplicate",
    "hard_non_refundable": "non_refundable",
    "hard_multilingual": "multilingual",
}

# Canonical types
_VALID_TYPES = set(_LEGACY_MAP.values())


def get_scenario(scenario_id: str, seed: Optional[int] = None) -> GeneratedScenario:
    """Get a procedurally generated scenario by ID or legacy alias.

    Each call with a different seed produces a distinct episode even for the
    same scenario type — guaranteeing agent generalization over memorization.
    """
    scenario_type = _LEGACY_MAP.get(scenario_id, scenario_id)
    if scenario_type not in _VALID_TYPES:
        raise ValueError(
            f"Unknown scenario '{scenario_id}'. "
            f"Available: {sorted(_LEGACY_MAP.keys())}"
        )
    generator = ScenarioGenerator(seed)
    return generator.generate(scenario_type=scenario_type)


def list_scenarios() -> List[Dict[str, Any]]:
    """List all available scenario IDs with metadata."""
    return [
        # Canonical IDs (preferred)
        {"scenario_id": "order_status",    "difficulty": "easy",   "description": "Order status and tracking check"},
        {"scenario_id": "order_cancel",    "difficulty": "easy",   "description": "Cancel a processing order"},
        {"scenario_id": "refund_request",  "difficulty": "medium", "description": "Refund a defective delivered item"},
        {"scenario_id": "fraud_duplicate", "difficulty": "hard",   "description": "Investigate and refund a duplicate charge"},
        {"scenario_id": "non_refundable",  "difficulty": "hard",   "description": "Handle non-refundable item; discover and escalate"},
        {"scenario_id": "multilingual",    "difficulty": "hard",   "description": "Route non-English customer to regional team"},
        # Legacy aliases (retained for backward compatibility)
        {"scenario_id": "easy_order_status",    "difficulty": "easy",   "description": "Legacy alias → order_status"},
        {"scenario_id": "easy_order_cancel",    "difficulty": "easy",   "description": "Legacy alias → order_cancel"},
        {"scenario_id": "medium_refund_request","difficulty": "medium", "description": "Legacy alias → refund_request"},
        {"scenario_id": "hard_fraud_detection", "difficulty": "hard",   "description": "Legacy alias → fraud_duplicate"},
        {"scenario_id": "hard_non_refundable",  "difficulty": "hard",   "description": "Legacy alias → non_refundable"},
        {"scenario_id": "hard_multilingual",    "difficulty": "hard",   "description": "Legacy alias → multilingual"},
    ]
