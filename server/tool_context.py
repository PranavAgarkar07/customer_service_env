from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class ToolContext:
    """
    Session-local snapshot of databases for one episode.

    Created fresh per episode from the scenario generator.
    Never mutates global state.
    Thread-safe by construction (each WebSocket session owns its own instance).

    Mutation logs:
        refund_log:  order_id → True when issue_refund succeeds
        routing_log: customer_id → language when route_to_regional_team succeeds
        tool_args_log: list of raw arg strings for every tool call (for verifiers)
    """
    users_db: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    orders_db: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    refund_policy: Dict[str, Any] = field(default_factory=dict)
    # Mutation logs populated by tools on success — used by rubrics for verification
    refund_log: Dict[str, bool] = field(default_factory=dict)
    routing_log: Dict[str, str] = field(default_factory=dict)   # NEW
    tool_args_log: List[str] = field(default_factory=list)       # NEW
