from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ToolContext:
    """
    Session-local snapshot of databases for one episode.

    Created fresh per episode from the scenario generator.
    Never mutates global state.
    Thread-safe by construction (each WebSocket session owns its own instance).
    """
    users_db: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    orders_db: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    refund_policy: Dict[str, Any] = field(default_factory=dict)
    refund_log: Dict[str, bool] = field(default_factory=dict)
