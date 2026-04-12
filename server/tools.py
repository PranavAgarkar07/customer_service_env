# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simulated customer service tools.

Each tool returns deterministic mock data for reproducible grading.
Tools: verify_user, check_order, issue_refund, check_policy, escalate_to_human
"""

from typing import Any, Dict, Optional

try:
    from .tool_context import ToolContext
except ImportError:
    from server.tool_context import ToolContext

# =============================================================================
# Mock Databases
# =============================================================================

USERS_DB: Dict[str, Dict[str, Any]] = {
    "USR-1001": {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "verified": True,
        "account_status": "active",
        "membership": "premium",
        "join_date": "2024-01-15",
    },
    "USR-1002": {
        "name": "Bob Smith",
        "email": "bob@example.com",
        "verified": True,
        "account_status": "active",
        "membership": "basic",
        "join_date": "2025-06-20",
    },
    "USR-1003": {
        "name": "Carol Davis",
        "email": "carol@example.com",
        "verified": True,
        "account_status": "suspended",
        "membership": "premium",
        "join_date": "2023-11-01",
    },
    "USR-1004": {
        "name": "Derek Wilson",
        "email": "derek@example.com",
        "verified": True,
        "account_status": "active",
        "membership": "basic",
        "join_date": "2025-03-10",
    },
    "USR-1005": {
        "name": "Elena Martinez",
        "email": "elena@example.com",
        "verified": True,
        "account_status": "active",
        "membership": "premium",
        "join_date": "2024-08-22",
    },
}

ORDERS_DB: Dict[str, Dict[str, Any]] = {
    "ORD-5001": {
        "user_id": "USR-1001",
        "product": "Wireless Headphones",
        "price": 79.99,
        "status": "delivered",
        "tracking": "TRK-998877",
        "delivery_date": "2026-03-28",
        "return_eligible": True,
        "category": "electronics",
    },
    "ORD-5002": {
        "user_id": "USR-1001",
        "product": "USB-C Cable",
        "price": 12.99,
        "status": "shipped",
        "tracking": "TRK-112233",
        "delivery_date": None,
        "return_eligible": False,
        "category": "accessories",
    },
    "ORD-5003": {
        "user_id": "USR-1002",
        "product": "Mechanical Keyboard",
        "price": 149.99,
        "status": "delivered",
        "tracking": "TRK-445566",
        "delivery_date": "2026-03-25",
        "return_eligible": True,
        "category": "electronics",
    },
    "ORD-5004": {
        "user_id": "USR-1002",
        "product": "Mechanical Keyboard",
        "price": 149.99,
        "status": "delivered",
        "tracking": "TRK-445567",
        "delivery_date": "2026-03-25",
        "return_eligible": True,
        "category": "electronics",
    },
    "ORD-5005": {
        "user_id": "USR-1003",
        "product": "Monitor Stand",
        "price": 45.00,
        "status": "cancelled",
        "tracking": None,
        "delivery_date": None,
        "return_eligible": False,
        "category": "furniture",
    },
    "ORD-5006": {
        "user_id": "USR-1004",
        "product": "Laptop Sleeve",
        "price": 29.99,
        "status": "processing",
        "tracking": None,
        "delivery_date": None,
        "return_eligible": False,
        "category": "accessories",
    },
    "ORD-5007": {
        "user_id": "USR-1004",
        "product": "Webcam HD Pro",
        "price": 89.99,
        "status": "delivered",
        "tracking": "TRK-778899",
        "delivery_date": "2026-03-20",
        "return_eligible": True,
        "category": "electronics",
    },
    "ORD-5008": {
        "user_id": "USR-1005",
        "product": "Ergonomic Mouse",
        "price": 59.99,
        "status": "shipped",
        "tracking": "TRK-334455",
        "delivery_date": None,
        "return_eligible": False,
        "category": "electronics",
    },
    "ORD-5009": {
        "user_id": "USR-1005",
        "product": "Digital Art Course",
        "price": 199.99,
        "status": "delivered",
        "tracking": None,
        "delivery_date": "2026-03-22",
        "return_eligible": False,
        "category": "digital_download",
    },
}

REFUND_POLICY = {
    "max_days_after_delivery": 30,
    "requires_user_verification": True,
    "requires_order_check": True,
    "eligible_statuses": ["delivered"],
    "non_refundable_categories": ["digital_download", "gift_cards"],
}


# =============================================================================
# Tool Implementations
# =============================================================================

AVAILABLE_TOOLS = [
    "verify_user",
    "check_order",
    "issue_refund",
    "check_policy",
    "escalate_to_human",
    "route_to_regional_team",
]

# Tool descriptions for the LLM
TOOL_DESCRIPTIONS = {
    "verify_user": {
        "name": "verify_user",
        "description": "Verify a customer's identity and retrieve their account information.",
        "parameters": {"user_id": "string - The user's unique ID (e.g., USR-1001)"},
    },
    "check_order": {
        "name": "check_order",
        "description": "Look up an order's details including product, price, status, and tracking.",
        "parameters": {"order_id": "string - The order ID (e.g., ORD-5001)"},
    },
    "issue_refund": {
        "name": "issue_refund",
        "description": "Process a refund for a delivered, return-eligible order.",
        "parameters": {
            "order_id": "string - The order ID to refund",
            "reason": "string - Reason for the refund",
        },
    },
    "check_policy": {
        "name": "check_policy",
        "description": "Look up company policy on a topic (refund, shipping, returns, warranty, cancellation).",
        "parameters": {"topic": "string - The policy topic"},
    },
    "escalate_to_human": {
        "name": "escalate_to_human",
        "description": "Escalate the case to a human agent when the issue is too complex.",
        "parameters": {"reason": "string - Reason for escalation"},
    },
    "route_to_regional_team": {
        "name": "route_to_regional_team",
        "description": "Route the case to a specialized regional team when the customer speaks a different language.",
        "parameters": {
            "language": "string - The language the customer is speaking",
            "reason": "string - Reason for routing"
        },
    },
}


def verify_user(user_id: str = "", ctx: Optional[ToolContext] = None, **kwargs) -> Dict[str, Any]:
    """Verify a user's identity and account status."""
    if not user_id:
        return {"success": False, "error": "user_id is required"}

    db = ctx.users_db if ctx else USERS_DB
    user = db.get(user_id)
    if not user:
        return {"success": False, "error": f"User {user_id} not found"}

    return {
        "success": True,
        "user_id": user_id,
        "name": user["name"],
        "email": user["email"],
        "verified": user["verified"],
        "account_status": user["account_status"],
        "membership": user["membership"],
    }


def check_order(order_id: str = "", ctx: Optional[ToolContext] = None, **kwargs) -> Dict[str, Any]:
    """Look up an order's details and status."""
    if not order_id:
        return {"success": False, "error": "order_id is required"}

    db = ctx.orders_db if ctx else ORDERS_DB
    order = db.get(order_id)
    if not order:
        return {"success": False, "error": f"Order {order_id} not found"}

    return {
        "success": True,
        "order_id": order_id,
        "user_id": order["user_id"],
        "product": order["product"],
        "price": order["price"],
        "status": order["status"],
        "tracking": order["tracking"],
        "delivery_date": order["delivery_date"],
        "return_eligible": order["return_eligible"],
        "category": order["category"],
    }


def issue_refund(order_id: str = "", reason: str = "", ctx: Optional[ToolContext] = None, **kwargs) -> Dict[str, Any]:
    """Process a refund for an order."""
    if not order_id:
        return {"success": False, "error": "order_id is required"}
    if not reason:
        return {"success": False, "error": "reason is required"}

    db = ctx.orders_db if ctx else ORDERS_DB
    order = db.get(order_id)
    if not order:
        return {"success": False, "error": f"Order {order_id} not found"}

    if order["status"] != "delivered":
        return {
            "success": False,
            "error": f"Cannot refund order with status '{order['status']}'",
        }

    if not order["return_eligible"]:
        return {"success": False, "error": "Order is not eligible for return/refund"}

    policy = ctx.refund_policy if ctx else REFUND_POLICY
    if order["category"] in policy["non_refundable_categories"]:
        return {
            "success": False,
            "error": f"Items in category '{order['category']}' are non-refundable per company policy",
        }

    if ctx:
        ctx.refund_log[order_id] = True

    return {
        "success": True,
        "order_id": order_id,
        "refund_amount": order["price"],
        "refund_method": "original_payment",
        "estimated_days": 5,
        "confirmation_id": f"REF-{order_id[-4:]}",
    }


def check_policy(topic: str = "", ctx: Optional[ToolContext] = None, **kwargs) -> Dict[str, Any]:
    """Look up company policy on a topic."""
    if not topic:
        return {"success": False, "error": "topic is required"}

    policies = {
        "refund": {
            "success": True,
            "topic": "refund",
            "policy": "Refunds are available within 30 days of delivery for eligible items. "
            "User verification and order lookup are required before processing. "
            "Digital downloads and gift cards are non-refundable.",
        },
        "shipping": {
            "success": True,
            "topic": "shipping",
            "policy": "Standard shipping takes 5-7 business days. "
            "Express shipping takes 2-3 business days. Tracking is provided for all orders. "
            "Processing orders are prepared within 24-48 hours.",
        },
        "returns": {
            "success": True,
            "topic": "returns",
            "policy": "Items must be in original condition within 30 days of delivery. "
            "Digital downloads and gift cards are non-refundable. "
            "Return shipping labels are provided for eligible items.",
        },
        "warranty": {
            "success": True,
            "topic": "warranty",
            "policy": "Electronics have a 1-year manufacturer warranty. "
            "Accessories have a 90-day warranty. Warranty claims require "
            "order verification and product inspection.",
        },
        "cancellation": {
            "success": True,
            "topic": "cancellation",
            "policy": "Orders in 'processing' status can be cancelled without penalty. "
            "Shipped orders cannot be cancelled but can be returned after delivery. "
            "Refund will be issued within 5 business days of cancellation.",
        },
    }

    result = policies.get(topic.lower())
    if not result:
        return {
            "success": True,
            "topic": topic,
            "policy": f"No specific policy found for '{topic}'. "
            "Please escalate to a human agent for further assistance.",
        }
    return result


def escalate_to_human(reason: str = "", ctx: Optional[ToolContext] = None, **kwargs) -> Dict[str, Any]:
    """Escalate the case to a human agent."""
    if not reason:
        return {"success": False, "error": "reason is required"}

    return {
        "success": True,
        "escalation_id": "ESC-7890",
        "assigned_agent": "Agent Sarah",
        "estimated_wait": "2 minutes",
        "priority": "high" if any(w in reason.lower() for w in ["fraud", "security", "urgent"]) else "normal",
        "message": f"Case escalated: {reason}",
    }


def route_to_regional_team(language: str = "", reason: str = "", ctx: Optional[ToolContext] = None, **kwargs) -> Dict[str, Any]:
    """Route the case to a specialized regional team."""
    if not language:
        return {"success": False, "error": "language is required"}
    if not reason:
        return {"success": False, "error": "reason is required"}

    return {
        "success": True,
        "routing_id": "RTG-1234",
        "assigned_team": f"{language.capitalize()} Regional Support",
        "priority": "high",
        "message": f"Case routed to {language.capitalize()} team: {reason}",
    }


# =============================================================================
# Tool Dispatcher
# =============================================================================

TOOL_REGISTRY = {
    "verify_user": verify_user,
    "check_order": check_order,
    "issue_refund": issue_refund,
    "check_policy": check_policy,
    "escalate_to_human": escalate_to_human,
    "route_to_regional_team": route_to_regional_team,
}


def call_tool(tool_name: str, tool_args: Dict[str, Any], ctx: Optional[ToolContext] = None) -> Dict[str, Any]:
    """Dispatch a tool call and return the result."""
    if tool_name not in TOOL_REGISTRY:
        return {
            "success": False,
            "error": f"Unknown tool '{tool_name}'. Available: {AVAILABLE_TOOLS}",
        }

    fn = TOOL_REGISTRY[tool_name]
    try:
        return fn(**tool_args, ctx=ctx)
    except Exception as e:
        return {"success": False, "error": f"Invalid arguments or tool execution failure for '{tool_name}': {e}"}
