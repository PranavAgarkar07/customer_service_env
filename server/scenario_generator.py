import random
from dataclasses import dataclass
from typing import Dict, List, Any, Callable, Optional

try:
    from .tool_context import ToolContext
except ImportError:
    from server.tool_context import ToolContext

# --- Data Pools ---

CUSTOMER_POOL = [
    # English-speaking customers
    {"first": "Alice", "last": "Johnson", "lang": "english", "membership": "premium"},
    {"first": "Derek", "last": "Wilson", "lang": "english", "membership": "basic"},
    {"first": "Elena", "last": "Martinez", "lang": "english", "membership": "premium"},
    {"first": "James", "last": "Carter", "lang": "english", "membership": "basic"},
    {"first": "Sarah", "last": "Thompson", "lang": "english", "membership": "premium"},
    {"first": "Mike", "last": "Davis", "lang": "english", "membership": "basic"},
    {"first": "Laura", "last": "White", "lang": "english", "membership": "premium"},
    {"first": "Kevin", "last": "Brown", "lang": "english", "membership": "basic"},
    {"first": "Sofia", "last": "Anderson", "lang": "english", "membership": "premium"},
    {"first": "Ryan", "last": "Taylor", "lang": "english", "membership": "basic"},
    # Non-English customers (for multilingual routing scenarios)
    {"first": "Carlos", "last": "Mendez", "lang": "espanol", "membership": "basic", "multilingual_query": "Hola, tengo un problema con mi pedido."},
    {"first": "Yuki", "last": "Tanaka", "lang": "japanese", "membership": "premium", "multilingual_query": "注文に問題があります。"},
    {"first": "Fatima", "last": "Al-Rashid", "lang": "arabic", "membership": "basic", "multilingual_query": "لدي مشكلة في طلبي."},
    {"first": "Ivan", "last": "Petrov", "lang": "russian", "membership": "basic", "multilingual_query": "У меня проблема с заказом."},
    {"first": "Mei", "last": "Lin", "lang": "mandarin", "membership": "premium", "multilingual_query": "我的订单有问题。"},
    {"first": "Lucas", "last": "Silva", "lang": "portuguese", "membership": "premium", "multilingual_query": "Tenho um problema com meu pedido."},
    {"first": "Amara", "last": "Osei", "lang": "french", "membership": "basic", "multilingual_query": "J'ai un problème avec ma commande."},
    {"first": "Priya", "last": "Sharma", "lang": "hindi", "membership": "premium", "multilingual_query": "मेरे ऑर्डर में समस्या है।"},
]

PRODUCT_POOL = [
    # Eligible for return
    {"name": "Wireless Headphones", "category": "electronics", "price_range": (49, 299), "return_eligible": True},
    {"name": "Mechanical Keyboard", "category": "electronics", "price_range": (79, 399), "return_eligible": True},
    {"name": "USB-C Cable", "category": "accessories", "price_range": (9, 29), "return_eligible": True},
    {"name": "Laptop Sleeve", "category": "accessories", "price_range": (19, 59), "return_eligible": True},
    {"name": "Webcam HD Pro", "category": "electronics", "price_range": (39, 199), "return_eligible": True},
    {"name": "Ergonomic Mouse", "category": "electronics", "price_range": (29, 149), "return_eligible": True},
    {"name": "Monitor Stand", "category": "furniture", "price_range": (30, 150), "return_eligible": True},
    {"name": "Gaming Chair", "category": "furniture", "price_range": (150, 600), "return_eligible": True},
    {"name": "Portable SSD", "category": "electronics", "price_range": (59, 299), "return_eligible": True},
    {"name": "Noise-Cancelling Earbuds", "category": "electronics", "price_range": (49, 249), "return_eligible": True},
    {"name": "Smart Watch", "category": "electronics", "price_range": (99, 499), "return_eligible": True},
    {"name": "LED Desk Lamp", "category": "accessories", "price_range": (19, 89), "return_eligible": True},
    {"name": "Mechanical Pencil Set", "category": "stationery", "price_range": (9, 49), "return_eligible": True},
    {"name": "Running Shoes", "category": "apparel", "price_range": (49, 199), "return_eligible": True},
    {"name": "Yoga Mat", "category": "sports", "price_range": (19, 99), "return_eligible": True},
    # Non-refundable items
    {"name": "Digital Art Course", "category": "digital_download", "price_range": (29, 499), "return_eligible": False},
    {"name": "Software License", "category": "digital_download", "price_range": (99, 999), "return_eligible": False},
    {"name": "Online Music Subscription (1 Year)", "category": "digital_download", "price_range": (49, 149), "return_eligible": False},
    {"name": "Gift Card", "category": "gift_cards", "price_range": (10, 500), "return_eligible": False},
    {"name": "e-Book Bundle", "category": "digital_download", "price_range": (19, 99), "return_eligible": False},
    {"name": "Video Game Download", "category": "digital_download", "price_range": (9, 79), "return_eligible": False},
]

# Separate pool for only return-eligible and non-eligible for targeted selection
RETURN_ELIGIBLE_PRODUCTS = [p for p in PRODUCT_POOL if p["return_eligible"]]
NON_REFUNDABLE_PRODUCTS = [p for p in PRODUCT_POOL if not p["return_eligible"]]

# --- Scenario Definition ---

@dataclass
class GeneratedScenario:
    scenario_type: str
    difficulty: str
    seed: int

    customer_name: str
    customer_id: str
    language: str

    primary_order_id: str
    all_order_ids: List[str]
    customer_query: str

    required_tools: List[str]
    minimum_steps: int
    resolution_keywords: List[str]
    
    # Needs to accept context and state
    terminal_state_check: Callable[[ToolContext, Any], bool]
    
    tool_context: ToolContext


class ScenarioGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed if seed is not None else random.randint(0, 100000)
        self.rng = random.Random(self.seed)

    def generate(self, scenario_type: str, difficulty: Optional[str] = None) -> GeneratedScenario:
        # 1. Draw customer and product
        cust = self.rng.choice(CUSTOMER_POOL)
        prod = self.rng.choice(PRODUCT_POOL)
        
        customer_name = f"{cust['first']} {cust['last']}"
        customer_id = f"USR-{self.rng.randint(10000, 99999)}"
        primary_order_id = f"ORD-{self.rng.randint(10000, 99999)}"
        tracking_num = f"TRK-{self.rng.randint(100000, 999999)}"
        price = round(self.rng.uniform(prod["price_range"][0], prod["price_range"][1]), 2)

        # 2. Setup Tool Context
        ctx = ToolContext()
        
        ctx.users_db[customer_id] = {
            "name": customer_name,
            "email": f"{cust['first'].lower()}@example.com",
            "verified": True,
            "account_status": "active",
            "membership": cust["membership"],
            "join_date": "2024-01-01",
        }
        
        ctx.refund_policy = {
            "max_days_after_delivery": 30,
            "requires_user_verification": True,
            "requires_order_check": True,
            "eligible_statuses": ["delivered"],
            "non_refundable_categories": ["digital_download", "gift_cards"],
        }
        
        # Base Order
        ctx.orders_db[primary_order_id] = {
            "user_id": customer_id,
            "product": prod["name"],
            "price": price,
            "status": "delivered",
            "tracking": tracking_num,
            "delivery_date": "2026-03-25",
            "return_eligible": prod["return_eligible"],
            "category": prod["category"],
        }

        diff = difficulty or "medium"
        query = ""
        req_tools = []
        min_steps = 1
        res_keywords = []

        # ------------------------------------------------------------------ #
        # Terminal check functions — MUST NOT reference state.resolved       #
        # (resolved is set AFTER terminal_state_check returns, so it would   #
        # create a circular dependency that prevents natural termination)     #
        # ------------------------------------------------------------------ #

        # Bug #1 fix: removed `state.resolved` — only inspect observable state
        # Bug #5 fix: verify the correct order ID was used (not just any check_order call)
        def check_status_term(c: ToolContext, state: Any) -> bool:
            # Primary: correct order ID appeared in tool_args_log
            if any(primary_order_id in arg for arg in getattr(c, 'tool_args_log', [])):
                return True
            # Fallback: agent called check_order at least once (for small models)
            return "check_order" in state.tools_called

        # Bug #2 fix: cancel requires check_order + check_policy, NOT issue_refund
        def check_cancel_term(c: ToolContext, state: Any) -> bool:
            return "check_order" in state.tools_called and "check_policy" in state.tools_called

        def check_refund_term(c: ToolContext, state: Any) -> bool:
            return c.refund_log.get(primary_order_id, False)

        def check_escalate_term(c: ToolContext, state: Any) -> bool:
            return state.escalated

        def check_route_term(c: ToolContext, state: Any) -> bool:
            return "route_to_regional_team" in state.tools_called

        term_check = check_status_term

        # --- Scenario Dispatch ---
        if scenario_type == "order_status":
            ctx.orders_db[primary_order_id]["status"] = "shipped"
            queries = [
                f"Hi, I'm {customer_name} (ID: {customer_id}). I placed order {primary_order_id} for {prod['name']}. What's the status and tracking info?",
                f"Can you check order {primary_order_id} for me? My user ID is {customer_id}. Need the tracking number.",
                f"Hello! {customer_name} here (ID: {customer_id}). Just wondering about the current status of my {prod['name']} order #{primary_order_id}.",
            ]
            query = self.rng.choice(queries)
            req_tools = ["check_order"]
            min_steps = 1
            res_keywords = ["shipped", tracking_num]
            term_check = check_status_term
            diff = "easy"

        elif scenario_type == "order_cancel":
            ctx.orders_db[primary_order_id]["status"] = "processing"
            queries = [
                f"Hi, I'm {customer_name} (ID: {customer_id}). I want to cancel order {primary_order_id} for {prod['name']}. It hasn't shipped yet.",
                f"Please cancel my order {primary_order_id} immediately. I'm {customer_name} (ID: {customer_id}) and I changed my mind.",
                f"{customer_name} here (ID: {customer_id}). Can I cancel {primary_order_id}? I ordered {prod['name']} by mistake.",
            ]
            query = self.rng.choice(queries)
            req_tools = ["check_order", "check_policy"]
            min_steps = 2
            res_keywords = ["cancel", "processing", primary_order_id]
            term_check = check_cancel_term
            diff = "easy"

        elif scenario_type == "refund_request":
            # Ensure product is return-eligible for a fair refund scenario
            prod = self.rng.choice(RETURN_ELIGIBLE_PRODUCTS)
            price = round(self.rng.uniform(prod["price_range"][0], prod["price_range"][1]), 2)
            ctx.orders_db[primary_order_id].update({
                "product": prod["name"], "price": price,
                "category": prod["category"], "return_eligible": True,
            })
            queries = [
                f"Hi, I'm {customer_name} (ID: {customer_id}). I received {primary_order_id} ({prod['name']}) but it's defective. I need a refund.",
                f"{customer_name} ({customer_id}) here. My {prod['name']} (order {primary_order_id}) arrived damaged. Please process a refund.",
                f"This is {customer_name}, ID {customer_id}. I'd like to return order {primary_order_id} ({prod['name']}) — it doesn't work as advertised.",
            ]
            query = self.rng.choice(queries)
            req_tools = ["verify_user", "check_order", "issue_refund"]
            min_steps = 3
            term_check = check_refund_term
            res_keywords = ["refund", "processed"]
            diff = "medium"

        elif scenario_type == "fraud_duplicate":
            # Ensure return-eligible product for the duplicate charge
            prod = self.rng.choice(RETURN_ELIGIBLE_PRODUCTS)
            price = round(self.rng.uniform(prod["price_range"][0], prod["price_range"][1]), 2)
            ctx.orders_db[primary_order_id].update({"product": prod["name"], "price": price, "category": prod["category"], "return_eligible": True})
            sec_order_id = f"ORD-{self.rng.randint(10000, 99999)}"
            ctx.orders_db[sec_order_id] = dict(ctx.orders_db[primary_order_id])  # exact duplicate
            queries = [
                f"Hi, I'm {customer_name} (ID: {customer_id}). I see two charges for {prod['name']}! Orders {primary_order_id} and {sec_order_id} — both ${price}. I only ordered once!",
                f"{customer_name} here (ID: {customer_id}). Something's wrong — I got billed twice for {prod['name']}. Orders {primary_order_id} and {sec_order_id} are both showing on my account.",
            ]
            query = self.rng.choice(queries)
            req_tools = ["verify_user", "check_order", "check_policy", "issue_refund"]
            min_steps = 4  # 4 required tools → minimum 4 steps (was wrongly 5)
            term_check = lambda c, s: c.refund_log.get(sec_order_id, False) or c.refund_log.get(primary_order_id, False)
            res_keywords = ["duplicate", "refund"]
            diff = "hard"

        elif scenario_type == "non_refundable":
            # Bug #4 fix: randomly select among all non-refundable products
            prod = self.rng.choice(NON_REFUNDABLE_PRODUCTS)
            price = round(self.rng.uniform(prod["price_range"][0], prod["price_range"][1]), 2)
            ctx.orders_db[primary_order_id].update({
                "product": prod["name"], "price": price,
                "category": prod["category"], "return_eligible": False,
            })
            queries = [
                f"Hello, I'm {customer_name} (ID: {customer_id}). I purchased {prod['name']} (order {primary_order_id}) for ${price} but the quality is terrible. I want a full refund.",
                f"{customer_name} here, ID {customer_id}. I'm very disappointed with my {prod['name']} purchase (order {primary_order_id}). Please refund me ${price} immediately.",
                f"This is {customer_name} (ID: {customer_id}). My recent purchase of {prod['name']} — order {primary_order_id} — is completely unusable. I demand a refund.",
            ]
            query = self.rng.choice(queries)
            req_tools = ["verify_user", "check_order", "check_policy", "issue_refund", "escalate_to_human"]
            min_steps = 5
            term_check = check_escalate_term
            res_keywords = ["non-refundable", "escalat"]
            diff = "hard"

        elif scenario_type == "multilingual":
            # Bug #3 fix: randomly select among all non-English customers
            non_english = [c for c in CUSTOMER_POOL if c.get("multilingual_query")]
            cust = self.rng.choice(non_english)
            customer_name = f"{cust['first']} {cust['last']}"
            ctx.users_db[customer_id]["name"] = customer_name
            base_query = cust["multilingual_query"]
            queries = [
                f"{base_query} Soy {customer_name} (ID: {customer_id}), pedido {primary_order_id}." if cust["lang"] == "espanol" else
                f"{base_query} ({customer_name}, ID: {customer_id}, order {primary_order_id})",
            ]
            query = queries[0]
            req_tools = ["verify_user", "check_order", "route_to_regional_team"]
            min_steps = 3
            # Two-tier terminal check:
            # - Primary: route_to_regional_team called (correct behavior, full reward)
            # - Fallback: verify + check_order done (avoids 12-step loops with small models)
            def check_route_term(c: ToolContext, state: Any) -> bool:
                if "route_to_regional_team" in state.tools_called:
                    return True
                # Fallback: if agent verified + checked but hasn't called route yet,
                # end episode after 4 steps to prevent infinite looping
                verified_and_checked = (
                    "verify_user" in state.tools_called and
                    "check_order" in state.tools_called
                )
                return verified_and_checked and state.step_count >= 4
            term_check = check_route_term
            res_keywords = [cust["lang"], "transfer", "regional"]
            diff = "hard"

        else:
            # Fallback: treat as order_status
            ctx.orders_db[primary_order_id]["status"] = "shipped"
            query = f"Hi, I'm {customer_name} ({customer_id}). Checking on order {primary_order_id}."
            req_tools = ["check_order"]
            min_steps = 1
            res_keywords = ["shipped"]
            term_check = check_status_term
            diff = "easy"

        return GeneratedScenario(
            scenario_type=scenario_type,
            difficulty=diff,
            seed=self.seed,
            customer_name=customer_name,
            customer_id=customer_id,
            language=cust.get("lang", "english"),
            primary_order_id=primary_order_id,
            all_order_ids=list(ctx.orders_db.keys()),
            customer_query=query,
            required_tools=req_tools,
            minimum_steps=min_steps,
            resolution_keywords=res_keywords,
            tool_context=ctx,
            terminal_state_check=term_check
        )
