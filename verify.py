#!/usr/bin/env python3
"""
verify.py — Customer Service Environment Verification Suite
============================================================

Graphify identified test_rewards_temp.py and test_benchmark.py as orphaned
thin communities with no connection to the main codebase. This script
replaces both, connects to the core abstractions, and runs 4 verification
tiers in a single command:

  Tier 1 — Static:  Import all core modules (catches syntax/import errors)
  Tier 2 — Oracle:  Instantiate env directly (no Docker) and run each scenario
  Tier 3 — HTTP:    Hit the live container's /reset and /step endpoints
  Tier 4 — Runtime: Run `openenv validate --url` against the live container

Usage:
    uv run python verify.py                      # all tiers (needs Docker on :8000)
    uv run python verify.py --tier static        # imports only — no Docker needed
    uv run python verify.py --tier oracle        # direct env — no Docker needed
    uv run python verify.py --tier http          # needs Docker on :8000
    uv run python verify.py --url http://localhost:8000   # override base URL
"""

import argparse
import json
import sys
import traceback
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────────── #
# Helpers
# ──────────────────────────────────────────────────────────────────────────── #

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

results: List[Tuple[bool, str]] = []


def ok(msg: str) -> None:
    results.append((True, msg))
    print(f"  {PASS} {msg}")


def fail(msg: str, detail: str = "") -> None:
    results.append((False, msg))
    print(f"  {FAIL} {msg}")
    if detail:
        print(f"      {detail}")


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ──────────────────────────────────────────────────────────────────────────── #
# Tier 1 — Static: import all core modules
# ──────────────────────────────────────────────────────────────────────────── #

def tier_static() -> None:
    section("Tier 1 — Static import check")

    modules = [
        ("models",                      "CustomerServiceAction, CustomerServiceObservation, CustomerServiceState"),
        ("server.scenario_generator",   "ScenarioGenerator, GeneratedScenario"),
        ("server.scenarios",            "get_scenario, list_scenarios"),
        ("server.tool_context",         "ToolContext"),
        ("server.reward_engine",        "RewardEngine"),
        ("server.rubrics",              "CustomerServiceRubric, BaseRubric"),
        ("server.customer_service_env_environment", "CustomerServiceEnvironment, safe_reward"),
    ]

    for mod_name, symbols in modules:
        try:
            mod = __import__(mod_name, fromlist=symbols.split(", "))
            for sym in symbols.split(", "):
                assert hasattr(mod, sym.strip()), f"Missing symbol: {sym}"
            ok(f"{mod_name}  ({symbols})")
        except Exception as e:
            fail(f"{mod_name}", str(e))


# ──────────────────────────────────────────────────────────────────────────── #
# Tier 2 — Oracle: direct environment instantiation
# ──────────────────────────────────────────────────────────────────────────── #

def tier_oracle() -> None:
    section("Tier 2 — Oracle (direct env, no Docker)")

    try:
        from server.customer_service_env_environment import CustomerServiceEnvironment, safe_reward
        from models import CustomerServiceAction
        from server.scenarios import list_scenarios
    except ImportError as e:
        fail("Import failed — run from project root", str(e))
        return

    # 2a. safe_reward bounds
    # safe_reward maps [-0.5, 1.5] → [0.01, 0.99] linearly
    # formula: 0.01 + (r + 0.5) / 2.0 * 0.98
    # 0.85 → 0.01 + 1.35/2.0*0.98 = 0.01 + 0.6615 = 0.6715
    for raw, expected_lo, expected_hi in [
        (-0.5,  0.01, 0.02),   # very negative → floor
        (0.00,  0.24, 0.27),   # zero → 0.255
        (0.05,  0.27, 0.29),   # small positive → 0.2795
        (0.85,  0.65, 0.69),   # good → 0.6715 per linear formula
        (1.50,  0.98, 0.99),   # above range → ceiling clamp
    ]:
        r = safe_reward(raw)
        if expected_lo <= r <= expected_hi:
            ok(f"safe_reward({raw:+.2f}) = {r:.4f}  ∈ [{expected_lo}, {expected_hi}]")
        else:
            fail(f"safe_reward({raw:+.2f}) = {r:.4f}  NOT in [{expected_lo}, {expected_hi}]")

    # 2b. All 6 scenarios: reset + optimal tool sequence → should reach done
    OPTIMAL_SEQUENCES = {
        "order_status":    ["check_order"],
        "order_cancel":    ["check_order", "check_policy"],
        "refund_request":  ["verify_user", "check_order", "issue_refund"],
        "fraud_duplicate": ["verify_user", "check_order", "check_order", "check_policy", "issue_refund"],
        "non_refundable":  ["verify_user", "check_order", "check_policy", "issue_refund", "escalate_to_human"],
        "multilingual":    ["verify_user", "check_order", "route_to_regional_team"],
    }

    TOOL_ARGS = {
        "check_order":           lambda ctx: {"order_id": list(ctx.orders_db.keys())[0]},
        "verify_user":           lambda ctx: {"user_id": list(ctx.users_db.keys())[0]},
        "check_policy":          lambda ctx: {"topic": "refund"},
        "issue_refund":          lambda ctx: {"order_id": list(ctx.orders_db.keys())[0], "reason": "test"},
        "escalate_to_human":     lambda ctx: {"reason": "non-refundable digital item"},
        "route_to_regional_team":lambda ctx: {"language": "spanish", "reason": "multilingual customer"},
    }

    for scenario_id, tools in OPTIMAL_SEQUENCES.items():
        try:
            env = CustomerServiceEnvironment()
            obs = env.reset(scenario_id=scenario_id, seed=42)
            ctx = env._ctx

            rewards = []
            done = False
            for tool_name in tools:
                args = TOOL_ARGS[tool_name](ctx)
                action = CustomerServiceAction(tool_name=tool_name, tool_args=args)
                obs = env.step(action)
                rewards.append(obs.reward)
                done = obs.done
                if done:
                    break

            total = sum(rewards)
            # All rewards must be in (0, 1)
            invalid = [r for r in rewards if not (0 < r < 1)]
            if invalid:
                fail(f"{scenario_id}: rewards out of (0,1): {invalid}")
            elif not done:
                fail(f"{scenario_id}: not done after optimal sequence (done={done})")
            elif total <= 0:
                fail(f"{scenario_id}: total reward ≤ 0: {total:.4f}")
            else:
                ok(f"{scenario_id}: done={done}  total={total:.4f}  steps={len(rewards)}  rewards={[round(r,3) for r in rewards]}")
        except Exception as e:
            fail(f"{scenario_id}: exception", traceback.format_exc(limit=2))

    # 2c. Reward range check: verify worst-case (wrong tools) stays in bounds
    try:
        env = CustomerServiceEnvironment()
        env.reset(scenario_id="refund_request", seed=99)
        ctx = env._ctx
        order_id = list(ctx.orders_db.keys())[0]

        # Call wrong tool 3 times to trigger penalties
        for _ in range(3):
            action = CustomerServiceAction(
                tool_name="route_to_regional_team",
                tool_args={"language": "spanish", "reason": "wrong tool"}
            )
            obs = env.step(action)
            if not (0 < obs.reward < 1):
                fail(f"Penalty reward out of (0,1): {obs.reward}")
                return
        ok(f"Penalty rewards stay in (0,1): latest={obs.reward:.4f}")
    except Exception as e:
        fail("Penalty reward check", str(e))


# ──────────────────────────────────────────────────────────────────────────── #
# Tier 3 — HTTP: live container check
# ──────────────────────────────────────────────────────────────────────────── #

def tier_http(base_url: str) -> None:
    section(f"Tier 3 — HTTP live check ({base_url})")

    try:
        import urllib.request
        import urllib.error
    except ImportError:
        fail("urllib not available")
        return

    def post(path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def get(path: str) -> dict:
        req = urllib.request.Request(f"{base_url}{path}")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    # Health
    try:
        r = get("/health")
        if r.get("status") == "healthy":
            ok(f"GET /health → {r}")
        else:
            fail(f"GET /health → unexpected: {r}")
    except Exception as e:
        fail("GET /health", str(e))
        return  # if health fails, no point continuing

    # /reset for each scenario
    SCENARIOS = ["order_status", "order_cancel", "refund_request",
                 "fraud_duplicate", "non_refundable", "multilingual"]

    for scenario_id in SCENARIOS:
        try:
            r = post("/reset", {"scenario_id": scenario_id, "seed": 1})
            obs = r.get("observation", r)
            reward = r.get("reward", obs.get("reward", "?"))
            done = r.get("done", obs.get("done", "?"))
            ok(f"POST /reset  scenario={scenario_id}  reward={reward}  done={done}")
        except Exception as e:
            fail(f"POST /reset  scenario={scenario_id}", str(e))

    # /state — verify state endpoint returns expected fields
    try:
        post("/reset", {"scenario_id": "order_status", "seed": 1})
        r = get("/state")
        has_episode = "episode_id" in r or "state" in r
        if has_episode:
            ok(f"GET /state → has episode_id/state field")
        else:
            fail(f"GET /state → unexpected response keys: {list(r.keys())}")
    except Exception as e:
        fail("GET /state", str(e))

    # /schema — verify action schema has tool_name
    try:
        r = get("/schema")
        action_props = r.get("action", {}).get("properties", {})
        if "tool_name" in action_props:
            ok(f"GET /schema → action.properties has tool_name")
        else:
            fail(f"GET /schema → missing tool_name in action schema")
    except Exception as e:
        fail("GET /schema", str(e))


# ──────────────────────────────────────────────────────────────────────────── #
# Tier 4 — openenv validate
# ──────────────────────────────────────────────────────────────────────────── #

def tier_openenv(base_url: str) -> None:
    section(f"Tier 4 — openenv validate ({base_url})")
    import subprocess

    # Local file validation
    try:
        r = subprocess.run(
            ["uv", "run", "openenv", "validate", "."],
            capture_output=True, text=True, timeout=30
        )
        output = r.stdout + r.stderr
        if "Not ready" in output or r.returncode != 0:
            # Extract specific issues
            lines = [l.strip() for l in output.splitlines() if l.strip()]
            for line in lines:
                if line.startswith("-"):
                    fail(f"openenv validate (local): {line}")
                elif "YES" in line or "NO" in line:
                    icon = PASS if "YES" in line else WARN
                    print(f"    {icon} {line}")
        else:
            ok("openenv validate (local): all checks passed")
    except Exception as e:
        fail("openenv validate (local)", str(e))

    # Runtime validation
    try:
        r = subprocess.run(
            ["uv", "run", "openenv", "validate", "--url", base_url],
            capture_output=True, text=True, timeout=30
        )
        report = json.loads(r.stdout)
        passed = report.get("summary", {}).get("passed_count", 0)
        total = report.get("summary", {}).get("total_count", 0)
        failed = report.get("summary", {}).get("failed_criteria", [])
        if not failed:
            ok(f"openenv validate --url: {passed}/{total} criteria passed")
        else:
            fail(f"openenv validate --url: {passed}/{total} — failed: {failed}")
    except Exception as e:
        fail(f"openenv validate --url {base_url}", str(e))


# ──────────────────────────────────────────────────────────────────────────── #
# Main
# ──────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="Customer Service Env — Verification Suite")
    parser.add_argument("--tier", choices=["static", "oracle", "http", "all"], default="all")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of running container")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  Customer Service Env — Verification Suite")
    print("  (replaces test_rewards_temp.py + test_benchmark.py)")
    print("═"*60)

    if args.tier in ("static", "all"):
        tier_static()
    if args.tier in ("oracle", "all"):
        tier_oracle()
    if args.tier in ("http", "all"):
        tier_http(args.url)
        tier_openenv(args.url)

    # Summary
    passed = sum(1 for ok_, _ in results if ok_)
    total = len(results)
    print(f"\n{'═'*60}")
    if passed == total:
        print(f"  \033[92m✓ ALL {total} CHECKS PASSED\033[0m")
    else:
        print(f"  \033[91m✗ {total - passed}/{total} CHECKS FAILED\033[0m")
        for ok_, msg in results:
            if not ok_:
                print(f"    ✗ {msg}")
    print("═"*60 + "\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
