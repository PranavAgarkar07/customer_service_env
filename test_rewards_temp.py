import asyncio
from client import CustomerServiceEnv
from models import CustomerServiceAction

async def test_rewards():
    env = CustomerServiceEnv(base_url="https://pranavagarkar-customer-service-env.hf.space")
    await env.reset(scenario_id="hard_multilingual")
    
    # Tool 1: verify_user
    r1 = await env.step(CustomerServiceAction(tool_name="verify_user", tool_args={"user_id": "USR-1002"}))
    print(f"Step 1 - verify_user: Reward = {r1.reward}")
    
    # Tool 2: check_order
    r2 = await env.step(CustomerServiceAction(tool_name="check_order", tool_args={"order_id": "ORD-5003"}))
    print(f"Step 2 - check_order: Reward = {r2.reward}")
    
    # Tool 3: route_to_regional_team
    r3 = await env.step(CustomerServiceAction(tool_name="route_to_regional_team", tool_args={"language": "spanish", "reason": "Customer speaks spanish"}))
    print(f"Step 3 - route_to_regional_team: Reward = {r3.reward} | Done = {r3.done}")

asyncio.run(test_rewards())
