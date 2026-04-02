import asyncio
from customer_service_env.client import CustomerServiceEnv
from customer_service_env.models import CustomerServiceAction

# Define the perfect sequence of actions for each of the 6 scenarios
SCENARIO_TESTS = {
    "easy_order_status": [
        CustomerServiceAction(tool_name="check_order", tool_args={"order_id": "ORD-5002"}),
        CustomerServiceAction(message="Your order shipped! Tracking: TRK-112233")
    ],
    "easy_order_cancel": [
        CustomerServiceAction(tool_name="check_order", tool_args={"order_id": "ORD-5006"}),
        CustomerServiceAction(tool_name="check_policy", tool_args={"topic": "cancellation"}),
        CustomerServiceAction(message="I have cancelled your processing order ORD-5006 as requested.")
    ],
    "medium_refund_request": [
        CustomerServiceAction(tool_name="verify_user", tool_args={"user_id": "USR-1001"}),
        CustomerServiceAction(tool_name="check_order", tool_args={"order_id": "ORD-5001"}),
        CustomerServiceAction(tool_name="issue_refund", tool_args={"order_id": "ORD-5001", "amount": 79.99, "reason": "Defective item"}),
        CustomerServiceAction(message="I've issued a refund of 79.99 for your defect item REF-5001.")
    ],
    "hard_fraud_detection": [
        CustomerServiceAction(tool_name="verify_user", tool_args={"user_id": "USR-1002"}),
        CustomerServiceAction(tool_name="check_order", tool_args={"order_id": "ORD-5003"}),
        CustomerServiceAction(tool_name="check_order", tool_args={"order_id": "ORD-5004"}),
        CustomerServiceAction(tool_name="check_policy", tool_args={"topic": "refunds"}),
        CustomerServiceAction(tool_name="issue_refund", tool_args={"order_id": "ORD-5004", "amount": 149.99, "reason": "Duplicate charge"}),
        CustomerServiceAction(message="I found the duplicate order ORD-5004 and have issued a refund for it.")
    ],
    "hard_non_refundable": [
        CustomerServiceAction(tool_name="verify_user", tool_args={"user_id": "USR-1005"}),
        CustomerServiceAction(tool_name="check_order", tool_args={"order_id": "ORD-5009"}),
        CustomerServiceAction(tool_name="check_policy", tool_args={"topic": "refunds"}),
        CustomerServiceAction(tool_name="issue_refund", tool_args={"order_id": "ORD-5009", "amount": 199.99, "reason": "Quality issue"}),
        CustomerServiceAction(tool_name="escalate_to_human", tool_args={"summary": "Customer demands refund for non-refundable digital purchase", "department": "customer_success"}),
        CustomerServiceAction(message="I apologize but digital items are non-refundable. I will escalate this to a human manager to assist you.")
    ],
    "hard_multilingual": [
        CustomerServiceAction(tool_name="verify_user", tool_args={"user_id": "USR-1002"}),
        CustomerServiceAction(tool_name="check_order", tool_args={"order_id": "ORD-5003"}),
        CustomerServiceAction(tool_name="route_to_regional_team", tool_args={"language": "spanish", "reason": "Defective mechanical keyboard"}),
        CustomerServiceAction(message="Lo transferiré a nuestro equipo regional en español.")
    ]
}

async def run_all_tests():
    URL = "https://pranavagarkar-customer-service-env.hf.space"
    print(f"Connecting to live OpenEnv deployment at: {URL}\n")
    
    env = CustomerServiceEnv(base_url=URL)
    
    total_score = 0.0
    
    for scenario_id, actions in SCENARIO_TESTS.items():
        print(f"==================================================")
        print(f"Testing Scenario: {scenario_id}")
        await env.reset(scenario_id=scenario_id)
        
        step_idx = 1
        for action in actions:
            res = await env.step(action)
            action_desc = action.tool_name if action.tool_name else f"Message: '{action.message[:30]}...'"
            print(f"  Step {step_idx} | Action: {action_desc:<30} | Reward: +{res.reward:<4.2f} | Done: {res.done}")
            step_idx += 1
            if res.done:
                break
                
        # To get the final score from the last observation
        print(f"-> Resulting Status: {res.observation.feedback.split('|')[1].strip()}")
        total_score += 1.0 # Assuming all tests hit 1.0
        
if __name__ == "__main__":
    asyncio.run(run_all_tests())
