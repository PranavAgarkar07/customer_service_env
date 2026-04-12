from typing import Any, Dict, List, Set
from typing_extensions import Protocol


class EpisodeStateProtocol(Protocol):
    step_count: int
    tools_called: List[str]
    escalated: bool
    user_verified: bool
    resolved: bool


class ScenarioProtocol(Protocol):
    minimum_steps: int
    required_tools: List[str]

    def terminal_state_check(self, ctx: Any, state: EpisodeStateProtocol) -> bool:
        ...


class RewardEngine:
    """
    Computes outcome-based and shaped rewards for the RL environment.

    Design principles:
    - Intermediate rewards are small (0.02) to avoid reward hacking.
    - Correct-tool rewards (0.05) clearly outweigh incorrect-tool rewards (0.0)
      to provide gradient signal without over-specifying the sequence.
    - Terminal reward is outcome-based + efficiency-scaled so the agent learns
      to solve tasks in minimum steps.
    - Partial credit on timeout prevents the cliff-edge reward problem where
      an agent doing everything right except the final step gets 0.
    """

    def compute_step_reward(
        self,
        action: Any,
        tool_result: Dict[str, Any],
        state: EpisodeStateProtocol,
        required_tools: List[str] = None,
    ) -> float:
        """Shaped intermediate reward per step.

        Provides a small positive signal for useful tool calls to guide
        gradient descent toward the optimal path without hard-coding the sequence.
        """
        if not action.tool_name and not action.message:
            return -0.05  # Penalty for doing nothing

        reward = 0.0

        if action.tool_name and tool_result:
            if tool_result.get("success"):
                # Base signal for any successful tool call
                reward += 0.02

                # Larger signal if the tool is in the required set for this scenario
                if required_tools and action.tool_name in required_tools:
                    reward += 0.03  # Makes correct tools worth 0.05 total vs. 0.02 for irrelevant ones

            # Small penalty for wrong credentials (e.g., verify_user with wrong ID)
            if not tool_result.get("success") and tool_result.get("error", "").startswith("User"):
                reward -= 0.02

        return reward

    def compute_terminal_reward(
        self,
        scenario: ScenarioProtocol,
        ctx: Any,
        state: EpisodeStateProtocol,
    ) -> float:
        """Outcome-based terminal reward with efficiency scaling and partial credit.

        Returns:
            - 0.5–1.0 scaled by efficiency if outcome was achieved
            - 0.05–0.25 partial credit if key required tools were used (timeout case)
            - 0.0 if neither outcome nor partial progress was made
        """
        outcome_achieved = scenario.terminal_state_check(ctx, state)

        if outcome_achieved:
            # --- Full outcome: efficiency-scaled reward ---
            steps_taken = state.step_count
            min_steps = scenario.minimum_steps

            # Efficiency: 1.0 if solved in minimum steps, decays as extra steps are used
            efficiency = min_steps / max(steps_taken, min_steps)

            # Redundancy penalty: penalize extra tool calls beyond what was needed
            tools_used: List[str] = state.tools_called
            required: List[str] = getattr(scenario, "required_tools", [])
            redundant_calls = max(0, len(tools_used) - len(required))
            redundancy_penalty = redundant_calls * 0.04

            # Base reward 0.5 + efficiency bonus up to 0.5, minus redundancy
            terminal = 0.5 + (0.5 * efficiency) - redundancy_penalty
            return max(0.15, terminal)  # floor to ensure success always pays

        # --- Partial credit: agent timed out but did useful work ---
        # Reward proportional to how many required tools they actually called
        required: List[str] = getattr(scenario, "required_tools", [])
        if required:
            tools_used_set: Set[str] = set(state.tools_called)
            required_set: Set[str] = set(required)
            coverage = len(tools_used_set & required_set) / len(required_set)
            if coverage > 0:
                # Up to 0.2 partial credit, only if they were on the right track
                return coverage * 0.2

        return 0.0
