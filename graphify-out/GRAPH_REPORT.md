# Graph Report - .  (2026-04-12)

## Corpus Check
- 14 files · ~9,768 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 101 nodes · 158 edges · 12 communities detected
- Extraction: 61% EXTRACTED · 39% INFERRED · 0% AMBIGUOUS · INFERRED: 61 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]

## God Nodes (most connected - your core abstractions)
1. `CustomerServiceAction` - 15 edges
2. `CustomerServiceObservation` - 15 edges
3. `CustomerServiceState` - 14 edges
4. `CustomerServiceEnvironment` - 13 edges
5. `ToolContext` - 11 edges
6. `GeneratedScenario` - 11 edges
7. `RewardEngine` - 10 edges
8. `CustomerServiceEnv` - 8 edges
9. `run_scenario()` - 8 edges
10. `Map any raw step reward to a value strictly inside (0.01, 0.99).      Per OpenEn` - 6 edges

## Surprising Connections (you probably didn't know these)
- `CustomerServiceEnvironment` --uses--> `CustomerServiceAction`  [INFERRED]
  server/customer_service_env_environment.py → models.py
- `Map any raw step reward to a value strictly inside (0.01, 0.99).      Per OpenEn` --uses--> `CustomerServiceAction`  [INFERRED]
  server/customer_service_env_environment.py → models.py
- `Customer Service Agent Environment.      The agent receives a customer query and` --uses--> `CustomerServiceAction`  [INFERRED]
  server/customer_service_env_environment.py → models.py
- `Reset the environment with a generative scenario.` --uses--> `CustomerServiceAction`  [INFERRED]
  server/customer_service_env_environment.py → models.py
- `Execute one step with outcome-based grading.` --uses--> `CustomerServiceAction`  [INFERRED]
  server/customer_service_env_environment.py → models.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.16
Nodes (15): Action, CustomerServiceEnv, Client for the Customer Service Agent Environment.      Example:         >>> wit, Convert action to JSON payload., Parse server response into StepResult., Parse server response into State object., Initialize the environment., CustomerServiceAction (+7 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (17): _call_llm_sync(), _extract_error(), get_agent_action(), log_end(), log_start(), log_step(), main(), Inference Script — Customer Service Agent Environment ========================== (+9 more)

### Community 2 - "Community 2"
Cohesion: 0.15
Nodes (16): Session-local snapshot of databases for one episode.      Created fresh per epis, ToolContext, call_tool(), check_order(), check_policy(), escalate_to_human(), issue_refund(), Verify a user's identity and account status. (+8 more)

### Community 3 - "Community 3"
Cohesion: 0.17
Nodes (10): CustomerServiceEnvironment, Execute one step with outcome-based grading., Map any raw step reward to a value strictly inside (0.01, 0.99).      Per OpenEn, Customer Service Agent Environment.      The agent receives a customer query and, Reset the environment with a generative scenario., safe_reward(), Environment, Computes outcome-based and shaped rewards for the RL environment.      Design pr (+2 more)

### Community 4 - "Community 4"
Cohesion: 0.29
Nodes (6): GeneratedScenario, ScenarioGenerator, get_scenario(), list_scenarios(), Get a procedurally generated scenario by ID or legacy alias.      Each call with, List all available scenario IDs with metadata.

### Community 5 - "Community 5"
Cohesion: 0.33
Nodes (4): Protocol, EpisodeStateProtocol, Outcome-based terminal reward with efficiency scaling and partial credit., ScenarioProtocol

### Community 6 - "Community 6"
Cohesion: 0.67
Nodes (2): main(), Run the server directly.

### Community 7 - "Community 7"
Cohesion: 1.0
Nodes (0): 

### Community 8 - "Community 8"
Cohesion: 1.0
Nodes (0): 

### Community 9 - "Community 9"
Cohesion: 1.0
Nodes (0): 

### Community 10 - "Community 10"
Cohesion: 1.0
Nodes (1): Convert string JSON or partial JSON to a dictionary.

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **17 isolated node(s):** `Action the agent takes each step.      The agent can:     - Call a tool by setti`, `Convert string JSON or partial JSON to a dictionary.`, `Observation the agent receives after each step.      Contains the customer query`, `Internal state tracking for the customer service environment.`, `Inference Script — Customer Service Agent Environment ==========================` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 7`** (2 nodes): `test_rewards_temp.py`, `test_rewards()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 8`** (2 nodes): `test_benchmark.py`, `run_all_tests()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 9`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 10`** (1 nodes): `Convert string JSON or partial JSON to a dictionary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 11`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GeneratedScenario` connect `Community 4` to `Community 0`, `Community 2`, `Community 3`?**
  _High betweenness centrality (0.277) - this node is a cross-community bridge._
- **Why does `ToolContext` connect `Community 2` to `Community 4`?**
  _High betweenness centrality (0.218) - this node is a cross-community bridge._
- **Why does `CustomerServiceEnvironment` connect `Community 3` to `Community 0`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `CustomerServiceAction` (e.g. with `CustomerServiceEnv` and `Client for the Customer Service Agent Environment.      Example:         >>> wit`) actually correct?**
  _`CustomerServiceAction` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `CustomerServiceObservation` (e.g. with `CustomerServiceEnv` and `Client for the Customer Service Agent Environment.      Example:         >>> wit`) actually correct?**
  _`CustomerServiceObservation` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `CustomerServiceState` (e.g. with `CustomerServiceEnv` and `Client for the Customer Service Agent Environment.      Example:         >>> wit`) actually correct?**
  _`CustomerServiceState` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `CustomerServiceEnvironment` (e.g. with `CustomerServiceAction` and `CustomerServiceObservation`) actually correct?**
  _`CustomerServiceEnvironment` has 6 INFERRED edges - model-reasoned connections that need verification._