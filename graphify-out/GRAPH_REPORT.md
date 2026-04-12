# Graph Report - .  (2026-04-12)

## Corpus Check
- 11 files · ~8,840 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 91 nodes · 146 edges · 11 communities detected
- Extraction: 60% EXTRACTED · 40% INFERRED · 0% AMBIGUOUS · INFERRED: 58 edges (avg confidence: 0.5)
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

## God Nodes (most connected - your core abstractions)
1. `CustomerServiceAction` - 19 edges
2. `CustomerServiceObservation` - 19 edges
3. `CustomerServiceState` - 18 edges
4. `CustomerServiceEnvironment` - 14 edges
5. `Scenario` - 13 edges
6. `CustomerServiceEnv` - 8 edges
7. `run_scenario()` - 8 edges
8. `Map any raw step reward to a value strictly inside (0.01, 0.99).      Per OpenEn` - 5 edges
9. `Customer Service Agent Environment.      The agent receives a customer query and` - 5 edges
10. `Initialize the environment.` - 5 edges

## Surprising Connections (you probably didn't know these)
- `CustomerServiceEnv` --uses--> `CustomerServiceAction`  [INFERRED]
  client.py → models.py
- `CustomerServiceEnv` --uses--> `CustomerServiceObservation`  [INFERRED]
  client.py → models.py
- `Client for the Customer Service Agent Environment.      Example:         >>> wit` --uses--> `CustomerServiceAction`  [INFERRED]
  client.py → models.py
- `Client for the Customer Service Agent Environment.      Example:         >>> wit` --uses--> `CustomerServiceObservation`  [INFERRED]
  client.py → models.py
- `Convert action to JSON payload.` --uses--> `CustomerServiceAction`  [INFERRED]
  client.py → models.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.17
Nodes (15): CustomerServiceEnvironment, Reset the environment with a specific scenario.          Args:             seed:, Execute one step in the environment.          Args:             action: The agen, Get the current environment state., Compute reward for a tool call based on scenario requirements., Compute reward for agent's message based on resolution keywords and politeness., Check if the episode should end., Customer Service Agent Environment.      The agent receives a customer query and (+7 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (17): _call_llm_sync(), _extract_error(), get_agent_action(), log_end(), log_start(), log_step(), main(), Inference Script — Customer Service Agent Environment ========================== (+9 more)

### Community 2 - "Community 2"
Cohesion: 0.13
Nodes (14): call_tool(), check_order(), check_policy(), escalate_to_human(), issue_refund(), Verify a user's identity and account status., Look up an order's details and status., Process a refund for an order. (+6 more)

### Community 3 - "Community 3"
Cohesion: 0.17
Nodes (7): Action, main(), Run the server directly., Map any raw step reward to a value strictly inside (0.01, 0.99).      Per OpenEn, safe_reward(), CustomerServiceAction, Action the agent takes each step.      The agent can:     - Call a tool by setti

### Community 4 - "Community 4"
Cohesion: 0.23
Nodes (8): CustomerServiceEnv, Client for the Customer Service Agent Environment.      Example:         >>> wit, Convert action to JSON payload., Parse server response into StepResult., Parse server response into State object., CustomerServiceState, Internal state tracking for the customer service environment., State

### Community 5 - "Community 5"
Cohesion: 0.4
Nodes (4): get_scenario(), list_scenarios(), Get a scenario by ID., List all available scenarios with metadata.

### Community 6 - "Community 6"
Cohesion: 1.0
Nodes (0): 

### Community 7 - "Community 7"
Cohesion: 1.0
Nodes (0): 

### Community 8 - "Community 8"
Cohesion: 1.0
Nodes (0): 

### Community 9 - "Community 9"
Cohesion: 1.0
Nodes (1): Convert string JSON or partial JSON to a dictionary.

### Community 10 - "Community 10"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **23 isolated node(s):** `Action the agent takes each step.      The agent can:     - Call a tool by setti`, `Convert string JSON or partial JSON to a dictionary.`, `Observation the agent receives after each step.      Contains the customer query`, `Internal state tracking for the customer service environment.`, `Inference Script — Customer Service Agent Environment ==========================` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 6`** (2 nodes): `test_rewards_temp.py`, `test_rewards()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 7`** (2 nodes): `test_benchmark.py`, `run_all_tests()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 8`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 9`** (1 nodes): `Convert string JSON or partial JSON to a dictionary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 10`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Scenario` connect `Community 0` to `Community 3`, `Community 5`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `CustomerServiceEnvironment` connect `Community 0` to `Community 3`, `Community 4`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `CustomerServiceAction` connect `Community 3` to `Community 0`, `Community 4`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `CustomerServiceAction` (e.g. with `CustomerServiceEnv` and `Client for the Customer Service Agent Environment.      Example:         >>> wit`) actually correct?**
  _`CustomerServiceAction` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `CustomerServiceObservation` (e.g. with `CustomerServiceEnv` and `Client for the Customer Service Agent Environment.      Example:         >>> wit`) actually correct?**
  _`CustomerServiceObservation` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `CustomerServiceState` (e.g. with `CustomerServiceEnv` and `Client for the Customer Service Agent Environment.      Example:         >>> wit`) actually correct?**
  _`CustomerServiceState` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `CustomerServiceEnvironment` (e.g. with `CustomerServiceAction` and `CustomerServiceObservation`) actually correct?**
  _`CustomerServiceEnvironment` has 5 INFERRED edges - model-reasoned connections that need verification._