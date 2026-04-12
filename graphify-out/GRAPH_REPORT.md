# Graph Report - .  (2026-04-12)

## Corpus Check
- 16 files · ~13,348 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 143 nodes · 233 edges · 15 communities detected
- Extraction: 70% EXTRACTED · 30% INFERRED · 0% AMBIGUOUS · INFERRED: 70 edges (avg confidence: 0.5)
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
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]

## God Nodes (most connected - your core abstractions)
1. `CustomerServiceAction` - 16 edges
2. `CustomerServiceObservation` - 16 edges
3. `CustomerServiceState` - 14 edges
4. `CustomerServiceEnvironment` - 14 edges
5. `ToolContext` - 11 edges
6. `RewardEngine` - 11 edges
7. `GeneratedScenario` - 11 edges
8. `BaseRubric` - 9 edges
9. `CustomerServiceRubric` - 9 edges
10. `CustomerServiceEnv` - 8 edges

## Surprising Connections (you probably didn't know these)
- `CustomerServiceEnvironment` --uses--> `CustomerServiceAction`  [INFERRED]
  server/customer_service_env_environment.py → models.py
- `Map any raw reward to strictly inside (0.01, 0.99) using linear scaling.      Ma` --uses--> `CustomerServiceAction`  [INFERRED]
  server/customer_service_env_environment.py → models.py
- `Customer Service Agent Environment.      The agent receives a customer query and` --uses--> `CustomerServiceAction`  [INFERRED]
  server/customer_service_env_environment.py → models.py
- `Initialize the environment.` --uses--> `CustomerServiceAction`  [INFERRED]
  server/customer_service_env_environment.py → models.py
- `Reset the environment with a generative scenario.` --uses--> `CustomerServiceAction`  [INFERRED]
  server/customer_service_env_environment.py → models.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.13
Nodes (18): Action, health(), main(), Liveness probe — returns 200 OK when the server is ready., Run the server directly., CustomerServiceEnv, Client for the Customer Service Agent Environment.      Example:         >>> wit, Convert action to JSON payload. (+10 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (16): BaseRubric, EscalationRubric, for_scenario(), MessageQualityRubric, OrderStatusRubric, Full credit if human escalation happened (state.escalated = True).      Used by, Full credit if the agent checked the correct order.      Verifies the scenario's, Score how closely the agent followed the required tool sequence.      Unlike a b (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.15
Nodes (17): _call_llm_sync(), _extract_error(), get_agent_action(), log_end(), log_start(), log_step(), main(), Inference Script — Customer Service Agent Environment ========================== (+9 more)

### Community 3 - "Community 3"
Cohesion: 0.15
Nodes (16): Session-local snapshot of databases for one episode.      Created fresh per epis, ToolContext, call_tool(), check_order(), check_policy(), escalate_to_human(), issue_refund(), Verify a user's identity and account status. (+8 more)

### Community 4 - "Community 4"
Cohesion: 0.21
Nodes (11): CustomerServiceEnvironment, Reset the environment with a generative scenario., Execute one step with outcome-based grading., Map any raw reward to strictly inside (0.01, 0.99) using linear scaling.      Ma, Customer Service Agent Environment.      The agent receives a customer query and, Initialize the environment., safe_reward(), Environment (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.19
Nodes (8): Protocol, EpisodeStateProtocol, Shaped intermediate reward per step.          Provides a small positive signal f, Outcome-based terminal reward using CustomerServiceRubric.          The rubric v, ScenarioProtocol, CustomerServiceRubric, Composite rubric that selects the right outcome + process + message rubrics, Compute the weighted composite score in [0.0, 1.0].

### Community 6 - "Community 6"
Cohesion: 0.67
Nodes (8): fail(), main(), ok(), section(), tier_http(), tier_openenv(), tier_oracle(), tier_static()

### Community 7 - "Community 7"
Cohesion: 0.25
Nodes (5): ScenarioGenerator, get_scenario(), list_scenarios(), Get a procedurally generated scenario by ID or legacy alias.      Each call with, List all available scenario IDs with metadata.

### Community 8 - "Community 8"
Cohesion: 1.0
Nodes (0): 

### Community 9 - "Community 9"
Cohesion: 1.0
Nodes (0): 

### Community 10 - "Community 10"
Cohesion: 1.0
Nodes (0): 

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (1): Convert string JSON or partial JSON to a dictionary.

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (1): Ultimate defense-in-depth against OOB task scores.                  The OpenEnv

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (1): Factory: return the right rubric mix for a given scenario type.

## Knowledge Gaps
- **26 isolated node(s):** `Action the agent takes each step.      The agent can:     - Call a tool by setti`, `Convert string JSON or partial JSON to a dictionary.`, `Observation the agent receives after each step.      Contains the customer query`, `Ultimate defense-in-depth against OOB task scores.                  The OpenEnv`, `Internal state tracking for the customer service environment.` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 8`** (2 nodes): `test_rewards_temp.py`, `test_rewards()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 9`** (2 nodes): `test_benchmark.py`, `run_all_tests()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 10`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 11`** (1 nodes): `Convert string JSON or partial JSON to a dictionary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (1 nodes): `Ultimate defense-in-depth against OOB task scores.                  The OpenEnv`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `Factory: return the right rubric mix for a given scenario type.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RewardEngine` connect `Community 4` to `Community 5`?**
  _High betweenness centrality (0.264) - this node is a cross-community bridge._
- **Why does `CustomerServiceRubric` connect `Community 5` to `Community 1`, `Community 4`?**
  _High betweenness centrality (0.240) - this node is a cross-community bridge._
- **Why does `GeneratedScenario` connect `Community 4` to `Community 3`, `Community 7`?**
  _High betweenness centrality (0.221) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `CustomerServiceAction` (e.g. with `CustomerServiceEnv` and `Client for the Customer Service Agent Environment.      Example:         >>> wit`) actually correct?**
  _`CustomerServiceAction` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `CustomerServiceObservation` (e.g. with `CustomerServiceEnv` and `Client for the Customer Service Agent Environment.      Example:         >>> wit`) actually correct?**
  _`CustomerServiceObservation` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `CustomerServiceState` (e.g. with `CustomerServiceEnv` and `Client for the Customer Service Agent Environment.      Example:         >>> wit`) actually correct?**
  _`CustomerServiceState` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `CustomerServiceEnvironment` (e.g. with `CustomerServiceAction` and `CustomerServiceObservation`) actually correct?**
  _`CustomerServiceEnvironment` has 7 INFERRED edges - model-reasoned connections that need verification._