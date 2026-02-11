# Design Specification: The Emergent Constitution

## Architecture

```
Initial config (N agents, endowments, preferences)
       |
       v
+------------------+
| Lead (Governor)  |  tick loop: update state, collect proposals, run votes, apply rules
+------------------+
       |
       +---> Citizen agents (50+): propose, vote, trade (decisions per tick)
       |
       +---> Constitution (mutable): property_rule, tax_rate, voting_rule, ...
       |
       v (every K ticks)
+------------------+
| Observer         |  compute Gini, output, Pareto; append to history
+------------------+
       |
       v
   Output: constitution snapshot + history log
```

- **Execution:** Can be single process (Lead runs a loop, calls agent logic for each citizen in turn or in batches) or multi-agent (each citizen is an agent that returns actions). For scale, a hybrid is likely: citizen logic in code, with LLM agents only for "proposal generation" or "vote reasoning" for a subset of ticks to save API cost.
- **State:** One shared state object: agent states, constitution, history log. Optionally compress old history into summaries to fit 1M context if full log is used by agents.

## Data Models

- **AgentState:** `{ id: string, wealth: number, productivity: number, utility_params: object, value_vector: number[], coalition_id?: string }`
- **Constitution:** `{ property_rule: string, tax_rate: number, voting_rule: "majority"|"supermajority"|..., redistribution_rule?: string }`
- **Proposal:** `{ rule_key: string, proposed_value: any, proposer_id: string }`
- **VoteOutcome:** `{ proposal: Proposal, passed: boolean, votes_for: number }`
- **TickState:** `{ tick: number, agent_states: AgentState[], constitution: Constitution, proposals_this_tick: Proposal[], votes: VoteOutcome[] }`
- **HistoryEntry:** `{ tick: number, gini: number, total_output: number, rule_changes?: string[] }`
- **Output:** `{ constitution: Constitution, history: HistoryEntry[], final_agent_states: AgentState[] }`

## Component Interfaces

- **Lead:** Input: initial config (N, endowments, preference schema), max_ticks, seed. Loop: (1) request actions from citizens (propose/vote/trade), (2) resolve votes and update constitution, (3) apply economic step (production, tax, redistribution), (4) call Observer. Output: final constitution and history.
- **Citizen (per agent):** Input: current TickState (or summary), own state. Output: Proposal (optional), vote (yes/no per proposal), trade offers (optional).
- **Observer:** Input: TickState. Output: HistoryEntry for this tick (or batch). Side effect: append to history log.

## Error Handling

- **Tie in vote:** Use deterministic tie-break (e.g., status quo wins, or random with fixed seed).
- **Invalid proposal:** Lead rejects proposals that do not match the rule schema (e.g., tax_rate must be in [0,1]); no update.
- **Numerical instability:** If wealth or utility goes NaN/Inf, cap or abort and report.

## Security Considerations

- Simulation runs in sandbox. No network or file access from citizen logic except via Lead-provided state.
- User-provided initial config (e.g., preference parameters) must be validated; no code injection in config.
