# Coding Standards: The Emergent Constitution

## General

- **Language:** Python. Use dataclasses or Pydantic for state and config.
- **Orchestration:** Single process with a clear tick loop; citizen logic can be pure functions or, for a subset of decisions, LLM-based agents with deterministic decoding (e.g., temperature=0).

## Patterns

- **Determinism:** Use a single RNG seed for all randomness (proposal order, tie-breaks, any stochastic citizen behavior). Store seed in config and log it in output (PROP-001).
- **State immutability per tick:** When passing state to citizens, pass a copy or read-only view so citizen code cannot mutate global state except through Lead-collected actions.
- **Constitution schema:** Define valid keys and value types (e.g., tax_rate in [0,1]); reject invalid proposals without crashing.

## Examples

- **Good (Observer):** Gini computed from current wealth list; total_output = sum of agent outputs. Append one HistoryEntry per tick.
- **Bad:** Observer that mutates agent state or constitution.
- **Good (Vote):** "majority" => passed iff votes_for > N/2; "supermajority" => votes_for >= 2*N/3. Document in Constitution.
- **Bad:** Undefined voting rule or non-deterministic tie-break.
