# Requirements (EARS)

> **Format:** EARS. **IDs:** [REQ-NNN], [PROP-NNN].

## Core Requirements

1. **[REQ-001]** [UBIQUITOUS] The system shall maintain a simulation of N citizen-agents (N ≥ 50), each with endowments (e.g., wealth, productivity), preferences (utility function), and a value vector used in voting and proposals.
2. **[REQ-002]** [STATE-DRIVEN] WHILE the simulation runs, time shall advance in discrete ticks; each tick the Lead shall update agent states (e.g., production, consumption, trades) according to current rules and agent decisions.
3. **[REQ-003]** [EVENT-DRIVEN] WHEN a tick allows proposals, citizen-agents may propose new or amended rules (e.g., tax rate, property rule, voting rule); the Lead shall collect proposals and run a vote according to the current voting rule.
4. **[REQ-004]** [EVENT-DRIVEN] WHEN a vote concludes, the Lead shall update the "constitution" (the set of active rules) with the winning proposal(s) and apply them in subsequent ticks.
5. **[REQ-005]** [UBIQUITOUS] The Observer shall compute, at least every K ticks, aggregate statistics: total output, Gini coefficient, and (optionally) Pareto efficiency checks or coalition composition.
6. **[REQ-006]** [EVENT-DRIVEN] WHEN the simulation ends (max ticks or user stop), the system shall output a constitutional summary (all active rules) and a history log (key events, rule changes, and statistics over time).

## Properties (Invariants)

1. **[PROP-001]** Determinism: For a fixed seed and initial configuration, the simulation outcome (constitution and history) shall be deterministic.
2. **[PROP-002]** Consistency: At every tick, every agent's state (wealth, consumption) shall be consistent with the current rules (e.g., budget constraints, tax deductions).
