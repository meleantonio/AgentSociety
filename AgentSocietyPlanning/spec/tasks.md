# Implementation Plan: The Emergent Constitution

> **Rules:** Two-level max (Task > Subtask). Sequential. Traceability to REQ IDs.

## Phase 1: Core simulation loop and state

- [ ] **Task 1: Data models and initial state**
    - [ ] 1.1: Define AgentState, Constitution, Proposal, VoteOutcome, TickState, HistoryEntry in code (e.g., Python dataclasses or Pydantic).
    - [ ] 1.2: Implement initialization: generate N agents with endowments and preferences (from config or default), initial constitution (e.g., no tax, majority vote).
    - *Traceability:* Implements REQ-001, PROP-002.

- [ ] **Task 2: Lead tick loop (economic step only)**
    - [ ] 2.1: Implement one tick: apply production (e.g., output = productivity), apply tax and redistribution from Constitution, update wealth.
    - [ ] 2.2: Lead advances tick counter and stores TickState. No proposals/votes yet.
    - *Traceability:* Implements REQ-002.

## Phase 2: Proposals and voting

- [ ] **Task 3: Proposal collection and voting**
    - [ ] 3.1: In each tick (or every K ticks), Lead allows each citizen to submit at most one Proposal; collect and validate against schema.
    - [ ] 3.2: Run voting: each citizen returns yes/no per proposal; aggregate by current voting_rule (e.g., majority); update Constitution with passed proposals.
    - *Traceability:* Implements REQ-003, REQ-004.

- [ ] **Task 4: Citizen decision logic**
    - [ ] 4.1: Implement citizen logic (can be rule-based or LLM): given TickState and own state, output Proposal (optional) and votes. Ensure determinism (same state => same decisions) when using seed.
    - [ ] 4.2: Optional: add simple trade (bilateral exchange) so state includes trades; apply budget constraints.
    - *Traceability:* Implements REQ-003, PROP-001.

## Phase 3: Observer and output

- [ ] **Task 5: Observer and history**
    - [ ] 5.1: Implement Observer: given TickState, compute Gini, total output; optionally Pareto check; append HistoryEntry.
    - [ ] 5.2: Lead writes history to log (in-memory or file). Simulation ends at max_ticks or user stop.
    - *Traceability:* Implements REQ-005, REQ-006.

- [ ] **Task 6: Constitutional summary and report**
    - [ ] 6.1: On end, output final Constitution and full history (or summarized). Optionally produce a human-readable "constitutional summary" document.
    - *Traceability:* Implements REQ-006.
