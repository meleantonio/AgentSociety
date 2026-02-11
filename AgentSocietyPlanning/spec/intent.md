# Intent: The Emergent Constitution

**Goal:** Simulate a society of 50+ citizen-agents with heterogeneous preferences, endowments, and values that self-organize governance from scratch. Track the emergence of property rights, voting rules, taxation, and coalition dynamics. Measure outcomes (Pareto efficiency, inequality indices such as Gini) and produce a readable "constitutional" and history log.

**Why:** Constitutional design and the emergence of institutions are central to political economy and AI alignment (e.g., value aggregation, collective decision-making). An agent-based simulation where agents negotiate and vote on rules showcases Agent Teams at scale and leverages 1M context to hold full state and history.

**The "Agent Team" Structure:**
- **Lead (Simulation Governor):** Advances time (ticks), enforces turn order, resolves conflicts (e.g., voting), aggregates global state, and triggers Observer.
- **Citizen Agents (Workers):** 50+ agents. Each has endowments (wealth, skills), preferences (utility over consumption, leisure, public goods), and a "value" vector (e.g., equality vs liberty). Each can propose rules, vote, trade, and form coalitions. Behavior is driven by simple micro-rules (e.g., maximize utility given beliefs).
- **Institution Emergence:** Rules that pass voting become part of the "constitution" (property rights, tax rate, voting rule itself, redistribution). Agents propose and vote on changes each tick or when triggered.
- **Observer / Statistician:** Reads full state each tick (or every K ticks). Computes aggregates: total output, Gini coefficient, Pareto dominance checks, coalition sizes. Writes a "history" log and a final constitutional summary.

**Key Technical Challenges:**
- **Scale:** 50+ agents and many ticks require efficient state representation and selective context (e.g., summaries) to stay within 1M context.
- **Consistency:** Voting and rule application must be deterministic and clearly defined so the simulation is interpretable.
- **Emergence vs chaos:** Tuning so that non-trivial institutions emerge (e.g., majority voting, basic property) without the run degenerating into noise.
