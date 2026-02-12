# Intent: The Emergent Constitution v2 — DSGE-HA with LLM Agents

## Vision

Redesign the simulation from an agent-based model (ABM) with ad-hoc behavioral rules into a **Dynamic Stochastic General Equilibrium model with Heterogeneous Agents (DSGE-HA)**, where citizen behavior is **microfounded** (grounded in explicit utility maximization) and **LLM-driven** (Large Language Models serve as the default decision-making engine for all aspects of agent life, not just voting).

The result is a hybrid at the frontier of computational economics: the structural rigor of DSGE-HA (budget constraints, market clearing, welfare-grounded evaluation) combined with the emergent institutional creativity of LLM agents (agents that can invent new rules, create firms, form complex organizations, and reason about the society they inhabit).

## Why This Redesign

The v1 simulation has several limitations that this redesign addresses:

1. **No microfoundation.** Agent behavior (proposals, votes, trade decisions) follows hand-coded heuristics, not utility maximization. This means welfare comparisons (Pareto efficiency, social welfare) lack theoretical grounding. In a DSGE-HA model, every agent action traces back to an explicit utility function and budget constraint, making welfare analysis rigorous.

2. **Limited economic structure.** v1 has a simple production-tax-redistribution loop. Real economies have capital accumulation, labor markets, firm creation, interest rate determination, and savings decisions. A DSGE-HA framework provides all of this.

3. **LLM as afterthought.** v1 treats LLM reasoning as an optional add-on for proposals/votes only. In v2, LLM citizens are the default and make all decisions: consumption vs. savings, labor supply, firm creation, hiring, R&D investment, rule proposals, voting, coalition formation, and institutional design.

4. **Rigid institutional design.** v1 constitutions are limited to four fixed parameters (property_rule, tax_rate, voting_rule, redistribution_rule). In v2, LLM citizens can propose arbitrary institutional mechanisms — new market designs, novel taxation schemes, regulatory frameworks, collective organizations — limited only by what the society can collectively agree on.

5. **No dynamics.** v1 lacks intertemporal optimization. Agents do not save, invest, or plan for the future. DSGE-HA models are inherently dynamic: agents choose consumption-savings paths to maximize lifetime utility, creating rich wealth dynamics and demand for social insurance.

## Core Framework: DSGE-HA

The model follows the Bewley-Huggett-Aiyagari tradition with extensions:

### Households (Citizens)

Each citizen-agent maximizes expected discounted lifetime utility:

```
max E_0 [ sum_{t=0}^{inf} beta^t * u(c_t, l_t, G_t) ]
```

subject to:
- **Budget constraint:** `a_{t+1} = (1 + r_t) * a_t + w_t * z_t * (1 - l_t) - c_t - T(y_t) + Tr_t`
- **Borrowing constraint:** `a_t >= a_min` (e.g., natural borrowing limit or zero)
- **Non-negativity:** `c_t >= 0`, `0 <= l_t <= 1`

where:
- `a_t` = wealth (assets)
- `c_t` = consumption
- `l_t` = leisure (1 - labor supply)
- `G_t` = public goods per capita
- `z_t` = idiosyncratic productivity (stochastic, Markov chain)
- `r_t` = interest rate (endogenous)
- `w_t` = wage rate (endogenous)
- `T(y_t)` = taxes (determined by the constitution)
- `Tr_t` = transfers (determined by the constitution)
- `beta` = discount factor (heterogeneous across agents)
- `u(.)` = Cobb-Douglas or CES utility

**Key departure from standard DSGE-HA:** Instead of solving the Bellman equation numerically, LLM agents make these decisions through reasoning — consuming, saving, working, and investing based on their understanding of economic trade-offs, their personal preferences (utility parameters), and their assessment of the economic environment. An explicit utility evaluator scores LLM decisions against the true utility function, providing a ground-truth welfare measure.

### Firms

Agents can create **firms** that:
- Hire labor from other citizens at the market wage
- Rent capital from savers at the market interest rate
- Produce output using a production function: `Y = A * K^alpha * L^(1-alpha)`
- Earn profits distributed to the firm owner
- Can invest in R&D to improve their firm-specific TFP (total factor productivity)

Firm creation, management, and R&D decisions are made by LLM agents. Some citizens will be entrepreneurs, others workers, others researchers — roles emerge from agent decisions, not exogenous assignment.

### Markets

Markets clear endogenously each period:
- **Labor market:** Aggregate labor supply (from households) = aggregate labor demand (from firms). Wage `w_t` clears the market.
- **Capital market:** Aggregate savings (household assets) = aggregate capital demand (from firms). Interest rate `r_t` clears the market.
- **Goods market:** `Y_t = C_t + I_t + G_t` (aggregate resource constraint).

### Stochastic Elements

- **Idiosyncratic productivity shocks:** Each agent's productivity `z_t` follows a Markov chain (Rouwenhorst discretization of an AR(1) process). This creates demand for social insurance.
- **Aggregate TFP shocks:** Economy-wide productivity shifts, creating business cycles.
- **Preference shocks:** Random shifts in discount factors or utility weights.
- **Policy uncertainty:** Constitutional changes create endogenous policy risk.

### Government / Constitution

The "government" is the constitution itself — a set of rules collectively chosen by citizens. Unlike v1's four fixed parameters, the constitution is an **open-ended document** that LLM citizens can extend with:
- Tax schedules (flat, progressive, consumption, capital, Pigouvian, etc.)
- Transfer programs (universal, means-tested, unemployment insurance, etc.)
- Public goods provision rules
- Market regulations (minimum wage, price controls, antitrust, etc.)
- Voting procedures (majority, supermajority, delegation, quadratic voting, etc.)
- Property rights regimes
- Firm regulations (licensing, labor protections, environmental rules, etc.)
- Any novel mechanism the agents can collectively agree to implement

Constitutional proposals and voting use LLM reasoning. The constitution is stored as structured data with a schema that can grow over time.

## LLM Agents as Default

LLM citizens are the primary decision-making mechanism, not an optional add-on:

- **Economic decisions:** consumption, savings, labor supply, firm creation, hiring, capital investment, R&D spending
- **Political decisions:** rule proposals, voting, constitutional amendments, coalition building, lobbying, public debate
- **Social decisions:** coalition formation, cooperation, reputation, trust, communication with other agents
- **Strategic decisions:** long-term planning, risk management, career choices, specialization

Each LLM agent receives:
1. Their personal state (wealth, productivity, utility parameters, value vector)
2. The current economic conditions (prices, GDP, unemployment, distribution stats)
3. The current constitution (all active rules)
4. Recent history (past events, trends, constitutional changes)
5. Messages from other agents (debates, proposals, negotiations)

The LLM responds with structured decisions that are validated against budget constraints and physical feasibility. A numerical utility evaluator provides the ground-truth welfare measure for each agent's realized outcomes.

**Fallback:** A numerical solver (value function iteration or policy function approximation) serves as the fallback for agents when LLM calls fail or for benchmarking LLM behavior against optimal behavior.

## What Emerges

With microfounded LLM agents in a DSGE-HA framework, we expect to observe:
- **Endogenous wealth distribution** shaped by savings behavior, productivity shocks, and institutional choices
- **Endogenous firm formation** with varying sizes, sectors, and productivity levels
- **Labor market dynamics** — wages, employment, specialization, and potentially unemployment
- **Institutional innovation** — agents inventing taxation, insurance, regulation, and governance mechanisms
- **Business cycles** driven by aggregate shocks and amplified by heterogeneous responses
- **Political economy dynamics** — coalitions forming around economic interests, voting on redistribution
- **Precautionary behavior** — agents near the borrowing constraint behaving differently from wealthy agents
- **Research and growth** — agents investing in R&D, discovering productivity improvements

## Key Technical Challenges

- **LLM cost and latency:** With 50+ agents making multiple decisions per tick, LLM calls dominate runtime. Requires batching, caching, and selective LLM use.
- **Budget constraint enforcement:** LLM decisions must respect budget constraints. Need a validation/correction layer.
- **Market clearing:** Must find prices (w, r) that clear all markets each period. Iterative price adjustment or tatonnement.
- **Open-ended constitution:** Storing, interpreting, and enforcing arbitrarily complex rules proposed by LLMs.
- **Evaluation:** Comparing LLM agent behavior against optimal (Bellman equation) benchmarks.
- **Scale:** Balancing agent count, tick count, and LLM call budget.
- **Determinism:** LLM outputs must be made deterministic (temperature=0, fixed seeds) for reproducibility.

## Relationship to Literature

This project sits at the intersection of:
- **Bewley-Huggett-Aiyagari** incomplete markets models (households facing idiosyncratic risk with borrowing constraints)
- **HANK** (Heterogeneous Agent New Keynesian) models (Kaplan-Moll-Violante 2018)
- **Political economy DSGE** (Krusell-Rios-Rull 1999, Meltzer-Richard)
- **LLM-as-economic-agent** (Horton 2023, Li et al. 2024 EconAgent, Park et al. 2023 Generative Agents)
- **AI Economist** (Zheng-Trott et al. 2022, Salesforce Research)
- **Constitutional economics** (Buchanan-Tullock 1962, Barbera-Jackson 2004)

## Success Criteria

1. Agents make economically coherent decisions (budget constraints satisfied, positive consumption, non-negative wealth)
2. Markets clear each period (supply = demand for labor and capital)
3. The wealth distribution evolves endogenously and responds to policy
4. Agents create firms, hire workers, and invest in R&D
5. Novel institutional mechanisms emerge from LLM deliberation
6. Welfare can be rigorously measured using realized utility
7. The simulation produces qualitatively realistic macroeconomic dynamics (business cycles, inequality, policy responses)
8. LLM agent behavior can be benchmarked against numerical optima
