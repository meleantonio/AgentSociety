# Development Guide

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Rye (recommended) or pip
- Git

### Setup

Clone the repository:

```bash
git clone https://github.com/meleantonio/AgentSociety.git
cd AgentSociety
```

Install dependencies with Rye (recommended):

```bash
rye sync
```

Or with pip:

```bash
pip install -e ".[dev]"
```

Activate the virtual environment:

```bash
# With Rye
. .venv/bin/activate

# Or source it directly
source .venv/bin/activate
```

## Project Structure

```
AgentSociety/
├── src/emergent_constitution/     # Main package
│   ├── models/                    # Pydantic data models
│   ├── __init__.py                # Public API exports
│   ├── __main__.py                # CLI entry point
│   ├── config.py                  # Configuration
│   ├── lead.py                    # Main simulation loop
│   ├── citizen.py                 # Agent decision logic
│   ├── voting.py                  # Voting mechanics
│   ├── economics.py               # Economic engine
│   ├── observer.py                # Statistics computation
│   ├── coalition.py               # Coalition formation
│   ├── reporter.py                # Report generation
│   ├── initialization.py          # State initialization
│   ├── rng.py                     # Seeded RNG wrapper
│   └── logging.py                 # Logging configuration
├── tests/                         # Test suite (186 tests)
│   ├── test_agent.py
│   ├── test_citizen.py
│   ├── test_coalition.py
│   ├── test_config.py
│   ├── test_constitution.py
│   ├── test_economics.py
│   ├── test_history.py
│   ├── test_initialization.py
│   ├── test_integration.py
│   ├── test_lead.py
│   ├── test_llm_citizen.py
│   ├── test_observer.py
│   ├── test_proposal.py
│   ├── test_reporter.py
│   ├── test_rng.py
│   ├── test_tick.py
│   └── test_voting.py
├── docs/                          # Documentation
├── AgentSocietyPlanning/          # Original specification
├── pyproject.toml                 # Project metadata
└── CLAUDE.md                      # Claude Code guidance
```

## Running Tests

### All Tests

```bash
pytest
```

### With Coverage

```bash
pytest --cov=emergent_constitution --cov-report=term-missing
```

Current coverage: **96%**

### Specific Test Files

```bash
pytest tests/test_lead.py
pytest tests/test_economics.py -v
```

### Excluding Slow Tests

Some integration tests are marked as slow:

```bash
pytest -m "not slow"
```

### Running a Single Test

```bash
pytest tests/test_voting.py::test_tally_votes_majority
```

### Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

### Test Debugging

```bash
# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s
```

## Code Quality

### Formatting with Ruff

Format all code:

```bash
ruff format .
```

Check what would change without modifying:

```bash
ruff format --check .
```

### Linting with Ruff

Run all linters:

```bash
ruff check .
```

Auto-fix issues where possible:

```bash
ruff check --fix .
```

### Configuration

Ruff is configured in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py310"
line-length = 99
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "SIM"]

[tool.ruff.lint.pydocstyle]
convention = "google"
```

### Pre-Commit Workflow

Before committing:

```bash
ruff format .
ruff check --fix .
pytest
```

## Coding Standards

### General Principles

1. **Explicit over implicit**: Use clear names, avoid magic
2. **Immutability**: Functions should not mutate inputs
3. **Type hints**: All functions must have type annotations
4. **Docstrings**: All public functions/classes must have Google-style docstrings
5. **Validation**: Use Pydantic for all data models
6. **Logging**: Use structlog with context
7. **Determinism**: All randomness through `SimulationRNG`

### Docstring Format

Use Google-style docstrings:

```python
def compute_gini(agents: list[AgentState]) -> float:
    """Compute the Gini coefficient of wealth distribution.

    The Gini coefficient measures inequality on a scale from 0 (perfect
    equality) to 1 (perfect inequality). Computed using the standard
    formula based on sorted wealth values.

    Args:
        agents: List of agent states.

    Returns:
        Gini coefficient in [0, 1].

    Raises:
        ValueError: If agents list is empty.

    Example:
        >>> agents = [AgentState(..., wealth=100), AgentState(..., wealth=200)]
        >>> gini = compute_gini(agents)
        >>> 0.0 <= gini <= 1.0
        True
    """
```

### Type Hints

Use modern Python 3.10+ syntax:

```python
# Correct
def process_agents(agents: list[AgentState]) -> dict[str, float]:
    ...

def get_coalition(agent_id: str) -> CoalitionInfo | None:
    ...

# Incorrect (old syntax)
from typing import List, Dict, Optional

def process_agents(agents: List[AgentState]) -> Dict[str, float]:
    ...

def get_coalition(agent_id: str) -> Optional[CoalitionInfo]:
    ...
```

### Immutability Pattern

Never mutate function arguments:

```python
# Correct
def economic_step(agents: list[AgentState], constitution: Constitution) -> list[AgentState]:
    """Apply economic step and return NEW agent states."""
    new_agents = []
    for agent in agents:
        new_agent = agent.model_copy(update={"wealth": compute_new_wealth(agent)})
        new_agents.append(new_agent)
    return new_agents

# Incorrect (mutates input)
def economic_step(agents: list[AgentState], constitution: Constitution) -> list[AgentState]:
    for agent in agents:
        agent.wealth = compute_new_wealth(agent)  # Mutates input!
    return agents
```

### Error Handling

Use structured logging and specific exceptions:

```python
import structlog

log = structlog.get_logger()

def validate_proposal(proposal: Proposal) -> bool:
    """Validate a proposal against constitution schema.

    Args:
        proposal: Proposal to validate.

    Returns:
        True if valid, False otherwise.
    """
    if proposal.rule_key not in CONSTITUTION_FIELDS:
        log.warning(
            "proposal.invalid_rule_key",
            rule_key=proposal.rule_key,
            proposer=proposal.proposer_id,
        )
        return False

    # ... more validation ...
    return True
```

Never silently ignore exceptions:

```python
# Correct
try:
    result = compute_utility(agent, wealth, public_goods)
except (ValueError, OverflowError) as exc:
    log.error("utility.computation_failed", agent_id=agent.id, error=str(exc))
    return 0.0

# Incorrect
try:
    result = compute_utility(agent, wealth, public_goods)
except:  # Too broad, no logging
    pass
```

## Adding New Features

### 1. Adding a New Constitution Rule

Example: Add a "wealth_cap" rule.

**Step 1**: Define the rule in `models/constitution.py`:

```python
class Constitution(BaseModel):
    # ... existing fields ...
    wealth_cap: float | None = Field(default=None, ge=0.0)

# Update the schema registry
CONSTITUTION_FIELDS["wealth_cap"] = float
```

**Step 2**: Implement enforcement in `economics.py`:

```python
def apply_wealth_cap(agents: list[AgentState], constitution: Constitution) -> list[AgentState]:
    """Cap agent wealth at constitutional limit if set."""
    if constitution.wealth_cap is None:
        return agents

    return [
        agent.model_copy(update={"wealth": min(agent.wealth, constitution.wealth_cap)})
        for agent in agents
    ]
```

**Step 3**: Add proposal logic in `citizen.py`:

```python
def decide_proposal(agent: AgentState, constitution: Constitution, rng: SimulationRNG) -> Proposal | None:
    # ... existing logic ...

    # Egalitarian agents propose wealth caps
    if agent.value_vector.equality > 0.7 and rng.random() < 0.1:
        return Proposal(
            rule_key="wealth_cap",
            proposed_value=200.0,
            proposer_id=agent.id,
        )
```

**Step 4**: Add tests:

```python
def test_wealth_cap_enforcement():
    agents = [AgentState(id="a1", wealth=300.0, ...)]
    constitution = Constitution(wealth_cap=200.0)
    capped = apply_wealth_cap(agents, constitution)
    assert capped[0].wealth == 200.0
```

### 2. Adding a New Statistic

Example: Track median productivity.

**Step 1**: Add field to `HistoryEntry` in `models/history.py`:

```python
class HistoryEntry(BaseModel):
    # ... existing fields ...
    median_productivity: float = Field(ge=0.0)
```

**Step 2**: Compute in `observer.py`:

```python
def observe_tick(...) -> HistoryEntry:
    # ... existing computations ...
    median_prod = statistics.median(a.productivity for a in agents)

    return HistoryEntry(
        # ... existing fields ...
        median_productivity=median_prod,
    )
```

**Step 3**: Display in `reporter.py`:

```python
def generate_report(output: SimulationOutput) -> str:
    # ... existing report sections ...
    lines.append(f"Median Productivity: {entry.median_productivity:.2f}")
```

**Step 4**: Add tests:

```python
def test_observe_tick_computes_median_productivity():
    agents = [AgentState(id=f"a{i}", productivity=float(i*10), ...) for i in range(5)]
    entry = observe_tick(...)
    assert entry.median_productivity == 20.0
```

### 3. Adding a Custom Citizen Strategy

Example: Implement LLM-based reasoning.

**Step 1**: Implement the `CitizenLLM` protocol from `llm_citizen.py`:

```python
from emergent_constitution.llm_citizen import CitizenLLM, PromptBuilder

class MyCitizenLLM(CitizenLLM):
    def propose_rule_change(self, context: str) -> Proposal | None:
        # Call your LLM API with context
        response = my_llm_api.generate(context)
        return parse_proposal(response)

    def vote_on_proposal(self, context: str) -> bool:
        response = my_llm_api.generate(context)
        return "yes" in response.lower()
```

**Step 2**: Use it in the simulation:

```python
from emergent_constitution import Lead, SimulationConfig

config = SimulationConfig(use_llm=True, llm_fraction=0.2)
llm = MyCitizenLLM()
lead = Lead(config, citizen_llm=llm)
output = lead.run()
```

## Testing Guidelines

### Test Organization

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test component interactions (mark with `@pytest.mark.slow`)
- **Property tests**: Test invariants (e.g., wealth conservation)

### Fixtures

Use pytest fixtures for common test data:

```python
import pytest

@pytest.fixture
def sample_agents() -> list[AgentState]:
    """Create a standard set of test agents."""
    return [
        AgentState(
            id=f"agent_{i:03d}",
            wealth=100.0,
            productivity=10.0,
            utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
            value_vector=ValueVector(equality=0.5, liberty=0.5),
        )
        for i in range(5)
    ]

def test_something(sample_agents):
    result = my_function(sample_agents)
    assert result > 0
```

### Parametrized Tests

Use `@pytest.mark.parametrize` for multiple test cases:

```python
@pytest.mark.parametrize(
    "votes_for,votes_against,rule,expected",
    [
        (6, 4, VotingRule.MAJORITY, True),       # 60% > 50%
        (5, 5, VotingRule.MAJORITY, False),      # 50% not > 50%
        (7, 3, VotingRule.SUPERMAJORITY, True),  # 70% >= 66.67%
        (6, 4, VotingRule.SUPERMAJORITY, False), # 60% < 66.67%
    ],
)
def test_vote_threshold(votes_for, votes_against, rule, expected):
    # ... test logic ...
```

### Determinism Tests

Always test determinism for randomized functions:

```python
def test_initialization_deterministic():
    config = SimulationConfig(seed=42, num_agents=10)
    state1, rng1 = initialize_simulation(config)
    state2, rng2 = initialize_simulation(config)

    assert state1.agent_states[0].wealth == state2.agent_states[0].wealth
    assert state1.agent_states[0].productivity == state2.agent_states[0].productivity
```

## Debugging

### Logging

Enable detailed logging:

```python
import logging
from emergent_constitution.logging import configure_logging

configure_logging(level=logging.DEBUG)
```

View structured logs:

```python
import structlog

log = structlog.get_logger()

log.info("simulation.tick", tick=42, num_proposals=3, gini=0.35)
# Output: simulation.tick tick=42 num_proposals=3 gini=0.35
```

### Debugging Tests

```bash
# Run test with print statements visible
pytest tests/test_lead.py::test_run_simulation -s

# Drop into pdb on failure
pytest tests/test_lead.py::test_run_simulation --pdb

# Set breakpoint in code
import pdb; pdb.set_trace()
```

### Profiling

Profile slow tests:

```bash
python -m cProfile -o profile.stats -m pytest tests/test_integration.py
python -m pstats profile.stats
# (Pstats) sort cumulative
# (Pstats) stats 10
```

## Git Workflow

### Branch Naming

- `feat/description` - New features
- `fix/description` - Bug fixes
- `refactor/description` - Code refactoring
- `docs/description` - Documentation updates
- `test/description` - Test additions/fixes

### Commit Messages

Use Conventional Commits format:

```
feat: add wealth_cap constitution rule

Allows egalitarian agents to propose maximum wealth limits.
Enforced in economics.py after taxation step.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

### Pull Request Process

1. Create feature branch from `main`
2. Implement feature with tests
3. Ensure all tests pass and coverage remains ≥95%
4. Format and lint: `ruff format . && ruff check --fix .`
5. Commit with clear message
6. Push and create PR
7. Address review feedback
8. Merge after approval

## Requirements Traceability

When adding features, trace them to requirements:

```python
def new_feature():
    """Implement XYZ capability.

    Satisfies REQ-007: The system shall support custom utility functions.
    """
```

Current requirements (see `AgentSocietyPlanning/spec/requirements.md`):
- REQ-001 through REQ-006: Core simulation
- PROP-001: Determinism
- PROP-002: Consistency

## Performance Optimization

### Profiling Hot Paths

```python
import time

start = time.perf_counter()
result = expensive_function()
elapsed = time.perf_counter() - start
log.info("function.timing", function="expensive_function", elapsed_ms=elapsed*1000)
```

### Common Bottlenecks

1. **Pareto computation**: O(N²), disable for N > 200
2. **Gini calculation**: O(N log N), unavoidable but fast
3. **Coalition clustering**: O(N × k × iterations)

### Memory Optimization

For long simulations, consider:

```python
# Clear old history entries
if len(lead.history) > 1000:
    lead.history = lead.history[-100:]  # Keep last 100
```

## Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Update CHANGELOG.md
3. Run full test suite: `pytest --cov`
4. Build: `rye build` or `python -m build`
5. Tag release: `git tag v0.2.0 && git push --tags`
6. Create GitHub release with notes

## Getting Help

- **Issues**: Check existing issues on GitHub
- **Documentation**: See `docs/` and `AgentSocietyPlanning/spec/`
- **Code examples**: Check `tests/` for usage patterns
- **Specifications**: See `AgentSocietyPlanning/spec/*.md`

## Contributing

Contributions are welcome. Please:

1. Open an issue to discuss major changes
2. Follow the coding standards in this guide
3. Add tests for new features (maintain ≥95% coverage)
4. Update documentation as needed
5. Use Conventional Commits format
