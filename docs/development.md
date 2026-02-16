# Development Guide

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Rye (recommended) or pip
- Git
- Anthropic API key (for LLM mode; not needed for benchmark/testing)

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

### Verify Installation

```bash
# Run tests
pytest

# Run a quick benchmark simulation (no API key needed)
emergent-constitution-v2 --benchmark -n 20 -t 10 -q
```

## Project Structure

```
AgentSociety/
├── src/emergent_constitution/     # Main package
│   ├── models/                    # Pydantic data models
│   │   ├── agent.py               #   v1 AgentState
│   │   ├── household.py           #   v2 HouseholdState (with liquid/illiquid fields)
│   │   ├── firm.py                #   v2 FirmState
│   │   ├── market.py              #   v2 MarketState
│   │   ├── shocks.py              #   v2 ShockState
│   │   ├── decisions.py           #   v2 EconomicDecision, EntrepreneurialDecision, PoliticalDecision
│   │   ├── constitution.py        #   v1 Constitution + v2 ConstitutionV2
│   │   ├── proposal.py            #   v1 Proposal + v2 ConstitutionalProposal
│   │   ├── tick.py                #   v1 TickState
│   │   └── history.py             #   v1 + v2 HistoryEntry, SimulationOutput
│   ├── __init__.py                # Public API exports (v1)
│   ├── __main__.py                # CLI entry points (v1 + v2)
│   ├── config.py                  # SimulationConfig (v1) + SimulationConfigV2 (~50 fields)
│   ├── lead.py                    # Lead (v1) + LeadV2 (9-step lifecycle with HANK)
│   ├── citizen.py                 # Rule-based agent decision logic (v1)
│   ├── citizen_v2.py              # Benchmark rule-based logic for v2
│   ├── llm_citizen.py             # LLM citizen interface (v1)
│   ├── llm_engine.py              # LLM decision engine (v2): batching, caching, fallback
│   ├── llm_providers.py           # LLM provider abstraction: Anthropic, Mock
│   ├── numerical_solver.py        # VFI benchmark solver (v2)
│   ├── egm_solver.py              # EGM solver with Fella upper envelope (HANK v3)
│   ├── entrepreneurial_solver.py  # Firm value and occupational choice
│   ├── distribution.py            # KFE distribution tracking (HANK v3)
│   ├── calibration.py             # SMM calibration (HANK v3)
│   ├── government.py              # Government debt, bonds, fiscal rule (HANK v3)
│   ├── nominal.py                 # New Keynesian nominal block (HANK v3)
│   ├── political_utility.py       # Microfounded political preferences (HANK v3)
│   ├── constitution_engine.py     # Rule enforcement with AST sandbox (v2)
│   ├── market_clearing.py         # Analytical/Walrasian market clearing (v2)
│   ├── shock_generators.py        # Rouwenhorst discretization + shock drawing (v2)
│   ├── economics.py               # Production, budget, utility (v1 + v2)
│   ├── voting.py                  # Proposal validation and tallying (v1 + v2)
│   ├── coalition.py               # Coalition formation (v1 + v2)
│   ├── initialization.py          # State initialization (v1 + v2)
│   ├── observer.py                # Statistics (v1) + ObserverV2 (20+ stats)
│   ├── reporter.py                # Report generation (v1 + v2)
│   ├── rng.py                     # Seeded RNG wrapper (PROP-001)
│   └── logging.py                 # structlog configuration
├── tests/                         # 50 test modules, 1412 tests
├── docs/                          # Documentation
├── AgentSocietyPlanning/          # Specification documents
│   ├── spec/                      #   requirements, design, tasks
│   └── steering/                  #   coding standards
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

### v2-Specific Tests

```bash
# v2 lead lifecycle
pytest tests/test_lead_v2.py -v

# v2 data models
pytest tests/test_v2_models.py -v

# v2 config
pytest tests/test_v2_config.py -v

# Property-based invariant tests (PROP-001 through PROP-006)
pytest tests/test_properties.py -v

# Market clearing
pytest tests/test_market_clearing.py -v

# LLM engine and providers
pytest tests/test_llm_engine.py tests/test_llm_providers.py -v
```

### HANK v3 Tests

```bash
# HANK integration tests
pytest tests/test_hank_integration.py -v

# EGM solver
pytest tests/test_egm_solver.py -v

# Distribution (KFE)
pytest tests/test_distribution.py -v

# Government sector
pytest tests/test_government.py -v

# Nominal block
pytest tests/test_nominal.py -v

# Political utility
pytest tests/test_political_utility.py -v

# Calibration
pytest tests/test_calibration.py -v

# Walrasian clearing
pytest tests/test_walrasian_clearing.py -v
```

### Excluding Slow Tests

Some integration and scale tests are marked as slow:

```bash
pytest -m "not slow"
```

### Running a Single Test

```bash
pytest tests/test_voting_v2.py::TestTallyVotesV2::test_majority_passes -v
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

```bash
ruff format .
```

### Linting with Ruff

```bash
ruff check .
ruff check --fix .  # Auto-fix where possible
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
2. **Immutability**: Functions should not mutate inputs — return new objects via `model_copy(update={...})`
3. **Type hints**: All functions must have type annotations (Python 3.10+ syntax)
4. **Docstrings**: All public functions/classes must have Google-style docstrings
5. **Validation**: Use Pydantic for all data models
6. **Logging**: Use structlog with context variables
7. **Determinism**: All randomness through `SimulationRNG` (PROP-001)
8. **Sandboxing**: Constitution enforcement code runs in AST sandbox (PROP-006)

### Docstring Format

Use Google-style docstrings:

```python
def compute_budget(
    agent: HouseholdState,
    wage: float,
    interest_rate: float,
    tax: float,
    transfer: float,
) -> float:
    """Compute the household budget constraint.

    Budget = (1+r)*a + w*z*(1-l) - tax + transfer

    Args:
        agent: Household state with wealth and productivity.
        wage: Market clearing wage w_t.
        interest_rate: Market clearing rate r_t.
        tax: Taxes owed this period.
        transfer: Transfers received this period.

    Returns:
        Available budget for consumption and savings.

    Traceability: REQ-004, PROP-002
    """
```

### Type Hints

Use modern Python 3.10+ syntax:

```python
# Correct
def process(households: list[HouseholdState]) -> dict[str, EconomicDecision]:
    ...

def get_firm(firm_id: str) -> FirmState | None:
    ...

# Incorrect (old syntax)
from typing import List, Dict, Optional
def process(households: List[HouseholdState]) -> Dict[str, EconomicDecision]:
    ...
```

### Immutability Pattern

Never mutate function arguments:

```python
# Correct — return new objects
def update_wealth(households: list[HouseholdState], income: dict[str, float]) -> list[HouseholdState]:
    return [
        h.model_copy(update={"wealth": h.wealth + income.get(h.id, 0.0)})
        for h in households
    ]

# Incorrect — mutates input
def update_wealth(households: list[HouseholdState], income: dict[str, float]) -> list[HouseholdState]:
    for h in households:
        h.wealth += income.get(h.id, 0.0)  # Mutates input!
    return households
```

### Error Handling

Use structured logging and specific exceptions:

```python
import structlog

log = structlog.get_logger()

def enforce_taxes(households: list[HouseholdState], constitution: ConstitutionV2) -> tuple[list[HouseholdState], float]:
    tax_rules = constitution.get_tax_rules()
    if not tax_rules:
        log.debug("enforce_taxes.no_rules")
        return households, 0.0

    try:
        # ... enforcement logic ...
    except ConstitutionEngineError as exc:
        log.error("enforce_taxes.failed", error=str(exc))
        return households, 0.0  # Safe fallback
```

## Adding New Features

### 1. Adding a New Constitutional Rule Type

v2 uses extensible rules. To add a new rule category:

**Step 1**: Add to `RuleType` enum in `models/constitution.py`:

```python
class RuleType(StrEnum):
    # ... existing ...
    ENVIRONMENTAL = "environmental"  # New rule type
```

**Step 2**: Add enforcement in `constitution_engine.py`:

```python
def enforce_environmental(
    self, firms: list[FirmState], constitution: ConstitutionV2
) -> list[FirmState]:
    env_rules = [r for r in constitution.rules.values() if r.rule_type == RuleType.ENVIRONMENTAL]
    # ... enforcement logic ...
```

**Step 3**: Wire into LeadV2 step 6 in `lead.py`:

```python
def _enforce_constitution(self, ...):
    # ... existing enforcement ...
    firms = self.constitution_engine.enforce_environmental(firms, constitution)
```

**Step 4**: Add tests in a new `tests/test_environmental_rules.py`.

### 2. Adding a New Statistic to ObserverV2

**Step 1**: Add field to `HistoryEntryV2` in `models/history.py`:

```python
class HistoryEntryV2(BaseModel):
    # ... existing fields ...
    income_mobility: float = 0.0  # New statistic
```

**Step 2**: Compute in `observer.py` `ObserverV2.observe()`:

```python
def observe(self, period_state: PeriodState, ...) -> HistoryEntryV2:
    # ... existing computations ...
    mobility = self._compute_income_mobility(period_state.households)

    entry = HistoryEntryV2(
        # ... existing fields ...
        income_mobility=mobility,
    )
```

**Step 3**: Display in `reporter.py` `_format_statistics_evolution_v2()`.

**Step 4**: Add test in `tests/test_observer_v2.py`.

### 3. Adding a New LLM Provider

**Step 1**: Implement the `LLMProvider` protocol in `llm_providers.py`:

```python
class OpenAIProvider:
    """OpenAI-compatible LLM provider."""

    def __init__(self, config: SimulationConfig) -> None:
        self.model = getattr(config, "llm_model", "gpt-4")

    def generate(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> str:
        # ... OpenAI API call ...
        return json_response

    def generate_batch(self, batch: list[list[dict[str, str]]], schema: type[BaseModel]) -> list[str]:
        return [self.generate(msgs, schema) for msgs in batch]
```

**Step 2**: Register in `create_provider()`:

```python
def create_provider(config, rng) -> LLMProvider:
    if config.llm_provider == "openai":
        return OpenAIProvider(config)
    elif config.llm_provider == "anthropic":
        return AnthropicProvider(config)
    else:
        return MockProvider(rng, config)
```

**Step 3**: Add tests using MockProvider pattern.

## Testing Guidelines

### Test Organization

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test component interactions (mark with `@pytest.mark.slow`)
- **Property tests**: Test formal invariants (PROP-001 through PROP-006 in `test_properties.py`)
- **Scale tests**: Test performance with large N (in `test_scale.py`)

### Property Invariants to Maintain

When modifying economics, market clearing, or state management code, verify these properties still hold:

| Property | What to test |
|----------|-------------|
| PROP-001 | Same seed + config produces identical output |
| PROP-002 | All household wealth >= `a_min` after every step |
| PROP-003 | Market clearing error < `tatonnement_tolerance` (analytical: zero) |
| PROP-004 | No NaN/Inf in wealth, consumption, or output |
| PROP-005 | Social welfare is finite and non-negative |
| PROP-006 | All constitutional rules pass AST validation |
| PROP-008 | KFE distribution mass conservation (sum = 1.0) |
| PROP-010 | Government budget identity holds each period |

**HANK v3 additions**:
- **PROP-008**: Mass conservation in KFE distribution tracking (total probability = 1.0)
- **PROP-010**: Government budget constraint identity: `B' = (1+r^b)*B + G + Tr - T`

### v2 Test Fixtures

Common fixtures are in `tests/conftest.py`. Key ones:

```python
@pytest.fixture
def v2_config() -> SimulationConfigV2:
    return SimulationConfigV2(num_agents=20, max_periods=10, seed=42, benchmark_mode=True)

@pytest.fixture
def sample_households(v2_config) -> list[HouseholdState]:
    period_state, _ = initialize_simulation_v2(v2_config)
    return period_state.households
```

### Testing LLM Integration

Always use `MockProvider` for unit tests — never make real API calls:

```python
def test_llm_engine_economic_decisions():
    config = SimulationConfigV2(num_agents=20, benchmark_mode=False, llm_provider="mock")
    rng = SimulationRNG(42)
    engine = LLMDecisionEngine(config=config, rng=rng)
    decisions = engine.collect_economic_decisions(households, market, constitution)
    assert len(decisions) == len(households)
```

### Determinism Tests

Always test determinism for any function that uses RNG:

```python
def test_initialization_v2_deterministic():
    config = SimulationConfigV2(num_agents=20, seed=42, benchmark_mode=True)
    state1, _ = initialize_simulation_v2(config)
    state2, _ = initialize_simulation_v2(config)
    assert state1.households[0].wealth == state2.households[0].wealth
    assert state1.shocks.productivity_grid == state2.shocks.productivity_grid
```

## Development Workflow

### Cost-Conscious Development

| Task | Recommended mode | Cost |
|------|-----------------|------|
| Writing tests | `MockProvider` | $0 |
| Debugging logic | `benchmark_mode=True` | $0 |
| Testing LLM integration | Haiku, 5 agents, 5 periods | ~$0.02 |
| Validation runs | Haiku, 20 agents, 50 periods | ~$3 |
| Full simulation | Sonnet, 50 agents, 100 periods | ~$56 |

### Debugging

Enable detailed logging:

```python
import logging
from emergent_constitution.logging import configure_logging

configure_logging(level=logging.DEBUG)
```

Structured log events to look for:
- `simulation_v2.initialized` — config summary
- `step2.markets_cleared` — wage, interest_rate, clearing_error
- `step3.decisions_collected` — counts
- `step7.governance_processed` — proposals, passed count
- `step8.states_updated` — total consumption, total wealth

## Git Workflow

### Branch Naming

- `feat/description` — New features
- `fix/description` — Bug fixes
- `refactor/description` — Code refactoring
- `docs/description` — Documentation updates
- `test/description` — Test additions/fixes

### Commit Messages

Use Conventional Commits format:

```
feat: add environmental regulation rule type

Allows agents to propose environmental rules that limit
firm pollution output based on R&D spending.

Traceability: REQ-022

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

### Pull Request Process

1. Create feature branch from `main`
2. Implement feature with tests
3. Ensure all 1412+ tests pass
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

    Traceability: REQ-007, PROP-004
    """
```

Requirements are in `AgentSocietyPlanning/spec/requirements.md`:
- REQ-001 through REQ-037: Full DSGE-HA simulation requirements
- PROP-001 through PROP-006: Formal invariant properties

## Performance

### Bottlenecks

1. **LLM latency** (dominant in LLM mode): ~1-3s per batch call
2. **VFI solver** (initial convergence): ~5-30s one-time, <0.1s/agent after
3. **Tatonnement**: <1s per period
4. **Pareto computation**: O(N^2), disable for N > 200

### Memory

- Per household: ~1KB
- Per period state: ~100KB for 50 agents
- Full history (100 periods): ~10MB

## Release Process

1. Update version in `pyproject.toml` and `src/emergent_constitution/__init__.py`
2. Run full test suite: `pytest --cov`
3. Build: `rye build` or `python -m build`
4. Tag release: `git tag v0.2.0 && git push --tags`
5. Create GitHub release with notes
