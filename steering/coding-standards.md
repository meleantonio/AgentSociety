# Coding Standards — HANK Upgrade

## Numerical Code

- All numerical solvers must use NumPy vectorized operations (no Python loops over grid points in production code)
- Policy functions must be tested for monotonicity in wealth
- Euler equation residuals must be reported at convergence
- All division operations must guard against zero denominators (`max(x, 1e-10)`)
- NaN/Inf values must be caught and handled (use `np.isfinite()` checks)
- Tolerances must be configurable, not hardcoded

## Backward Compatibility

- Every new feature must be gated behind a config flag with default OFF
- `solver_method="vfi_numpy"` must reproduce exact v2 behavior
- `market_clearing_method="analytical"` must reproduce exact v2 behavior
- The existing 977 tests must pass at every phase boundary
- No breaking changes to Pydantic model interfaces (add fields with defaults)

## Testing

- Every new function must have unit tests
- Numerical accuracy tests: compare against analytical solutions where available
- Property-based tests for invariants (budget balance, mass conservation, market clearing)
- Integration tests for each phase: full-period simulation with new features

## Performance

- EGM solver must be faster than brute-force VFI for the same grid resolution
- KFE forward step must be O(n_a * n_z) per period
- Market clearing bisection must converge in < 50 iterations typically
- VFI cache must be preserved across periods when prices are stable

## Code Organization

- One solver per file: `egm_solver.py`, `numerical_solver.py`, `entrepreneurial_solver.py`
- Models in `models/` subdirectory with Pydantic
- Configuration in `config.py` using `SimulationConfigV2`
- All new modules must have module-level docstring referencing requirements
