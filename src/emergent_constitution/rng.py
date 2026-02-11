"""Seeded RNG wrapper — the single source of randomness (PROP-001)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


class SimulationRNG:
    """Deterministic random number generator wrapping `random.Random`.

    All simulation randomness must flow through this instance to guarantee
    reproducibility (PROP-001). Pass explicitly — never use module-level random.

    Args:
        seed: Integer seed for reproducibility.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def random(self) -> float:
        """Uniform float in [0, 1)."""
        return self._rng.random()

    def uniform(self, a: float, b: float) -> float:
        """Uniform float in [a, b]."""
        return self._rng.uniform(a, b)

    def gauss(self, mu: float, sigma: float) -> float:
        """Gaussian-distributed float."""
        return self._rng.gauss(mu, sigma)

    def choice(self, seq: Sequence[T]) -> T:
        """Pick one element uniformly at random."""
        return self._rng.choice(seq)

    def shuffle(self, seq: list[T]) -> None:
        """Shuffle list in place."""
        self._rng.shuffle(seq)

    def randint(self, a: int, b: int) -> int:
        """Random integer in [a, b] inclusive."""
        return self._rng.randint(a, b)
