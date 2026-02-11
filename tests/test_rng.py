"""Tests for SimulationRNG — determinism guarantee (PROP-001)."""

from __future__ import annotations

from emergent_constitution.rng import SimulationRNG


class TestSimulationRNG:
    def test_same_seed_same_sequence(self):
        rng1 = SimulationRNG(seed=42)
        rng2 = SimulationRNG(seed=42)
        for _ in range(100):
            assert rng1.random() == rng2.random()

    def test_different_seed_different_sequence(self):
        rng1 = SimulationRNG(seed=42)
        rng2 = SimulationRNG(seed=99)
        results1 = [rng1.random() for _ in range(20)]
        results2 = [rng2.random() for _ in range(20)]
        assert results1 != results2

    def test_gauss_deterministic(self):
        rng1 = SimulationRNG(seed=123)
        rng2 = SimulationRNG(seed=123)
        for _ in range(50):
            assert rng1.gauss(0, 1) == rng2.gauss(0, 1)

    def test_uniform_range(self):
        rng = SimulationRNG(seed=42)
        for _ in range(100):
            val = rng.uniform(5.0, 10.0)
            assert 5.0 <= val <= 10.0

    def test_choice_deterministic(self):
        rng1 = SimulationRNG(seed=42)
        rng2 = SimulationRNG(seed=42)
        items = ["a", "b", "c", "d", "e"]
        for _ in range(20):
            assert rng1.choice(items) == rng2.choice(items)

    def test_shuffle_deterministic(self):
        rng1 = SimulationRNG(seed=42)
        rng2 = SimulationRNG(seed=42)
        list1 = [1, 2, 3, 4, 5]
        list2 = [1, 2, 3, 4, 5]
        rng1.shuffle(list1)
        rng2.shuffle(list2)
        assert list1 == list2

    def test_randint_range(self):
        rng = SimulationRNG(seed=42)
        for _ in range(100):
            val = rng.randint(1, 6)
            assert 1 <= val <= 6

    def test_seed_stored(self):
        rng = SimulationRNG(seed=42)
        assert rng.seed == 42
