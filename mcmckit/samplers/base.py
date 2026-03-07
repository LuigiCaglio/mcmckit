from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """Abstract base class for all samplers."""

    @abstractmethod
    def initialize(self, problem, x0):
        """Set up sampler state before stepping.

        Parameters
        ----------
        problem : Problem
        x0 : array-like
            Starting point in parameter space.
        """

    @abstractmethod
    def step(self):
        """Advance the chain by one step. Must call initialize first."""

    @abstractmethod
    def run(self, problem, x0):
        """Run the sampler to completion and return a Result."""

    @abstractmethod
    def get_result(self):
        """Return a Result from samples collected so far."""
