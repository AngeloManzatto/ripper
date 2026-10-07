"""
Created on Sat Jun  6 18:50:21 2026

@author: Angelo Antonio Manzatto
--------------
Activation function registry for NEAT.

Design
------
- Every activation is a pure function: float → float. No state, no classes.
- The module exposes one primary object: DEFAULT_REGISTRY, an ActivationRegistry
  instance pre-loaded with all standard functions.
- NEATConfig stores activation names as strings. Code resolves them through
  the registry — nothing imports activation functions directly.
- Custom functions can be registered at runtime without touching this file.
- Numerical stability is handled explicitly for functions with overflow risk.

Adding a new activation
-----------------------
1. Write the function (float → float).
2. Call DEFAULT_REGISTRY.register("name", function) after this module is imported.
   Or add it to _BUILTIN_ACTIVATIONS below and it is available everywhere.

Usage
-----
    from neat.activations import DEFAULT_REGISTRY

    fn  = DEFAULT_REGISTRY.get("tanh")       # raises if not found
    fn  = DEFAULT_REGISTRY.get("tanh", None) # returns None if not found
    val = fn(0.5)

    names = DEFAULT_REGISTRY.names()         # sorted list of all names
    DEFAULT_REGISTRY.validate("sigmoid")     # raises ActivationNotFoundError if missing
"""

###############################################################################
# libraries
###############################################################################

import math
from typing import Callable, Dict, Iterable, List, Optional

###############################################################################
# Type alias
###############################################################################

ActivationFn = Callable[[float], float]

###############################################################################
# Error
###############################################################################

class ActivationNotFoundError(KeyError):
    """Raised when an activation name is not in the registry."""

    def __init__(self, name: str, available: Iterable[str]) -> None:
        self.name      = name
        self.available = sorted(available)
        super().__init__(
            f"Activation '{name}' not found. "
            f"Available: {self.available}"
        )


###############################################################################
# Pure activation functions
###############################################################################

def sigmoid(x: float) -> float:
    """
    Logistic sigmoid with 4.9 steepness — matches the original NEAT paper.
    Clamped to avoid overflow on large negative inputs.
    """
    x = max(-60.0, min(60.0, x))   # clamp: exp(60) ≈ 1e26, safe
    return 1.0 / (1.0 + math.exp(-4.9 * x))


def tanh(x: float) -> float:
    """Hyperbolic tangent. Output in (-1, 1)."""
    return math.tanh(x)


def relu(x: float) -> float:
    """Rectified linear unit. Output in [0, ∞)."""
    return x if x > 0.0 else 0.0


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """Leaky ReLU. Allows small gradient for negative inputs."""
    return x if x > 0.0 else alpha * x


def elu(x: float, alpha: float = 1.0) -> float:
    """
    Exponential linear unit.
    Clamped to avoid overflow on large negative inputs.
    """
    if x > 0.0:
        return x
    x_clamped = max(-60.0, x)
    return alpha * (math.exp(x_clamped) - 1.0)


def swish(x: float) -> float:
    """
    Swish / SiLU: x * sigmoid(x).
    Clamped to avoid overflow.
    """
    x_clamped = max(-60.0, min(60.0, x))
    return x * (1.0 / (1.0 + math.exp(-x_clamped)))


def softplus(x: float) -> float:
    """
    Softplus: log(1 + exp(x)). Smooth approximation of ReLU.
    Uses numerically stable form for large x.
    """
    if x > 30.0:
        return x                          # log(1 + exp(x)) ≈ x for large x
    return math.log(1.0 + math.exp(x))


def gaussian(x: float) -> float:
    """Gaussian: exp(-x²). Output in (0, 1], maximum at x=0."""
    x_clamped = max(-10.0, min(10.0, x))  # exp(-100) underflows to 0, fine
    return math.exp(-(x_clamped ** 2))


def identity(x: float) -> float:
    """Identity / linear. Output = input. Useful for regression outputs."""
    return x


def sinusoid(x: float) -> float:
    """Sine function. Periodic, output in [-1, 1]."""
    return math.sin(x)


def cosine(x: float) -> float:
    """Cosine function. Periodic, output in [-1, 1]."""
    return math.cos(x)


def inverse(x: float) -> float:
    """Negation: output = -input."""
    return -x


def step(x: float) -> float:
    """Heaviside step. Output is 0 or 1."""
    return 1.0 if x > 0.0 else 0.0


def arctan(x: float) -> float:
    """Arc tangent. Output in (-π/2, π/2)."""
    return math.atan(x)


def sign(x: float) -> float:
    """Sign function. Output is -1, 0, or 1."""
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


###############################################################################
# Registry
###############################################################################

class ActivationRegistry:
    """
    Maps activation names (strings) to callable functions.

    Immutable in the sense that the default registry is built once at module
    load. You can create a custom registry or extend via register().

    Thread safety: register() is not thread-safe. Build registries before
    spawning threads.
    """

    def __init__(self, functions: Optional[Dict[str, ActivationFn]] = None) -> None:
        self._registry: Dict[str, ActivationFn] = dict(functions or {})

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str, fn: ActivationFn) -> None:
        """
        Register a new activation function.

        Overwrites if name already exists — this is intentional so callers
        can replace a builtin with a custom version for experimentation.

        Args:
            name: String key used in NEATConfig.activation_options.
            fn:   Pure function float → float.
        """
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn)}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"name must be a non-empty string, got {name!r}")
        self._registry[name] = fn

    def register_many(self, functions: Dict[str, ActivationFn]) -> None:
        """Register multiple functions at once."""
        for name, fn in functions.items():
            self.register(name, fn)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(
        self,
        name: str,
        default: Optional[ActivationFn] = None,
    ) -> Optional[ActivationFn]:
        """
        Return the activation function for name, or default if not found.

        Args:
            name:    Activation name.
            default: Returned when name is not registered. Defaults to None.

        Returns:
            The activation function, or default.
        """
        return self._registry.get(name, default)

    def get_or_raise(self, name: str) -> ActivationFn:
        """
        Return the activation function for name.

        Raises:
            ActivationNotFoundError: if name is not registered.
        """
        fn = self._registry.get(name)
        if fn is None:
            raise ActivationNotFoundError(name, self._registry)
        return fn

    def validate(self, name: str) -> None:
        """
        Raise ActivationNotFoundError if name is not registered.
        Used by NEATConfig.__post_init__ for early validation.
        """
        self.get_or_raise(name)

    # ── Introspection ─────────────────────────────────────────────────────────

    def names(self) -> List[str]:
        """Return sorted list of all registered activation names."""
        return sorted(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"ActivationRegistry(n={len(self)}, names={self.names()})"

    # ── Factory ───────────────────────────────────────────────────────────────

    def copy(self) -> "ActivationRegistry":
        """Return a shallow copy — useful as a base for custom registries."""
        return ActivationRegistry(dict(self._registry))


###############################################################################
# Built-in registry
###############################################################################

_BUILTIN_ACTIVATIONS: Dict[str, ActivationFn] = {
    "sigmoid":    sigmoid,
    "tanh":       tanh,
    "relu":       relu,
    "leaky_relu": leaky_relu,
    "elu":        elu,
    "swish":      swish,
    "softplus":   softplus,
    "gaussian":   gaussian,
    "identity":   identity,
    "sinusoid":   sinusoid,
    "cosine":     cosine,
    "inverse":    inverse,
    "step":       step,
    "arctan":     arctan,
    "sign":       sign,
}

# Single shared instance — import this everywhere.
DEFAULT_REGISTRY: ActivationRegistry = ActivationRegistry(_BUILTIN_ACTIVATIONS)