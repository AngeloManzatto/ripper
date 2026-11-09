"""
Created on Mon Oct 19 14:07:47 2026

@author: Angelo Antonio Manzatto

"""

###############################################################################
# libraries
###############################################################################

import math

###############################################################################
# Activation functions
###############################################################################

def sigmoid(x):
    """Logistic sigmoid with steepness 4.9, as in the original NEAT paper. Output in (0, 1)."""
    x = max(-60.0, min(60.0, x))   # clamp: exp(4.9 * 60) is still a normal float
    return 1.0 / (1.0 + math.exp(-4.9 * x))


def tanh(x):
    """Hyperbolic tangent. Output in (-1, 1)."""
    return math.tanh(x)


def relu(x):
    """Rectified linear unit. Output in [0, inf)."""
    return x if x > 0.0 else 0.0


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: small slope (alpha) for negative inputs."""
    return x if x > 0.0 else alpha * x


def elu(x, alpha=1.0):
    """Exponential linear unit. Clamped to avoid overflow."""
    if x > 0.0:
        return x
    return alpha * (math.exp(max(-60.0, x)) - 1.0)


def swish(x):
    """Swish / SiLU: x * sigmoid(x) (plain logistic, steepness 1). Clamped to avoid overflow."""
    x_clamped = max(-60.0, min(60.0, x))
    return x * (1.0 / (1.0 + math.exp(-x_clamped)))


def softplus(x):
    """Softplus: log(1 + exp(x)), a smooth ReLU. Stable for large x."""
    if x > 30.0:
        return x                      # log(1 + exp(x)) is x for large x
    return math.log(1.0 + math.exp(x))


def gaussian(x):
    """Gaussian: exp(-x^2). Output in (0, 1], maximum at x = 0."""
    x_clamped = max(-10.0, min(10.0, x))
    return math.exp(-(x_clamped ** 2))


def identity(x):
    """Identity / linear: output = input."""
    return x


def sinusoid(x):
    """Sine. Periodic, output in [-1, 1]."""
    return math.sin(x)


def cosine(x):
    """Cosine. Periodic, output in [-1, 1]."""
    return math.cos(x)


def inverse(x):
    """Negation: output = -input (NOT 1/x)."""
    return -x


def step(x):
    """Heaviside step. Output is 0 or 1 (0 at x = 0)."""
    return 1.0 if x > 0.0 else 0.0


def arctan(x):
    """Arc tangent. Output in (-pi/2, pi/2)."""
    return math.atan(x)


def sign(x):
    """Sign. Output is -1, 0 or 1."""
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0

###############################################################################
# Name -> function
###############################################################################

ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "elu": elu,
    "swish": swish,
    "softplus": softplus,
    "gaussian": gaussian,
    "identity": identity,
    "sinusoid": sinusoid,
    "cosine": cosine,
    "inverse": inverse,
    "step": step,
    "arctan": arctan,
    "sign": sign,
}


def get_activation(name):
    """Return the function for name. Unknown name -> ValueError listing the valid names."""
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; available: {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]