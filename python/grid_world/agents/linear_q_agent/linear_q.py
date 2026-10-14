"""
Created on Sun Oct 11 08:44:34 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np

###############################################################################
# Math utils
###############################################################################

def with_bias(x, n_features: int) -> np.ndarray:
    
    """Validate x and return it as a float64 vector with a trailing 1.0 (the bias input)."""
    
    x = np.asarray(x, dtype="float64")
    
    if x.shape != (n_features,):
        raise ValueError(f"expected {n_features} features, got shape {x.shape}")
        
    return np.append(x, 1.0)

def q_values(W: np.ndarray, x) -> np.ndarray:
    
    """Q(x, a) for every action a. W has shape (n_actions, n_features + 1)."""
    
    return W @ with_bias(x, W.shape[1] - 1)

def update(W: np.ndarray, 
           x, action: int, 
           reward: float, next_x,
           terminated: bool, 
           alpha: float, 
           gamma: float) -> float:
    """
    One Q-learning step on row `action` of W. Modifies W IN PLACE.
    Returns the TD error (target - prediction).
    Only `terminated` cuts the bootstrap.
    """
    
    if not 0 <= action < W.shape[0]:
        raise ValueError(f"action {action} out of range for {W.shape[0]} actions")

    phi = with_bias(x, W.shape[1] - 1)
    prediction = W[action] @ phi

    if terminated:
        target = reward
    else:
        target = reward + gamma * q_values(W, next_x).max()

    delta = target - prediction
    W[action] += alpha * delta * phi
    
    return float(delta)

def epsilon_greedy(q, epsilon, rng) -> int:
    
    if not 0 <= epsilon <= 1.0:
        raise ValueError(f"expected epsilon: {epsilon} to be between 0.0 and 1.0")
        
    values = np.asarray(q)
    
    if np.isnan(values).any():
        raise ValueError("q contains NaN: the weights have diverged (alpha too large?)")
    
    action_idx = int(np.argmax(rng.random(values.shape) * (values==values.max())))
        
    if rng.random() < epsilon:
        action_idx = rng.choice(len(values))

    return int(action_idx)
    