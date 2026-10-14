"""
Created on Tue Oct 13 13:11:54 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np

from collections import deque

###############################################################################
# Math
###############################################################################

def action_margins(Q):
    """
    Takes an array of shape (T, A) and returns the best minus second-best 
    value per row, shape (T,)
    """
    # Convert to array
    Q = np.asarray(Q, dtype=float)

    if Q.ndim != 2:
        raise ValueError(f"Q must have shape (T, A), got shape {Q.shape}")
    
    if Q.shape[1] < 2:
        raise ValueError(f"Q needs at least 2 actions to have a margin, got {Q.shape[1]}")
    
    sort = np.sort(Q, axis=1)
    return sort[:, -1] - sort[:, -2]

def feature_second_moments(X):
   """
   Calculate mean(x²) per column of (T, n_features)
   """
   # Convert to array
   X = np.asarray(X, dtype=float)
   
   if X.shape[0] == 0:
       raise ValueError("X cannot be empty")
   
   if len(X.shape) == 1:
       raise ValueError(f"X have to be (T, n_features) but got {X.shape}")
   
   second_moments = np.mean(X**2, axis=0)
   
   return second_moments
   
def moment_eigenvalues(X):
    """
    Calculate eigenvalues of XᵀX / T, sorted largest first.
    """
    # Convert to array
    X = np.asarray(X, dtype=float)
    
    if X.shape[0] == 0:
        raise ValueError("X cannot be empty")
    
    if len(X.shape) == 1:
        raise ValueError(f"X have to be (T, n_features) but got {X.shape}")
    
    eigenvalues = (X.T @ X) / X.shape[0]
    
    eigenvalues = np.linalg.eigvalsh(eigenvalues)
    
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    return eigenvalues
    
def condition_number(X, tol=1e-8):
    """
    Calculate largest eigenvalue divided by the smallest one that is bigger 
    than tol times the largest. Eigenvalues below that are ignored, 
    because a feature that is always 0 would otherwise make the ratio infinite.
    """
    
    eigenvalues = moment_eigenvalues(X)
    
    max_value = eigenvalues[0]
    
    if max_value <= 0:
        raise ValueError("X has no variation: all feature moments are zero")
    
    min_values = [v for v in eigenvalues if v >= max_value * tol]
    
    min_value = min_values[-1]
    
    return max_value / min_value
    
def binomial_se(p, n):
    
    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, provided was: {p}")
        
    if n < 1:
        raise ValueError(f"n must be higher or equal 1, provided was: {n}")
    
    return np.sqrt(p * (1-p)/n)
    
def expected_q_change(x, delta, alpha):
    """
    Change in Q(s, a) caused by one update at the same state:
        alpha * delta * (x_aug . x_aug), where x_aug is x with the bias 1.0 appended.
    """
    x = np.asarray(x, dtype=float)
    x_aug = np.append(x, 1.0)

    return alpha * delta * (x_aug @ x_aug)


###############################################################################
# Recorder
###############################################################################

class Recorder():
    
    def __init__(self, max_rows=100_000):
        
        if max_rows <= 0:
            raise ValueError(f"Capacity cannot be lower than 1, provided:{max_rows}")
        
        # Initialize buffer
        self.decisions = deque(maxlen=max_rows)
        self.updates = deque(maxlen=max_rows)
        
    def log_decision(self, x, q, training):
        
        self.decisions.append({
            "x": np.array(x, dtype=float),
            "q": np.array(q, dtype=float),
            "training" : bool(training)
        })
        
    def log_update(self, x, delta, q_change, reward):
        
        self.updates.append({
            "x"        : np.array(x, dtype=float),
            "delta"    : float(delta),
            "q_change" : float(q_change),
            "reward"   : float(reward)
        })
        
    
    def reset(self):
        
        # Clear buffer
        self.decisions.clear()
        self.updates.clear()
        
    def summary(self, margin_threshold=0.01):
        
        if not self.decisions and not self.updates:
            raise ValueError("Both decisions and updates are empty")
            
        summary = {
            "n_decisions" : None, 
            "n_updates"   : None,  
            "margin_median"      : None, 
            "margin_share_below" : None, 
            "dq_quiet_median"    : None, 
            "dq_quiet_p95"     : None, 
            "dq_reward_median" : None,  
            "dq_reward_p95"    : None, 
            "snr" : None, 
            "feature_second_moments" : None, 
            "condition_number" : None, 
        }
        
        summary["n_decisions"] = len(self.decisions)
        summary["n_updates"]   = len(self.updates)
        
        q_vectors = []
        
        for decision in self.decisions:
            
            if not decision["training"]:
                q_vectors.append(decision["q"])
        
        if q_vectors:
            Q = np.stack(q_vectors)
            margin = action_margins(Q)
        
            summary["margin_median"] = float(np.median(margin))
        
            below = margin < margin_threshold
            summary["margin_share_below"] = float(np.mean(below))
            
        quiet, rewarded = [], []
        
        for update in self.updates:
            
            q_change = abs(update["q_change"])
            
            if update["reward"] == 0:
                quiet.append(q_change)
                
            if update["reward"] > 0:
                rewarded.append(q_change)
                
        if quiet:
            summary["dq_quiet_median"] = float(np.median(quiet))
            summary["dq_quiet_p95"]    = float(np.percentile(quiet, 95))
        
        if rewarded:
            summary["dq_reward_median"] = float(np.median(rewarded))
            summary["dq_reward_p95"]    = float(np.percentile(rewarded, 95))
        
        if summary["margin_median"] is not None and summary["dq_reward_median"] is not None:
            if summary["dq_reward_median"] > 0:
                summary["snr"] = summary["margin_median"] / summary["dq_reward_median"]
                
        xs = []
        for decision in self.decisions:
            xs.append(decision["x"])
            
        if xs:
            
            X = np.stack(xs)
            summary["feature_second_moments"] = feature_second_moments(X)
            
            if np.any(X):
                summary["condition_number"] = float(condition_number(X))
                
        return summary
    
    
    