"""
Created on Tue Oct  6 16:17:24 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent

###############################################################################
# Registry
###############################################################################

AGENTS: dict[str, type[BaseAgent]] = {
    "random": RandomAgent,
}

def make_agent(name: str, seed: int, **params) -> BaseAgent:
    if name not in AGENTS:
        raise ValueError(f"Unknown agent '{name}'. Registered: {sorted(AGENTS)}")
    return AGENTS[name](seed=seed, **params)