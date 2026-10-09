"""
Created on Fri Oct  9 10:36:28 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from core.transition import Transition

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

###############################################################################
# Test
###############################################################################

a = SimpleNamespace(tag="before")
b = SimpleNamespace(tag="after")
t = Transition(obs=a, action="Up", next_obs=b, terminated=False, truncated=True)

assert t.obs is a and t.next_obs is b
assert t.action == "Up"
assert t.terminated is False and t.truncated is True

try:
    t.terminated = True
except FrozenInstanceError:
    pass
else:
    raise AssertionError("Transition must be frozen")