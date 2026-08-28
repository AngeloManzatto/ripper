Option A — Self-play (agent vs. past selves)
Simplest to define, most established (AlphaZero-style). The environment stays fixed; what recurses is the opponent pool. Engine needs: fast checkpointing, ability to load N policy snapshots and run them against each other in parallel, deterministic replay for debugging. Doesn't require any environment-generation machinery — you could start training within days of having a working sim loop.

Option B — Self-generated curricula (agent/controller proposes tasks)
The environment is parameterized (e.g., grid layouts, goal placements, obstacle density), and something — a controller, a GAN-like adversary, or the agent's own learning-progress signal — picks which parameters to train on next. This is IMGEP/PowerPlay/POET territory. Engine needs: an environment-generation API where "make a level" is a first-class function taking parameters, not a hand-authored asset. More work upfront, but directly usable for ARC-AGI-3-style goal discovery research, which you're already doing.

Option C — Self-modifying environment (open-ended, POET-style)
Both agent and environment evolve together, environments get harder as agents solve them, potentially forking into diverging environment lineages. Most ambitious, most research-y, least likely to converge to something usable quickly. Probably a "later" idea, not a "first engine version" idea.