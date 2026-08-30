Option A — Self-play (agent vs. past selves)
Simplest to define, most established (AlphaZero-style). The environment stays fixed; what recurses is the opponent pool. Engine needs: fast checkpointing, ability to load N policy snapshots and run them against each other in parallel, deterministic replay for debugging. Doesn't require any environment-generation machinery — you could start training within days of having a working sim loop.

Option B — Self-generated curricula (agent/controller proposes tasks)
The environment is parameterized (e.g., grid layouts, goal placements, obstacle density), and something — a controller, a GAN-like adversary, or the agent's own learning-progress signal — picks which parameters to train on next. This is IMGEP/PowerPlay/POET territory. Engine needs: an environment-generation API where "make a level" is a first-class function taking parameters, not a hand-authored asset. More work upfront, but directly usable for ARC-AGI-3-style goal discovery research, which you're already doing.

Option C — Self-modifying environment (open-ended, POET-style)
Both agent and environment evolve together, environments get harder as agents solve them, potentially forking into diverging environment lineages. Most ambitious, most research-y, least likely to converge to something usable quickly. Probably a "later" idea, not a "first engine version" idea.

###########################

Weirder training setups than pursuit-evasion

Asymmetric information games — the enemy sees the full grid, but the player only sees a small radius around themselves (real fog-of-war). Forces genuinely different strategies to emerge — the player has to learn to predict threat from partial information, not just react to it. This is a much richer problem than full observability and maps directly onto your ARC-AGI-3 goal-inference work.
A referee/rule-generator agent — instead of you hand-placing walls and goals, a third "director" agent (POET-style, from way back in our very first conversation) that's itself being trained to place obstacles/goals to keep the match interesting — too easy and it gets penalized, too impossible and it also gets penalized. The level design becomes a third learned participant in the game, not a static input.
Deception/bluffing dynamics — an enemy that can only see the player's recent trajectory, not their exact position, so the player can learn to fake a direction and cut back — genuine game-theoretic bluffing emerging from simple grid rules.
Multi-goal negotiation — several players, only one goal tile, whoever's policy is "smarter" gets there first — competitive multi-agent racing instead of pursuit.

Where the "self-recursive" idea gets genuinely strange

The environment mutates its own rules based on who's winning — not just harder terrain (POET), but literally different physics: if the player is dominating, walls start behaving differently (teleporting, moving), rules the player has to re-discover mid-match. This is close to your ARC-AGI-3 agent's whole domain — an agent that has to infer mechanics it wasn't told — except now the mechanics are actively adversarial and shifting in response to the agent's own success.
Agents that can modify the grid itself as an action — not just move, but place a wall, dig through terrain, drop a decoy. Suddenly "the map" is a shared mutable resource both agents are fighting over, not just a stage they move around on.
A meta-agent that watches many matches and designs new enemy archetypes — effectively an automated game designer that's observing what strategies dominate and inventing counters, which is basically automated balance-patching, done by a learning system instead of a human dev team.

The pure "just for fun/aesthetic" direction

Once wgpu rendering eventually exists, watching hundreds of training episodes running in fast-forward, visually, side by side — a "population" view, like watching evolution happen on screen, which is genuinely mesmerizing to watch even without any deep research point to it.
A spectator mode where a human can jump into an otherwise fully-AI match as one of the agents mid-training, and feel directly how much harder/easier the opponent has gotten over training time — a very visceral way to feel your own research progress, not just read a loss curve.