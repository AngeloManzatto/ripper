# RIPPER — POMDP Training Diagnostics Log

Sep 25, 2026 · @Angelo

## Context

RIPPER trains two DDQN agents (player, enemy) adversarially inside a procedurally-generated, partially-observable grid world, as part of a POET-style (Wang et al., 2019) open-ended environment loop. Each agent sees only a local, fog-of-war-limited one-hot view of the grid — never privileged ground-truth information (this is a deliberate project rule, not an oversight).

For a long time, layouts used fixed spawn positions. Adding `world.reset(reposition=True)` — randomizing player/enemy/goal spawns every episode — was meant to force genuine generalization instead of trajectory memorization. It worked, in the sense that it exposed the problem: win rate capped around \~20% no matter how reward shaping was tuned. This log is the trail of diagnosing why, and what actually fixed it.

## Timeline

1. **Fixed-spawn baseline** — both agents reached 80–95% win rate on a memorized layout, with a visible co-adaptation curve (player dipping as enemy improved). Confirmed the network, loss, and training loop were all sound.
2. **Repositioned spawns, \~20% ceiling** — same agents, same reward, but a new layout every episode. Win rate capped at roughly 20% for both roles regardless of reward-shaping changes, with Timeout as the dominant outcome (\~60% of episodes).
3. **Isolated the variable** — re-running with `reposition=False` reproduced the 80–95% fixed-spawn result again. This ruled out the reward function, network architecture, and training loop as the bottleneck: the agents could clearly execute a path well, they just couldn't search a novel one.
4. **Observed the actual failure modes** — running a trained policy on a brand-new layout showed two distinct pathologies:
   - **Flip-flop**: alternating Left/Right or Up/Down between two adjacent cells indefinitely.
   - **Wall-bump self-loop**: repeatedly choosing an action that bumps a wall and does nothing, until an epsilon-random action finally broke it.
5. **Root cause** — the DQN is memoryless (a single current-view one-hot frame in, Q-values out). Searching an unseen, partially-observable layout for a goal requires remembering where you've already been; a purely reactive policy has no way to do that. See Failure modes below.

## Failure mode: perceptual / state aliasing

The flip-flop is a case of **perceptual aliasing**: a partially-observable agent treats two genuinely different world-states as identical (or near-identical) because its observation doesn't distinguish them, so a reactive policy oscillates between the locally-best-looking actions. This is a foundational, named problem in POMDP reinforcement learning, not something specific to RIPPER.

- Whitehead, S. & Ballard, D. (1991). [Learning to Perceive and Act by Trial and Error](https://link.springer.com/article/10.1023/A:1022619109594) — *Machine Learning*. The paper that coined "perceptual aliasing" and showed why purely reactive policies fail under it. ([Semantic Scholar PDF](https://www.semanticscholar.org/paper/Learning-to-perceive-and-act-by-trial-and-error-Whitehead-Ballard/6104d0e68c7f67da58b2f84a663df45d82d86b18))
- Whitehead, S. & Ballard, D. (1990). [Reinforcement Learning with Perceptual Aliasing: The Perceptual Distinctions Approach](https://dl.acm.org/doi/10.5555/1867135.1867164) — AAAI. Follow-up proposing a distinctions-based fix.

**RIPPER's fix**: give the network a persistent, per-episode memory of its own trajectory (a decaying "trail" channel appended to the one-hot observation), so two cells that look identical in the current local view no longer look identical once recent history is included — the RL-literature answer to aliasing is exactly "add memory or a richer observation," since a reactive policy alone can't resolve it by construction.

## Failure mode: invalid-action self-loop

The wall-bump loop is a distinct problem: when an action fails (bumps a wall), position doesn't change, so `state` and `next_state` are literally identical. The network gets zero new information from repeating it — if that action is (wrongly) ranked greedy-best for that state, it self-loops until epsilon-random exploration breaks it. This is the well-studied **invalid/illegal action masking** problem.

- Huang, S. & Ontañón, S. (2020). [A Closer Look at Invalid Action Masking in Policy Gradient Algorithms](https://arxiv.org/abs/2006.14171) — arXiv:2006.14171. Empirically shows masking invalid actions substantially outperforms letting the agent learn to avoid them via penalty.
- Vinyals, O. et al. (2019). [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z) — *Nature* (AlphaStar). Masks invalid actions in a huge action space as standard practice.
- Related: Yu, T. et al. (2021). [Addressing Action Oscillations through Learning Policy Inertia](https://arxiv.org/abs/2103.02287) — arXiv:2103.02287. Studies action oscillation directly (adjacent to the flip-flop above) and proposes an inertia term to dampen it, an alternative angle to adding memory.

**RIPPER's fix**: action masking at inference time. Before `argmax`, Q-values for any action that would walk into a wall or off the grid boundary are set to `-inf`, using the already-visible Wall channel — not privileged information, since the agent can already see adjacent walls in its own local view. This is a hard constraint applied at decision time, not something the network has to learn, so it eliminates the failure mode outright rather than reducing its frequency.

## Fixes applied

| Fix | Targets | What it changes | Requires learning? |
| --- | --- | --- | --- |
| Trail channel (`TrailTracker`) | Perceptual aliasing (flip-flop) | Appends a decaying, per-agent, per-episode history of own positions as an extra one-hot observation channel | Yes — network must learn to use it |
| Revisit penalty (`reward_for_revisit`) | Weak credit assignment for the trail signal | Small immediate penalty for stepping onto a cell your own trail already marks, instead of relying on a delayed timeout penalty | Yes, but faster (direct gradient) |
| Action masking | Wall-bump self-loop | Hard-excludes illegal moves (wall/boundary) from both greedy and epsilon-random action selection, using the already-visible Wall channel | No — geometric constraint, not learned |

All three are legitimate under the project's core constraint: no privileged information reaches an agent's own decision (fog-of-war one-hot views only). Trail = the agent's own trajectory. Revisit penalty = derived from the trail. Action mask = the agent's own currently-visible Wall channel.

## Open items

- [ ] Results of the current run (trail + revisit penalty + action masking together) not yet in — fill in once training completes.
- [ ] If the flip-flop persists even with masking removing the wall-bump case, consider whether `maxlen=4` on the trail is too short for the map size, or whether the revisit-penalty coefficient (`k_revisit=-0.03`) needs tuning.
- [ ] Longer-horizon alternative not yet tried: a persistent full-episode "discovered" mask (reusing the existing Rust-side fog-of-war `self.discovered` set) instead of a bounded 4-step trail — more complete but a bigger change.
- [ ] `min_enemies` floor still missing in `MutationConfig` (Rust) — current workaround is a Python-side `enemy_nothing_weight=1.0` stopgap.
- [ ] Broader future-ideas backlog (DM-as-regret-agent, MAP-Elites redesign of `EA_list`, cellular automata as a live mechanic, multi-agent parameter sharing) parked separately, not part of this diagnostic thread.
