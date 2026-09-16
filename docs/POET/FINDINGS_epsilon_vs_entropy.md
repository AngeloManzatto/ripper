# Finding: DDQN's epsilon-floor vs PPO's lack of an exploration floor,
# observed on a degenerate seed layout

## Setup
A generated seed layout placed the Player immediately adjacent to the Goal
(`.PG.`), with an Enemy elsewhere. This is a near-trivial task — a single
correct move solves it — providing almost no genuine learning signal.

## Observation
- **PPO**, trained on this seed, reached 100% win rate almost immediately
  (Iteration 0) and then never produced a single accepted mutation across
  560 generations. Every candidate evaluation scored 0%.
- **DDQN**, trained on the exact same seed, reached its first accepted
  mutation at Generation 41 (win_rate=0.43 on a genuinely different candidate).

## Diagnosis
At Generation 41, DDQN's epsilon was still 0.577 — meaning ~58% of actions
were still forced-random, even while "solving" the trivial seed. This kept
injecting diverse, exploratory experience into the replay buffer throughout
training, preventing the policy from ever fully collapsing onto one brittle,
repeated trajectory — this appears to have been incidental, not by design,
but it functioned as a real safeguard against degenerate-seed collapse.

PPO has no equivalent exploration floor. Its exploration comes entirely from
the sharpness of its own learned action-distribution. On a trivial task,
that distribution sharpens toward near-certainty almost immediately (nothing
to be uncertain about), and `Categorical.sample()` then reproduces nearly
the same trajectory every rollout — genuine entropy collapse, with no
mechanism to counteract it as currently implemented (no entropy bonus term
in the loss).

## Implication
This is not a bug in either algorithm — it's a real, structural difference
in how value-based (epsilon-greedy) and policy-gradient (distribution-based)
methods handle exploration, especially under degenerate/low-signal tasks.
DDQN's epsilon-floor is a blunt but effective incidental safeguard; PPO
requires an explicit entropy bonus to get the equivalent property, which
this implementation currently lacks.

## Action items
- Add an entropy bonus term to PPO's loss (deferred, tracked in
  POET_ISSUES.md / POET_ROADMAP.md).
- Fix the root cause independently: prevent degenerate seeds via a
  min_start_goal_distance constraint at generation time (see below) —
  helps both algorithms regardless of the entropy-bonus fix.


# 2026-09-16
---

## Follow-up: sample efficiency, after fixing the degenerate-seed problem

After adding min_player_goal_distance (eliminating trivial seeds as a
confound), a fresh 1000-generation comparison was run:

- **DDQN**: archive_size > 100
- **PPO**: archive_size = 28

## Compute budget check
Verified this was not explained by an unfair compute allocation — if
anything, PPO received MORE raw environment steps per generation:
- PPO: episodes(20) * rollout_steps(512) = 10,240 steps/generation (fixed)
- DDQN: episodes(20) episodes, each running to `done` or max_tick(100) —
  worst case 2,000 steps/generation, realistically well under that since
  episodes typically resolve before hitting max_tick

## Conclusion
With PPO receiving comparable-or-greater raw compute per generation and
still producing a meaningfully smaller archive, the gap is best explained
by genuine algorithmic sample-efficiency difference, not a configuration
imbalance:
- DDQN reuses every transition via its replay buffer across the agent's
  entire training lifetime (thousands of reuses per transition over a
  full lineage).
- PPO is on-policy — each rollout's data is used for only
  `epochs_per_update` passes, then discarded entirely.

This matches the well-known theoretical tradeoff between off-policy
(DDQN) and on-policy (PPO) methods, now observed directly on this
project's own task rather than taken from the literature: for a small,
discrete-action, replay-buffer-friendly domain like this grid-world,
DDQN's data reuse gives it a real, measurable curriculum-progress
advantage over PPO as currently implemented (no entropy bonus, standard
on-policy data handling).

## Open question
Would PPO close some of this gap with an entropy bonus (encouraging
sustained exploration, addressing the still-unfixed collapse tendency
from the first finding)? Not yet tested — a natural next experiment.

# Test

seed - '################\n#............#.#\n#.#....#.......#\n#..............#\n#..........G...#\n#....#...#.....#\n#..........#.#.#\n#..###.#.#.....#\n#.............##\n#.............##\n#..#P..........#\n#..............#\n#.E.....#....#.#\n#...#.....#....#\n#..............#\n################'

 1 - DDQN - 73 - 1000
 1 - PPO  - 