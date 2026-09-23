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
1 - PPO  - 28 - 1000

seed - '################\n##...#.........#\n#.#.#....#.....#\n#E.#.....#.....#\n#..............#\n#.........#....#\n#.#............#\n#..............#\n#..#...........#\n#..............#\n#.....#........#\n#G.............#\n#.#..P.........#\n#..............#\n#..........#...#\n################'

1 - DDQN - 59 - 1000
1 - PPO  - 47 - 1000

seed - '################\n#.........#....#\n#.......#..#..##\n#..............#\n#..............#\n#..............#\n#....#..#......#\n#.............##\n#.........#....#\n#G#.#..........#\n#..............#\n#.#......P..#..#\n#..............#\n##....#........#\n#.E..#....#...##\n################'

1 - DDQN - 89 - 1000
1 - PPO - 60 - 1000

seed - '################\n#...........#..#\n##...#.........#\n#E....#..#.....#\n#.......#......#\n#...###........#\n##.#.........#.#\n#..##.#........#\n#......#.......#\n#...G..........#\n#............#.#\n#P.##...#......#\n#....#.........#\n##.........##..#\n#..#...........#\n################'

1 - DDQN - 101 - 1000

seed - '################\n#......G.....#.#\n#.....E..##....#\n#..............#\n#..............#\n##....#........#\n#..............#\n#.....#...#....#\n#.......#......#\n#....#.P.......#\n#..........#...#\n#..............#\n#.....#...#...##\n#..............#\n#..............#\n################'

1 - PPO - 42 - 1000

seed - '################\n#......P##.....#\n#......#.......#\n#..............#\n#...........#.##\n#........##....#\n##...#..#.#..#.#\n#..............#\n#...........#..#\n#E.............#\n#..............#\n#.#......#..G..#\n#..............#\n#..............#\n#.#............#\n################'

1 - DDQN - 87 - 1000

seed - '################\n#........#.....#\n#...........#..#\n##.#.....#.....#\n#.#E......P....#\n#.#.....#......#\n#.......#..##..#\n##...##.#..#...#\n#...........#.##\n#G#..........#.#\n#.#..#.........#\n#.......#......#\n#.............##\n#..............#\n#....#.........#\n################'

1 - PPO - 53 - 1000

# 2026-09-23
---

# FINDINGS: Static-Layout Deadlock in POET's Minimal-Criterion Gate

## Summary

A single-entry POET population (one static layout, player+enemy DDQN) can get
permanently stuck if the starting seed's win-rate imbalance is severe enough
that it never enters the minimal-criterion admit band. Once stuck, the
mutation step (`mutate_envs`) never fires for that entry — because mutation
eligibility itself requires passing the same minimal criterion the entry is
failing — so the population can never grow, and the same static layout keeps
training indefinitely.

## Observed symptoms

- Population size stayed at 1 across 150+ outer iterations (~15,000+ episodes)
  of a single static layout.
- Player win_rate pinned at ~1.0 (GoalReached), enemy win_rate ~0.0 (never
  Caught), stable across many consecutive evaluations.
- Player avg_loss dropped to ~0.0003 — consistent with the network converging
  to a single memorized trajectory rather than a general policy.
- Confirmed via added logging: `mutate_envs`'s reproduction-eligibility check
  (`passes_minimal_criterion(parent["last_eval"], ...)`) rejected the entry on
  every single outer iteration it was checked.

## Root cause 1: eligibility-gated mutation is a chicken-and-egg trap

`mutate_envs` only attempts mutation for entries that already pass the
minimal criterion. If a seed's imbalance is severe enough that win_rate never
enters `[win_low, win_high]` regardless of training duration, that entry can
never become reproduction-eligible — the mechanism meant to introduce
diversity is gated behind already having escaped the imbalance it's supposed
to help escape. The original POET paper (ES-based) is less prone to this
because ES's population-averaged gradient step doesn't collapse into a single
deterministic extreme the way a greedy DQN policy can (see the separate
"greedy-argmax cycling" finding, already documented).

## Root cause 2: deterministic spawns enable memorization, not generalization

`World::reset()` currently restores player/enemy/goal to the exact same
parsed layout positions every episode. Across thousands of repetitions of one
static entry, the network never observes a different spatial configuration —
it is not learning a general "navigate toward goal while evading a hunter"
policy, it is learning the one specific move sequence that solves the one
specific board it has ever seen. This is consistent with the near-zero
avg_loss observed. Critically: this would eventually happen on *any* static
seed given enough repetitions, including a geometrically fair one — layout
fairness only changes how quickly collapse happens, not whether it happens.

## Two distinct candidate fixes (not mutually exclusive)

1. **Generator-side fairness constraint** — add a `min_player_enemy_distance`
   (or similar) to the initial layout generator/mutator, mirroring the
   existing `min_player_goal_distance`. Addresses root cause 1 (gets an
   initial seed into the admit band faster / more reliably) but does nothing
   for root cause 2.

2. **Spawn randomization on reset** — instead of `reset()` always restoring
   the parsed layout's fixed spawn cells, sample player/enemy/goal positions
   fresh each episode from the layout's walkable free-space, keeping
   walls/traps fixed. Addresses root cause 2 directly, forcing genuine
   spatial generalization instead of single-trajectory memorization. Larger
   design decision: would require Rust-side changes to `World::reset()`
   and/or `layout.rs` to support sampling valid positions rather than always
   using the parsed originals.

## Open questions (not yet decided)

- Does spawn randomization happen *within* one entry's lifetime (same
  layout/walls, new random spawns each episode), or does POET's own mutation
  mechanism become responsible for spawn diversity instead?
- If per-episode spawn randomization is added, does it change what "the
  environment" means for POET's own bookkeeping (is a layout still one
  `EA_list` entry if its spawns vary episode-to-episode, or does that
  effectively turn one entry into a distribution over related environments)?
- Should the minimal-criterion eligibility check itself have an escape hatch
  (e.g. a hard iteration cap after which mutation is attempted regardless of
  MC status) to prevent permanent deadlock even if root cause 2 isn't fully
  addressed?

## Status

Diagnosed, not yet fixed. Next step: decide whether to pursue fix (1), fix
(2), or both, before resuming a long POET run — further training time alone
will not resolve this.