# POET with NEAT, DDQN and PPO: what we built and what we learned

Nov 14, 2026 · @Angelo

## Summary

POET works as a machine with a PPO learner, but in this game its curriculum adds nothing measurable beyond training on many layouts.

We rebuilt POET (a player and an enemy per layout, layouts mutated by the Rust engine) as a tested `poet/` library and plugged in three learners: NEAT, DDQN and PPO. The main findings:

- **The learner's speed decides the pool's fate.** A pair only helps POET while it sits inside the criterion band. DDQN learns fastest but stays in the band for 1-2 rounds and its pool never grew. PPO stays 4-7 rounds and fills a pool of five by generation 7. NEAT needed 9 rounds (about 80 minutes) just to become eligible.
- **PPO is the practical learner.** Ten generations took 7.7 minutes; NEAT took roughly half an hour once its pool had five entries. With the wall cap lifted to 0.6, the pool reached the engine's density limit (176 walls) in 17-19 generations in all three variants we ran.
- **The pool settles at an equilibrium.** The player ends near 0.66 and the enemy near 0.26 whether the player trained 150 or 300 games per call and the enemy 300 or 600. About half of all mutations are rejected because the player is already above 0.7.
- **No evidence that the pool beats plain training.** On unseen dense layouts, a player trained on many random dense layouts scored 0.79-0.82, pool players 0.74-0.79, and a player trained on one layout 0.62-0.70. Enemies were indistinguishable.
- **The testbed is too easy.** Every layout family we tried is solved to about the level of the hand-written heuristic within minutes of training.

Bottom line: the machinery is sound and fast, but the claim that POET's curriculum helps is not supported here. Testing it needs a harder task. Update from Angelo's finding: the generator guarantees a path from the player to the goal but not from the enemy to the player, so on dense layouts the enemy is often sealed in a pocket. Several dense-layout findings below may be partly artifacts and should be re-run after the generator is fixed.

## What we built

A library that runs POET with any learner, plus two new learners (DDQN and PPO), each delivered with tests and a matching pipeline block.

- **The `poet/` library.** An `Entry` holds a layout and one learner per role ("player", "enemy\_0"). A learner has three methods: `agent()`, `train(layout, opponent, rng)` and `clone()`. `run_poet` then mutates layouts, tests children with the minimal criterion (player and enemy rates in \[0.2, 0.7\], timeouts under a cap), admits them, trains every entry one round, and runs transfer rounds.
- **Evaluation helpers.** `held_out_layouts`, `skill_on_layouts` and `player_outcomes_on_layouts` measure an agent against a fixed opponent on layouts it never saw, so two agents can be compared on the same games.
- **DDQN.** `DdqnAgent` is a Double DQN with a replay buffer, a target network and a decaying epsilon, plus `DdqnLearner`.
- **PPO.** `PpoAgent` uses a shared-body actor-critic, GAE advantages, the clipped objective and an on-policy rollout of 512 steps, plus `PpoLearner`.
- **A fair comparison.** DDQN and PPO see exactly what the NEAT agent sees (its `build_input`) and use the same reward, so differences come from the learning rule.
- **Tests.** DdqnAgent 51 checks, DdqnLearner 25, PpoAgent 49, PpoLearner 26 (five more real-engine tests per learner run on Angelo's machine). We also planted deliberate bugs in each module; the tests caught all of them.
- **Design rules learned the hard way.** Engine observations cannot be copied or pickled, so `agent()` returns a fresh frozen clone each call, training happens in place, and a learner never holds a played agent. The server run script got a settings block, atomic checkpoints and a preflight check of the wall cap.

## Results by algorithm

The three learners differ most in how long a pair stays inside the criterion band, not in how fast it learns.

|  | NEAT | DDQN | PPO |
| --- | --- | --- | --- |
| Training games per call | about 10,000 (6-10 min) | 300 | 300 |
| Rounds until the pair is eligible | 9 (about 80 min) | 1 (43 s) | 4 |
| Rounds inside the band | not measured | 1-2 | 4-7 |
| Pool in POET | grew; about two thirds of children accepted; transfer collapsed it to 2 distinct agents | 3 generations: could not grow after round 1 | full by generation 7; stable for 40 generations |
| Time | about 30 min per generation (pool of 5) | 3.4 min for 3 generations | about 1.5 min per generation (pool of 5) |

The criterion rate of the co-trained pair, per training round, on the base layout (same seeds for both learners):

| Round | DDQN player | DDQN enemy | PPO player | PPO enemy |
| --- | --- | --- | --- | --- |
| 1 | 0.39 | 0.48 | 0.04 | 0.34 |
| 2 | 0.68 | 0.23 | 0.03 | 0.65 |
| 3 | 0.69 | 0.19 | 0.17 | 0.67 |
| 4 | 0.79 | 0.10 | 0.24 | 0.61 |
| 5 | 0.70 | 0.11 | 0.23 | 0.66 |
| 6 | 0.79 | 0.12 | 0.46 | 0.45 |
| 7 | 0.67 | 0.22 | 0.47 | 0.48 |
| 8 | 0.71 | 0.15 | 0.72 | 0.28 |

Three things explain the table:

1. **The enemy's criterion rate is not independent.** It equals 1 minus the player's goal rate minus timeouts, so a strong player mechanically squeezes it. Measured against a fixed player, the DDQN enemy kept learning (caught share 0.08 to about 0.45 and then flat) while its criterion column fell.
2. **DDQN's player is partly tuned to its own enemy.** It scored 0.7 against its co-trained enemy but only about 0.35 against the heuristic. PPO's player scored the same against both. Why is a hypothesis: replay keeps old opponents in the data, while on-policy learning tracks the current one.
3. **A fresh DDQN enemy trained only against the final player reached 0.26-0.30 after 8 calls**, against 0.16 for the co-trained one. So the player was mildly tuned to its enemy, but mostly just hard to catch.

## PPO inside POET

The PPO pool reaches the engine's wall-density limit and then stresses at the player's upper bound, regardless of the training budget.

| Run | Wall cap | Generations | Games per call (player / enemy) | What happened |
| --- | --- | --- | --- | --- |
| 1 | 0.3 | 10, then 30 more | 300 / 300 | First eligible parent at generation 5; pool of 5 by generation 7; walls 86, 94, 102, 110, then stuck at 118 (the cap) from generation 4 of the continuation |
| 2 | 0.6 | 20 | 300 / 300 | Walls 118 to 176 (the limit) by generation 17; children accepted fell from 15/15 to 2/6; eligible parents fell from 5 to 2 |
| 3 | 0.6 | 20 | 300 / 600 | Same pattern; 33.5 min against 27.4 |
| 4 | 0.6 | 20 | 150 / 300 | Same pattern; 23.4 min |

The last five generations of runs 2-4, which is where the stress shows:

| Run | Eligible parents (of 5) | Children accepted | Player rate | Enemy rate | Whole-run acceptance |
| --- | --- | --- | --- | --- | --- |
| 2 (300 / 300) | 3.4 | 49% | 0.670 | 0.240 | 72% |
| 3 (300 / 600) | 4.0 | 40% | 0.659 | 0.267 | 70% |
| 4 (150 / 300) | 4.4 | 38% | 0.659 | 0.263 | 71% |

What we found:

- **Rejections are on the player side.** Of 60 mutation attempts on the final pool, 32 were accepted. Of the 28 rejected, 27 had the player above 0.7 (14 of them also with the enemy below 0.2, which follows mechanically) and one had only the enemy too weak.
- **The pool sits at an equilibrium near the upper bound.** The player ends at about 0.66 in every variant, so about half of the mutations push it above 0.7. Changing the training budget moved the early generations slightly but not the end state.
- **Dense layouts are harder to catch on.** Against `frozen_player`, the enemies' caught share halved from 0.48 (118 walls) to 0.24 (176 walls), while the player's goal rate against the heuristic stayed near 0.7. Caveat: `frozen_player` was trained on the base layout, so it stalls on dense ones (timeouts 0.33 to 0.53).
- **A fresh enemy trained only on a 176-wall layout plateaued at about 0.32** after 9,000 games, against 0.20-0.26 for the pool's enemies. That comparison favored the fresh enemy, which trained against the very yardstick it was measured on.
- **The heuristic is useless as a chaser control.** It catches only 2-6% of games on both pools, so it cannot show whether dense layouts are hard for any chaser.

## Does the pool produce anything useful?

Not beyond what plain layout variety gives. Pool players generalize no better than players trained on a rotating set of random dense layouts.

All players below were measured against the same heuristic enemy on the same 20 held-out dense layouts (about 176 walls, none seen in training), 20 games each.

| Trained how | Training games | Mean goal rate | Worst layout |
| --- | --- | --- | --- |
| Untrained | 0 | 0.15 | 0.00 |
| One dense layout, direct (one seed) | 18,000 / 36,000 / 54,000 / 90,000 | 0.68 / 0.70 / 0.62 / 0.62 | 0.25-0.35 |
| Pool lineage A, 5 players | 18,000 each | 0.78 0.78 0.81 0.78 0.80 (mean 0.79) | 0.35-0.55 |
| Pool lineage B, 5 players | 18,000 each | 0.70 0.73 0.82 0.72 0.72 (mean 0.74) | 0.20-0.60 |
| Pool lineage C, 5 players | 18,000 each | 0.78 0.77 0.67 0.71 0.78 (mean 0.74) | 0.30-0.55 |
| 20 rotating dense layouts, no curriculum | 18,000 / 36,000 / 90,000 | 0.79 / 0.80 / 0.82 | 0.55-0.65 |
| Hand-written heuristic (reference) | none | 0.77 | 0.50 |

How the argument goes:

1. **Equal games, one layout.** The pool's players beat the single-layout control in all three lineages (by 0.06-0.11), but only one lineage cleared the 0.1 margin we set in advance. The verdict was "not demonstrated".
2. **More compute doesn't help the single-layout control.** At 90,000 games, about the compute the pool consumed in total, it sat at 0.62. The pool's edge was not just extra compute.
3. **But the control trained on one layout and was tested on unseen ones.** A control that rotates through 20 dense layouts, with no curriculum and no criterion, reached 0.79 after 18,000 games and 0.82 after 90,000, level with or above every lineage. The pool's advantage over the first control was layout variety.
4. **The enemy role shows nothing either.** Against a heuristic player on the same layouts, every trained enemy caught 0.12-0.15 (direct 0.12, rotating layouts 0.14, all 15 pool enemies 0.12-0.15), against 0.04 for an untrained one. That yardstick looks saturated, so it cannot separate them.

## Probing the testbed

We found no layout family where trained players fall far below the hand-written heuristic, except traps, and traps were learned in about 3 minutes.

Goal rate against the heuristic enemy on 20 fresh layouts per family (16x16 grid, 1 enemy, generator base settings: wall density 0.1, 0 traps, goal distance 10):

| Family | Randomized control (90k) | Pool lineage A, entry 2 | Heuristic | Untrained |
| --- | --- | --- | --- | --- |
| Base settings | 0.89 | 0.88 | 0.90 | 0.07 |
| Wall density 0.3 | 0.68 | 0.60 | 0.61 | 0.05 |
| 3 traps | 0.79 | 0.66 | 0.88 | 0.04 |
| 6 traps | 0.73 | 0.61 | 0.82 | 0.03 |
| No walls | 0.95 | 0.89 | 0.96 | 0.09 |
| Goal distance 14 | 0.90 | 0.84 | 0.90 | 0.05 |

- **The pool's player generalizes worse than the randomized one on every family**, by 0.01 to 0.13. This is one lineage entry, so it counts as consistent, not proven.
- **Traps are the only real gap**, and they were absent from training. Fine-tuning a copy of the randomized control for 30 rounds on layouts with 3 and 6 traps lifted it to 0.86 and 0.82, level with the heuristic (0.88 and 0.82).
- **Fine-tuning has a cost.** The same 30 rounds lowered the base family from 0.89 to 0.81 and wall density 0.3 from 0.68 to 0.60. With one seed and about 0.04 of noise this is suggestive only, but it is the standard forgetting effect of one agent that keeps training on new layout types.
- **A harness limit.** Layouts with 2 enemies crashed ("1 enemy agents for 2 enemies"): `play_games` supports only the `enemy_0` role. The generator itself makes them fine.

## Conclusions

The pipeline results are solid; the claims about why POET did not help are mostly hypotheses.

| Claim | Evidence | Confidence |
| --- | --- | --- |
| How long a pair stays in the band depends on the learner (DDQN 1-2 rounds, PPO 4-7, NEAT 9 to enter) | Same table and seeds for DDQN and PPO; NEAT from the baseline run | Solid for DDQN and PPO; one seed each |
| PPO runs POET stably and fast | Three 20-generation runs plus a 40-generation run; all reached the density limit and kept 4-5 eligible parents | Solid |
| The pool's end state is an equilibrium that training budget does not move | Player about 0.66 and enemy about 0.26 in three budget variants | Likely; one seed each; may be partly caused by sealed-off enemies (see Caveats) |
| The criterion's enemy rate is capped by the player's | It equals 1 minus the player rate minus timeouts | Solid (arithmetic) |
| Wall density is not a difficulty axis for this pair | Enemy catch ceiling falls on dense layouts, player skill stays flat | Hypothesis; probably explained by enemies sealed off from the player (see Caveats) |
| The pool does not beat plain training on unseen dense layouts | Randomized control 0.79-0.82 against pool 0.74-0.79; single-layout control 0.62-0.70 | Likely for the player; one control seed; the enemy yardstick is saturated |
| Pool players generalize slightly worse than randomized ones | Lower on all six families, by 0.01-0.13 | Hypothesis; one lineage entry |
| DDQN's player is tuned to its own enemy | 0.7 against its enemy, 0.35 against the heuristic; PPO shows no such gap | Hypothesis; the replay-versus-on-policy explanation is untested |

What this means for the project: algorithm choice matters for POET mainly through how long a co-trained pair stays inside the criterion band. PPO gave the widest and most stable band; why (its on-policy updates, its pace) we have not tested. On this game, however, the problem is easy enough that a curriculum has little to add.

## Caveats

Read the numbers as patterns, not rankings: most results come from a single seed per setup and budgets that differ between learners.

**Biggest caveat, found after the first draft: enemies can be sealed off.** The layout generator checks that the player can reach the goal, but not that the enemy can reach the player. On a 177-wall layout Angelo pasted, the open cells split into 21 separate regions; the enemy sits in a 6-cell pocket and the player in a 19-cell region, so it can never catch the player, and only about 12% of random pairs of open cells can reach each other. On dense layouts this plausibly explains the enemy's falling catch rate, the timeouts, the pool settling near the player's upper bound, and why the dense-layout game looks easy: in many games it is a one-agent navigation task. It has not been quantified across the pools yet. Wall counts in this document include the 60 border cells (176 = 60 border + 116 interior; the cap 0.6 is interior walls divided by 196).

- **One seed per setup.** Every learner run, every control and every budget variant. Where three variants agree (the equilibrium), that is some replication; elsewhere there is none.
- **Budgets are not comparable across learners.** NEAT plays about 10,000 games per call, DDQN and PPO 300. So these results cannot rank the algorithms on sample efficiency.
- **Measurement noise.** Rates from 100 games carry about ±0.05; means over 400 games (20 layouts × 20 games) about ±0.03; per-cell values in the fixed-yardstick tables with 50 games about ±0.07.
- **Yardsticks are imperfect.** `frozen_player` was trained only on the base layout and stalls on dense ones. The heuristic enemy is a floor-level chaser. The heuristic player is hand-written, so it is a reference, not an upper bound.
- **The test favors the randomized control.** It trained on layouts from the same generator as the held-out test family, while the pool lineages only ever saw mutations of one starting layout. A different test family could change the gap.
- **The compute comparison is approximate.** Each pool entry trained 18,000 games, but the pool trained five entries per generation plus the eligibility tests, so its total compute was several times higher than one control.
- **The NEAT baseline ran at an effective wall cap of 0.3**, not the 0.45 it was thought to have. A first 0.6-cap run was invalid for the same reason (a stale `mutation_config`), which is why the run script now prints a preflight check.
- **Predictions we got wrong.** We expected the PPO pool to drift out of the band from the enemy side (it settled instead), and we expected the player handicap to lower the player rate (it did not).

## Ideas to discuss

The most useful next step is to make the generator and the mutations keep the enemy connected to the player, then re-run the key experiments; after that, a task where plain training fails, so that POET's claim can be tested at all.

1. **A harder task.** Keys and doors (subtasks, longer horizon), several enemies, or larger mazes. Several enemies first needs the harness to pass one agent per enemy (roles `enemy_0`, `enemy_1`, ...). The test stays the same: confirm that randomized training scores low on the new task, then ask whether the pool beats it.
2. **A different claim: populations.** Fine-tuning one agent on a new layout type cost about 0.08 elsewhere. Does a pool of specialists plus selection keep more skill across layout families than one agent that keeps training? We can measure that with what we have.
3. **A better criterion.** Measure the enemy against a fixed yardstick instead of through the player, and decide on purpose where the player's upper bound should sit, since the pool settles near 0.66.
4. **Mutations aligned with difficulty.** Check what `mutate_layout` can change (traps, enemy position, goal) and whether any of it moves difficulty; walls did not. This is an open question about the engine.
5. **Diversity of the pool.** Pairwise layout distances and a player-by-enemy cross-play matrix, to see whether transfer collapses the pool (it did for NEAT). The code is written; its output has not been collected.
6. **Replication.** At least two more seeds for the controls and for one PPO pool run, before any claim is called settled.
7. **Back to DDQN.** Test whether capping training per POET step (so children are tested while still balanced) lets its pool grow, and whether resetting epsilon on clones helps on new layouts.
8. **Connection to RIPPER.** In-game adaptation and a Dungeon Master that designs layouts are natural homes for these learners and for the evaluation tools built here.

Questions for you as the pilot: which task would you add first, do you want to keep curriculum (POET) as the main claim or switch to the population idea, and how much of this goes into a paper-style write-up?
