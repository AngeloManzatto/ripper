# POET Pipeline — Roadmap

## Completed (v1 pipeline)
- [x] Rust engine: generate_layout, is_layout_reachable (BFS), 
      generate_valid_layout / mutate_valid_layout (max_attempts, panic-loudly)
- [x] Full mutation redesign: independent axes for walls, enemies, traps,
      and entity repositioning (WeightedIndex-based Add/Remove/Nothing dice
      rolls with proper ceilings, replacing an earlier saturating design)
- [x] Python bindings for the above (PyGenerationConfig, PyMutationConfig)
      to the same interface quality as the rest of the pyo3 layer
- [x] BaseAgent(ABC) contract: act, train, set_eval_mode, set_train_mode,
      save, load — proven algorithm-agnostic across two real implementations
- [x] DDQNAgent — full implementation, persistent epsilon/replay
      buffer/train_step_count as agent state, checkpointing
- [x] PPOAgent — full implementation (rollout collection, GAE, clipped
      surrogate objective, multi-epoch updates), same contract as DDQN
- [x] evaluate() / satisfies_minimal_criterion() — fully agnostic, work
      identically for either agent type
- [x] run_lineage() — single-lineage mutate/train/evaluate/accept loop,
      with checkpointing on every acceptance plus a final save
- [x] Diagnosed and fixed a severe over-training/overfitting bug
      (train_episodes_per_gen * rollout_steps was ~1M+ steps per generation
      on one unchanging layout) — confirmed fix via real before/after results
- [x] MILESTONE: full 200-generation DDQN lineage, archive_size=31
- [x] MILESTONE: PPO lineage producing consecutive real acceptances after
      the over-training fix (Gen 33, Gen 39, ...)
- [x] Clear, separated verbose logging (current_layout vs candidate_layout)
      after resolving a real training-vs-evaluation-win-rate confusion

## Near-term (deferred, documented in POET_ISSUES.md)
- [ ] Mastery-based early stopping (agent.is_mastered(threshold), skip/shorten
      training once current_layout is already mastered)
- [ ] Enemy-spawn-distance structural fairness floor
- [ ] Re-examine n_repositions / mutation-difficulty-calibration with clean data
- [ ] Re-examine Minimal Criterion band against a realistic noise floor

## Future architecture (deliberate future phases, not gaps)
- [ ] Multiple parallel lineages + cross-lineage agent transfer (real POET's
      actual mechanism — this project only ever ran a single lineage)
- [ ] Quality-diversity archiving (organize accepted layouts by behavioral
      dimensions, not a flat list)
- [ ] A learned generator agent trained directly on the Minimal Criterion
      signal (PCGRL-style), replacing the hand-coded random mutator
- [ ] Give the enemy real (fog-of-war or trained) behavior in POET runs,
      instead of the current random fallback — would surface the
      spawn-distance and pursuit-evasion fairness questions for real