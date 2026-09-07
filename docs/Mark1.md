[x] - ECS redesign — moved from a Grid/Cell/Entity struct system to real ECS (entities are just ids, data lives in component maps: positions, kinds)
[x] - Layout parser — parse_layout(&str) -> World, single-character level format, panics loudly on ragged/unknown input
[x] - Terminal + visual rendering — print_world (terminal) and draw_world (macroquad window), both querying the same component maps
[x] - Environment trait — a general step/reset/observation/is_done contract with associated Action/Observation types, so future games can share the same shape
[x] - Multi-enemy support — get_entity_ids_by_kind() as a general query, no more single enemy_id
[x] - Walls, Traps, despawn — walls block movement (is_walkable), traps kill anyone who steps on them (player loses, enemy despawns), despawn_entity() as a reusable engine capability
[x] - Reward-free contract — step returns { done, reason: Option<EndReason> } only; reward is explicitly a training-side concern, not the environment's
[x] - Timeout as a real environment rule — World tracks its own tick/max_tick, not just a Python training-harness cap
[x] - Live play — keyboard input wired into the visual binary, is_key_pressed (edge-triggered), R to reset
[x] - "Stale state" fix — step() refuses to act once done is true, until reset()
[x] - Clean pyo3 boundary — core engine (grid_world/) is 100% pyo3-agnostic; a separate src/python/ module holds PyWorld, PyObservation, PyStepResult, PyAction as translation-layer wrapper types with From conversions
[ ] - Rewrite training scripts (q_learning.py, dqn.py, ddqn.py) against the new PyWorld/PyAction/PyObservation contract
[ ] - Per-entity fog-of-war observation (currently deferred — v1 observation is global/labeled, not per-viewer)
[ ] - Full chase/fog-of-war behavior for enemies (currently random movement, chase logic designed but not yet ported)
[ ] - ActionSource trait / game-loop abstraction (human vs. agent-driven input — designed in discussion, not yet built)
[ ] - Sprites instead of colored rectangles in the visual renderer