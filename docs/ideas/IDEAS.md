# RIPPER — Ideas Registry

A living list of the ideas for where the engine and the research can go. It is meant to be edited: add an idea when it shows up, move its status when something happens, and write down what was decided and why.

- **Started:** 2026-11-20 (the first eight ideas came from a long walk and were written down the same day).
- **Voice:** *"Yours"* marks what Angelo said. *"Addition"* marks something added while writing this file, which he may keep, change or throw away. *"Source"* marks something checked against a paper or a page (links at the end).
- **Where to keep it:** `docs/IDEAS.md`, next to `POET_ISSUES.md` and `POET_ROADMAP.md`.

---

## How to maintain this file

**Status values** (one per idea): `seed` (just noted) → `exploring` (reading, small experiments) → `planned` (has a first step and a place in the roadmap) → `building` → `done`. Also `parked` (on purpose, with the reason written down) and `dropped` (with the reason).

**Rules for editing**
1. Never delete an idea. Change its status and write the reason.
2. Each idea has the same sections: *Idea*, *Why it matters*, *Related work*, *Additions*, *Needs from the engine*, *First small step*, *Open questions*.
3. Keep "First small step" small enough to do in one sitting. If it is not, split the idea.
4. Add a line to the **Changelog** at the bottom whenever something changes.

---

## Index

| ID | Idea | Status | Needs first |
|----|------|--------|-------------|
| I-01 | From a generic grid world to a roguelike | seed | items, floors, a look |
| I-02 | Entities as symbols (many cells) | seed | footprints, rendering |
| I-03 | A Dungeon Master that invents mechanics | seed | data-driven definitions (building) |
| I-04 | Fixed 64×64 or 128×128 world, many entities | seed | `Grid<T>` (done), performance plan |
| I-05 | Graph of sub-tasks to finish the game | seed | events, entities, door and key |
| I-06 | A Rust ML sub-package | seed | stable `Brain` interface |
| I-07 | Memory as a graph that entities write to files | seed | I-06, I-05 events |
| I-08 | Contribute to RL research | ongoing | rigorous experiments |
| I-09 | A learned generator for entity symbols | seed | I-02 footprints, I-06, a way to score symbols |

## Foundations that unlock the ideas (current engine work)

| Piece | State | Unlocks |
|-------|-------|---------|
| Definitions loader (packs, JSON, validator, `Registry`) | done, tested, shipped `base` pack loads | I-03, I-05 |
| `Grid<T>` (generic 2-D container) | done (`get`, `get_mut`) | I-04, rendering |
| Glyph lookup in the registry | next | building terrain from text |
| `World` built from the registry (terrain grid, trap and goal bridged) | planned | every idea |
| Rendering from definitions | planned (the visible goal) | I-01, I-02 |
| Entity definitions (player, enemy, trap, goal, door, key) | planned | I-02, I-03, I-05 |
| Events (picked up, stepped on, killed) | not started | I-05, I-07 |
| Single-region and reachability-with-keys rule | not started | I-03, I-05 |

---

## I-01 — From a generic grid world to a roguelike

**Status:** seed

**Idea (yours).** Evolve the engine from a generic `grid_world` into a RIPPER roguelike-style game, with an identity of its own.

**Why it matters.** A genre gives the engine a set of expectations (turns, glyphs, fog of war, procedural levels) that guide what to build next and what to leave out. It also fits the project's pieces already: turn-based stepping, fog of war, procedural layouts, enemies, traps.

**Related work.** NetHack is the classic of the genre and the base for several RL environments (NLE, MiniHack); Craftax mixes elements of Crafter and NetHack and is a fast RL benchmark (source: Craftax page).

**Additions.**
- The usual roguelike ingredients, and where each stands: procedural levels (have), turn-based (have), fog of war (have), permadeath (have, one life per episode), items and an inventory (missing, and the door/key is the first), depth with several floors (missing), glyph-first display (missing, planned with rendering).
- A "run" could be a sequence of floors with the agent's knowledge carried over (links to I-07).

**Needs from the engine.** Items and an inventory component; floors; the rendering work.

**First small step.** Write down, in one page, which roguelike features are in and out of scope (no combat stats, no items beyond keys, etc.), so it does not grow by accident.

**Open questions.** Is it real-time or strictly turn-based for the visual driver? How many floors before it stops being "one level" and starts being a campaign?

---

## I-02 — Entities as symbols (many cells)

**Status:** seed

**Idea (yours).** Entities should become symbols made of several cells, not a single cell. Fascinated by how a model learns symbol representations.

**Why it matters.** With one character per entity, the identity of a thing is a lookup. With a multi-cell pattern, the model has to *recognise a shape* and find out what it means, which is closer to perception in ARC-AGI-3 (frames and nothing else) and to how people learn glyphs.

**Related work.** ARC-AGI-3's interactive frames (already studied in the project); object-centric representation learning in general.

**Additions.**
- The definitions design already has a place for it: *footprints* were on the "later" list. A footprint is a list of cell offsets, each with its own tile or glyph.
- An experiment that isolates the idea: **permute the glyph-to-meaning mapping in every level** (the per-level palette makes this almost free). The agent can no longer memorise "`#` means wall" and has to learn meaning from interaction. A control run with fixed glyphs shows the cost.
- A second experiment: symbols whose *shape* carries information, for example a key and a door with matching shapes.
- Rendering must support a footprint as one thing (select, colour, move together).

**Needs from the engine.** Footprints in entity definitions; collision and line of sight for multi-cell entities; a rendering path.

**First small step.** Sketch (on paper or in a JSON example, no code) how a 2×2 entity would be described in a definition file.

**Open questions.** What happens at the edges: an entity half in fog? Does the agent see symbols as raw cells or through a learned encoder? Who makes the symbols: a person, or a learned generator (see I-09)?

---

## I-03 — A Dungeon Master that invents mechanics

**Status:** seed (it grows out of the earlier "Dungeon Master as a layout creator" idea)

**Idea (yours).** A dungeon master able to create new mechanics, and new entities if possible.

**Why it matters.** The current mutator only moves walls, enemies and traps. A DM that can add *rules* makes the environment space open-ended, which is the point of POET-style work.

**Related work.**
- *Unsupervised Environment Design* and PAIRED: an adversary generates environments, rewarded for the regret between two agents, which yields a curriculum of increasing complexity (source).
- *OMNI-EPIC*: foundation models write the code for new tasks, including environments and reward functions, judge what is interesting and novel, and adapt difficulty to the agent's progress (source).
- *Voyager*: a growing library of executable skills and an automatic curriculum (source).
- POET (Wang et al. 2019), already read, for the minimal-criterion filter.

**Additions.**
- **Levels of power**, from safest to most open: (1) parameters of existing mechanics; (2) layouts; (3) combinations of existing effects in a new entity; (4) new effects.
- **The key design choice: the DM outputs data, not code.** OMNI-EPIC lets a model write code. In RIPPER the DM can write *pack files* (JSON) that the same strict loader and validator check. A bad idea is rejected with a clear error instead of crashing the engine. That makes the strict validator from the definitions work a safety layer for I-03, not just tidiness.
- **Vocabulary first.** A DM can only combine what the engine knows. Effects and conditions (tags, states, triggers) have to exist as a small, fixed vocabulary before the DM can combine them.
- **Admission test for anything the DM makes:** valid by the validator, solvable by the reachability rule, and passing the POET minimal criterion (not trivial, not impossible).
- A natural training signal for the DM is regret (PAIRED) or the minimal-criterion band (POET). It also connects to the earlier idea of balancing by *start positions as handicaps* for strong and weak agents.

**Needs from the engine.** Tags/states/effects in definitions; a solvability check that understands doors and keys; an interface where the DM proposes a pack.

**First small step.** Hand-write a second pack (say `dungeon:sticky_floor`) as if the DM had made it, and check that the loader accepts it and rejects broken variants.

**Open questions.** How does the DM learn which combinations are *interesting* rather than merely valid (OMNI-EPIC uses a model of interestingness)? Who judges, a score, a human, a model?

---

## I-04 — Fixed 64×64 or 128×128 world with many entities

**Status:** seed

**Idea (yours).** Keep the grid size flexible for now, then move to a fixed 64×64, or better 128×128, view with many enemies, challenges and puzzles.

**Why it matters.** Small rooms make planning shallow. A large world forces long-horizon behaviour, exploration and memory.

**Related work.** Large procedurally generated worlds are the norm in open-ended benchmarks such as Craftax and NetHack-based environments.

**Additions.**
- **Sizes in perspective:** a 128×128 grid has 16,384 cells, about 32 KB with 2-byte terrain ids. Storage is not the problem. The costs are per-entity work (line of sight, fog of war, pathfinding every tick) and what the *agent* receives.
- **Do not hand the model the whole map.** The engine already gives each entity a perceived grid limited by its perception range. At 128×128 keep that: the observation stays a local window (or a downsampled map), and the long-range structure has to go through memory (I-07).
- **Fixed size is a decision for the observation, not for the engine.** The engine can keep arbitrary `Grid<T>` sizes; only the *view* given to a network needs a fixed shape.
- Flat storage in `Grid<T>` already helps, and the same container can hold the terrain, the discovered cells, distances and visited sets.
- Chunking is the natural bridge to the far-future "metroidvania of connected grids".

**Needs from the engine.** A profile of the per-tick cost with many entities; sparse structures for entities; a cheaper fog-of-war update (only near entities that moved).

**First small step.** A benchmark that steps a 128×128 world with, say, 50 random entities and prints ticks per second. No optimisation, just a number to compare against later.

**Open questions.** Is the view centred on the agent, or fixed to the map? What is the right trade-off between a local window and a downsampled global map?

---

## I-05 — A graph of sub-tasks to finish the game

**Status:** seed

**Idea (yours).** Define a graph of sub-tasks the player must complete before finishing: take the key, step on a tile, kill enemy X, move to the goal; or kill enemy A, find player B, move to the goal. Many combinations, much diversity.

**Why it matters.** It turns "reach the goal" into structured, multi-step reasoning, and lets the DM (I-03) generate diversity by changing the graph instead of only the map. It is also how you learn what it takes to bring an agent to that kind of reasoning, which was a stated goal.

**Related work.**
- *Subtask graphs* (Sohn et al.): a task as subtasks with rewards and AND/OR dependencies; the agent must find the best order. Their graph solver generalises zero-shot to unseen and larger graphs (source).
- *Reward machines* (Icarte et al.): a finite-state machine that defines the reward while exposing its structure, so a task can be specified at a high level and split into parts (source).

**Additions.**
- **Objectives are data.** The graph is a small section of the level file: nodes (conditions such as `has_item key`, `entered tile`, `killed entity`, `reached goal`) and edges (AND/OR). The loader validates it like everything else.
- **The engine reports events, the objective tracker consumes them.** `picked_up`, `entered`, `killed`. The tracker says which nodes are complete. This keeps your earlier decision that *reward is not the environment's concern*: the engine says what happened and what is complete; turning that into reward happens in training.
- **Solvability check on the graph:** every node must be reachable given the map (the reachability-with-keys rule), and the graph must have no node that can never be satisfied.
- The agent needs to *see* progress. Either part of the observation shows which sub-tasks are done, or it must remember. The two options make a good ablation.
- A reward machine is a good mental model for the tracker: states are "which sub-tasks are done".

**Needs from the engine.** Entity definitions (key, door, an enemy with an id); an event stream; the objective section in levels.

**First small step.** Draw by hand the graph for "take the key, open the door, reach the goal" and the one for "kill A or kill B, then reach the goal", in the JSON you would like to write. No code yet.

**Open questions.** Can enemies also have objectives (their own graph)? Can the graph change during play (DM interventions)?

---

## I-06 — A Rust ML sub-package

**Status:** seed

**Idea (yours).** Build a sub-package in Rust dedicated to ML (NEAT, ANN, cellular automata, kNN and others) that can be used to train the entities.

**Why it matters.** It keeps the engine self-contained: entities can think without Python, brains can be saved and run inside the visual driver, and it matches the plan to export policies for standalone inference. Cellular automata are also already on the list for level generation.

**Related work.** Rust roguelike toolkits split things into small crates (for example geometry, pathfinding, random, noise), where algorithms talk to the map through traits instead of knowing its type (source: bracket-lib).

**Additions.**
- **A separate crate in a Cargo workspace** (say `ripper_ml`) that does **not** depend on the engine. The engine depends on it, never the other way. Same layering rule as the folders: foundation underneath.
- **One small interface for every brain.** For example: something that takes a slice of numbers and returns an action index, plus `save` and `load`. NEAT genomes, a fixed ANN, kNN over remembered situations and a CA rule would all fit behind it. The engine would not care which one an entity has.
- **Serialisation from day one** (JSON through serde, like the definitions), because I-07 needs brains and memories on disk.
- **Determinism:** every algorithm takes a seed and uses no global state, so runs reproduce and the golden-test habit carries over.
- Python can remain the place where heavy training happens (DDQN, PPO), while the Rust package covers cheap, evolvable brains.

**Needs from the engine.** Nothing yet; this is a new crate. It needs a stable shape for "inputs for an entity" (features) and "outputs" (actions).

**First small step.** Define the brain interface and implement the simplest brain that is not random (for instance a fixed table or a tiny perceptron) with tests.

**Open questions.** How do features get built in Rust (the Python `agents/features` toolkit has this already)? Which algorithm to port first?

---

## I-07 — Memory as a graph that entities write to files

**Status:** seed

**Idea (yours).** Give entities knowledge memory: they learn as a graph while playing, store the graph they uncover in files, and evolve together as an ecosystem, like Rain World. He does not think a single simple ML algorithm beats all.

**Why it matters.** It moves learning from the weights of one network to a shared, inspectable store of what has been discovered, which is closer to how knowledge accumulates across lives and generations. It also makes behaviour explainable, because a graph can be read.

**Related work.**
- *Voyager*: knowledge grows as a library of reusable skills that can be reused in a new world (source).
- Reward machines and subtask graphs show that *structure* (states and dependencies) can be exposed and learned (source).
- *Rain World* (the game you named): its creatures have survival goals of their own and are not there only to challenge the player; there is a food chain with the player in the middle of it; and the world keeps simulating when the player is not watching (the level designer's words: "outside of your frame they're all still alive"). The designers also note that the unpredictability can produce unwinnable situations, which they managed through level layout and where each creature likes to live (source: Game Developer article).

**Additions.**
- **What a node and an edge could be** (decide before building):
  - *Places and events* (a map of "where things are" — nodes are landmarks, edges are paths);
  - *Cause and effect* (node = situation, edge = "action X here led to Y"), the kind of learned mechanics graph;
  - *Skills* (node = reusable behaviour, edge = precondition), as in Voyager.
  Start with one.
- **The file is the interface.** JSON files per entity (a "journal"), validated on load like the definitions. A bad file fails loudly.
- **Inheritance and culture:** a new entity can start from its parent's graph, or from a shared pool. This is how an ecosystem would form.
- **Forgetting and merging:** graphs grow. Without a rule for merging duplicates and pruning rarely used nodes, they become unusable.
- **What a Rain World-style ecosystem would ask of the engine (addition):**
  - Entities with goals of their own (food, shelter, avoiding a predator), not only "catch the player" or "reach the goal". That connects to the open question in I-05 about enemies having their own sub-task graphs.
  - Simulating entities **outside the player's view**. In a 128×128 world this is a real cost (I-04), so it needs a cheaper update for entities far from anything that is watching.
  - A warning to keep in mind: unpredictability can make a level unwinnable. You have already met this with dense layouts where the enemy could not reach the player. The solvability checks in I-03 and I-05 are the engineering answer, and "layout and habitat" is the design answer the Rain World designers used.
- **The test that matters:** an ablation with memory on and off over the same levels. If the memory does not help, it is not worth the complexity.

**Needs from the engine.** Events (I-05); a Rust brain interface (I-06); a persistent id for entities across episodes.

**First small step.** Pick the smallest useful graph (for example "which cells I have seen are walls") and write it to a file and read it back, with a test.

**Open questions.** What is the unit of an ecosystem: one world with many agents, or many worlds sharing a pool? How do two entities share knowledge without sharing everything?

---

## I-08 — Contribute to RL research

**Status:** ongoing

**Idea (yours).** Help improve the RL scientific research field.

**Why it matters.** It is the reason the engine is written the way it is: seeded, tested, and with the decisions written down.

**Additions.**
- **What this project is in a good position to offer:**
  1. *A data-driven, validated environment description* that a generator can write to (I-03), which makes unsupervised environment design experiments cheap to run.
  2. *Adversarial co-evolution with several roles* (player, enemy, environment) rather than one agent per environment, with start positions as handicaps for balance.
  3. *Honest methodological findings* already recorded (the exploration-floor effect between DDQN and PPO, the greedy-evaluation artifact, the unconnected-enemy layouts). Findings like these are worth writing up even when small.
- **Habits that make results trustworthy:** several seeds for every comparison, reporting negative results, a golden test for the engine, and a written note for every confound found (for example the dense layouts where the enemy could not reach the player).
- **A possible first writing target:** a short technical note on the validated, data-driven environment format and the loader's checks, since it is self-contained and reproducible.

**First small step.** Keep a `docs/FINDINGS_*.md` for every result, as is already done.

**Open questions.** Which venue and which claim? Better to decide after the engine stabilises and one clean experiment exists.

---

## I-09 — A learned generator for entity symbols

**Status:** seed (added 2026-11-20, a day after the first eight)

**Idea (yours).** Instead of a person designing or defining the symbol of each entity, have an ML generator that produces them.

**Why it matters.** It completes I-02 and I-03. I-02 says an entity is a multi-cell symbol; I-03 says a Dungeon Master may invent new entities. A new entity with no symbol is incomplete, and asking a human to draw each one would break the loop. A generator makes "invent an entity" a closed operation: it produces the rules *and* the look.

**Related work (from general knowledge, not re-checked for this file).**
- **CPPNs** (Stanley, 2007), small networks that take a cell position and output its value, evolved with NEAT. This is the closest fit, since I-06 already lists NEAT, and a CPPN can be sampled at any size, so it works for 2×2 and for 8×8.
- **Picbreeder** (Secretan et al.), where people pick which evolved CPPN images to keep. It is a model for the "human as selector" mode below.
- **Discrete-latent models** (VQ-VAE style), which learn a codebook of small patterns. Fits a generator that must output grids of a few tile types.
- **Emergent communication / learned symbols** in multi-agent RL, where agents invent the symbols they use. Related, but the symbols there carry messages and not entity identity.

**Additions.**
- **The hard part is the objective, not the generator.** Good symbols need a score. Candidates:
  1. *Distinguishable:* two entities of different kinds should look different (a distance between shapes).
  2. *Learnable:* a small classifier trained on the symbols reaches high accuracy fast.
  3. *Informative:* the shape correlates with a property (dangerous, key-like), which connects to the second experiment in I-02.
  4. *Not too easy and not too hard for the agent*, the same "minimal criterion" idea used for levels in POET.
- **Three modes, in order of difficulty:** (a) a person picks among generated candidates; (b) the generator is scored by a classifier; (c) generator and agent co-evolve, so symbols that the agent finds hard to tell apart get pushed apart.
- **A validity check fits the existing design.** The generator would write a footprint into an entity definition, and the same strict validator rejects what is malformed (wrong size, glyph clash, unknown tile). No silent fallback.
- **Seeded and recorded.** The generated symbol is written into the pack as data, so a run can be replayed and the symbol inspected afterwards.
- A fun control: show the agent the generated symbols and a shuffled set, and measure whether it learns faster with the structured one (link to the permuted-palette experiment in I-02).

**Needs from the engine.** Footprints in entity definitions (I-02); entity definitions as data (planned); a way to measure how distinguishable two symbols are; for mode (c), a stable Brain interface (I-06).

**First small step.** Without any ML: write a tiny program that fills a 3×3 grid at random with three tile types, discards duplicates under rotation and reflection, and prints ten of them. Then ask the question, "what would make me prefer one of these over another?" That is the objective, and it is the real first design decision.

**Open questions.** Does a symbol need to be symmetric, or recognisable under rotation? Are symbols fixed per entity kind for the whole run, or regenerated per level? Who decides when a generated symbol is "good enough" to enter a pack?

---

## Cross-cutting notes

**Dependency picture (rough)**

```
definitions loader (done) ──► World from registry ──► rendering ──► I-01, I-02
        │                          │
        │                          └──► entity definitions ──► events ──► I-05 ──► I-07
        │                                       │                              ▲
        └──► I-03 (DM writes packs) ◄───────────┘                              │
                                                                  I-06 (Rust brains)
Grid<T> (done) ──► I-04 (large worlds)
```

**Principles that keep recurring (from decisions already made)**
- The environment reports what happened; reward is decided in training.
- Fail loudly: no silent fallback when data is wrong.
- One small, tested step at a time, written by Angelo.
- Data over code: new content should be a file, not a change to many `match` statements.

**Parking lot (smaller thoughts to place later)**
- A metroidvania-style world of connected W×H grids (far future, chunking in I-04 is the bridge).
- Can enemies open doors? Is a key an entity or an inventory item?
- Id stability when packs change (a planned test; a lock file of name → number is one answer).
- Observation channels for the new terrain and entities.
- Entities that want things of their own (Rain World style), and entities simulated while nobody is looking.

---

## Related work and references

- Wang et al., **POET** (2019) — paired open-ended trailblazer; already studied in the project.
- Dennis et al., **PAIRED / Unsupervised Environment Design** (NeurIPS 2020) — [arXiv:2012.02096](https://arxiv.org/abs/2012.02096).
- Faldor et al., **OMNI-EPIC** (2024) — [arXiv:2405.15568](https://www.arxiv.org/abs/2405.15568).
- Wang et al., **Voyager** (2023) — [arXiv:2305.16291](https://arxiv.org/abs/2305.16291).
- Sohn et al., **Hierarchical RL for zero-shot generalization with subtask dependencies** (NeurIPS 2018) — [arXiv:1807.07665](https://arxiv.org/pdf/1807.07665).
- Toro Icarte et al., **Reward machines** (ICML 2018) — [PMLR v80](https://proceedings.mlr.press/v80/icarte18a.html).
- Matthews et al., **Craftax** (ICML 2024) — [project page](https://craftaxenv.github.io), [paper](https://arxiv.org/pdf/2402.16801).
- **bracket-lib** (Rust roguelike toolkit, split into geometry, pathfinding, random, noise, color and terminal crates) — [docs.rs](https://docs.rs/bracket-lib).
- **Rain World** (Videocult, 2017), on its living ecosystem — [Game Developer: crafting the ecosystem of Rain World](https://gamedeveloper.com/design/crafting-the-complex-chaotic-ecosystem-of-i-rain-world-i-).
- Also in the background, not re-checked for this file: PCGRL, NetHack/NLE/MiniHack, ARC-AGI-3.

---

## Changelog

- **2026-11-20** — File created from the eight ideas of the walk, plus additions and references.
- **2026-11-20** — I-07: the reference is Rain World (it had been written down as "Cat World"); added what that kind of ecosystem would ask of the engine.
- **2026-11-20** — New idea I-09 (a learned generator for entity symbols); I-02 now points to it.