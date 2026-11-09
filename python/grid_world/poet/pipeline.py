"""
Created on Mon Nov  2 08:46:42 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import copy
import time
import numpy as np
import random
 

from neat.config import default_config


from engine import GenerationConfig, generate_valid_layout
 

from collections import Counter

from poet.child import make_child
from poet.eligibility import is_eligible
from poet.entry import Entry, entry_agents, train_entry
from poet.reproduction import reproduce, entry_distance
from poet.pool import select_novel, admit
from poet.learner import Learner
from poet.loop import run_poet
from poet.criterion import passes_minimal_criterion
from poet.games import split_roles, build_game, run_game, watch_game, play_games, goal_rate
from poet.environment import make_world, mutate_layout, layout_picture, layout_distance
from poet.transfer import role_rate, transfer_into, transfer_round
from poet.generalization import held_out_layouts, pressure_on_layouts, skill_on_layouts, summarize

from agents.heuristic_agent.agent import HeuristicAgent

from experiments.poet.learner_config import TrainConfig, ChampionConfig, check_seed_ranges
from experiments.poet.neat_scoring import play_with_visits, score_population
from experiments.poet.neat_champion import pick_champion
from experiments.poet.neat_learner import NeatLearner

###############################################################################
# Settings
###############################################################################
 
generation_config = {
    "width"  : 16,
    "height" : 16,
    "wall_density" : 0.1,
    "num_enemies" : 1,
    "num_traps" : 0,
    "min_player_goal_distance" : 10
}
 
mutation_config = {
    "n_wall_mutations" : 4,
    "max_wall_density" : 0.3,
    "max_enemies" : 1,
    "enemy_add_weight"     : 0.0,
    "enemy_remove_weight"  : 0.0,
    "enemy_nothing_weight" : 1.0,
    "max_traps" : 3,
    "trap_add_weight"     : 0.2,
    "trap_remove_weight"  : 0.1,
    "trap_nothing_weight" : 0.7,
    "n_repositions" : 1,
    "min_player_goal_distance" : 10,
}
 
###############################################################################
# Base layout
###############################################################################
 
layout = generate_valid_layout(GenerationConfig(**generation_config))

###############################################################################
# Piece: make_world
###############################################################################
 
world, player_id, enemy_ids = make_world(layout, seed=1)
 
print(world.render_world())
print("player id:", player_id, "| enemy ids:", enemy_ids)
 
# The layout fixes the ids and the starting positions; the seed does not change them here.
# Expected: True True
other_world, other_player_id, other_enemy_ids = make_world(layout, seed=2)
print("same ids       :", (player_id, enemy_ids) == (other_player_id, other_enemy_ids))
print("same picture   :", world.render_world() == other_world.render_world())

###############################################################################
# Piece: mutate_layout
###############################################################################
 
child_layout = mutate_layout(layout, mutation_config, seed=1)
 
print(make_world(child_layout, seed=1)[0].render_world())
 
# Same seed, same child. Expected: True
again = mutate_layout(layout, mutation_config, seed=1)
print("same seed, same child:", make_world(again, 1)[0].render_world() == make_world(child_layout, 1)[0].render_world())
 
# Other seeds, other children. Expected: 10 different pictures, none equal to the parent
pictures = {make_world(mutate_layout(layout, mutation_config, seed=s), 0)[0].render_world() for s in range(10)}
print("different children from 10 seeds:", len(pictures), "| parent among them:", make_world(layout, 0)[0].render_world() in pictures)

###############################################################################
# Piece: split_roles
###############################################################################
 
# Stand-ins: split_roles only sorts, so any object can take a role
print("only player :", split_roles({"player": "P"}))
print("with enemy  :", split_roles({"player": "P", "enemy_0": "E0"}))
print("two enemies :", split_roles({"enemy_1": "E1", "player": "P", "enemy_0": "E0"}))
 
# A gap in the enemy numbers must fail loudly. Expected: the ValueError message
try:
    split_roles({"player": "P", "enemy_1": "E1"})
except ValueError as error:
    print("gap         :", error)
    
###############################################################################
# Piece: build_game
###############################################################################
 
# The heuristic player takes the role "player"; no one takes the enemy, so the engine moves it
player = HeuristicAgent()
world, agents, role_of = build_game(layout, 0, {"player": player})
 
print(world.render_world())
print("roles (entity id -> role)     :", role_of)
print("agents (entity id -> agent)   :", agents)
 
# The ids belong to the layout, the roles do not. Expected: the same roles, the ids may differ
_, _, child_roles = build_game(child_layout, 0, {"player": player})
print("roles in the mutated layout   :", child_roles)
print("same roles, ids may differ    :", sorted(child_roles.values()) == sorted(role_of.values()))
 
# An agent for an enemy that does not exist must fail loudly. Expected: the ValueError message
try:
    build_game(layout, 0, {"player": player, "enemy_0": "E0", "enemy_1": "E1"})
except ValueError as error:
    print("too many enemy agents         :", error)
    
###############################################################################
# Piece: run_game
###############################################################################
 
result, role_of = run_game(layout, 0, {"player": HeuristicAgent()})
print("roles (entity id -> role) :", role_of)
print("outcomes (entity id)      :", result.agent_outcomes, "| steps", result.steps)
 
# The same layout, ten seeds: the seed decides where the entities start, so the games differ.
# Expected: ten different starting cells (or close to it) and a mix of outcomes
starts, outcomes = [], []
for seed in range(10):
    agent = HeuristicAgent()
    result, role_of = run_game(layout, seed, {"player": agent})
    starts.append(agent.positions[0])
    outcomes.append(list(result.agent_outcomes.values())[0])
 
print("player start cells        :", starts)
print("different start cells     :", len(set(starts)))
print("outcomes of the player    :", dict(Counter(outcomes)))
 
# Same layout, same seed, same agent: the same game. Expected: True
again, _ = run_game(layout, 3, {"player": HeuristicAgent()})
first, _ = run_game(layout, 3, {"player": HeuristicAgent()})
print("same seed, same game      :", (again.agent_outcomes, again.steps) == (first.agent_outcomes, first.steps))

###############################################################################
# Piece: watch_game
###############################################################################
 
# The same game as run_game(layout, 3, ...) drawn frame by frame; it returns the agents by role
agents_by_role = watch_game(layout, 3, {"player": HeuristicAgent()}, delay=0.1)
print("agents by role:", agents_by_role)

###############################################################################
# Piece: play_games
###############################################################################
 
# 100 games on the fixed layout, seeds 0..99. Expected: a list of 100 outcomes per role with an
# agent (only "player" here, nobody controls the enemy yet)
outcomes = play_games(layout, {"player": HeuristicAgent()}, n_games=100)
print("roles with outcomes :", list(outcomes))
print("games played        :", len(outcomes["player"]))
print("player outcomes     :", dict(Counter(outcomes["player"])))
 
# Other seeds, other games. Expected: different counts (same layout, different starts)
other = play_games(layout, {"player": HeuristicAgent()}, n_games=100, first_seed=1000)
print("seeds 1000..1099    :", dict(Counter(other["player"])))
 
# The same seeds again give the same outcomes. Expected: True
again = play_games(layout, {"player": HeuristicAgent()}, n_games=100)
print("repeatable          :", again == outcomes)
 
###############################################################################
# Piece: goal_rate
###############################################################################
 
# Expected: the number of GoalReached in the two counts above divided by 100 (0.91 and 0.94 before)
print("goal rate, seeds 0..99      :", goal_rate(outcomes["player"]))
print("goal rate, seeds 1000..1099 :", goal_rate(other["player"]))
 
# A typo is an error, not a loss. Expected: ValueError and its message
try:
    goal_rate(["GoalReached", "GoalReched"])
except ValueError as error:
    print("typo ->", error)
 
###############################################################################
# Piece: passes_minimal_criterion
###############################################################################
 
# 1) Made-up outcomes, to see the rule alone. Expected: True / False / False
def made_up(player_goals, enemy_goals, timeouts, total=100):
    assert player_goals + timeouts <= total, "more player games than the total"
    player = ["GoalReached"] * player_goals + ["Timeout"] * timeouts + ["Caught"] * (total - player_goals - timeouts)
    enemy = ["GoalReached"] * enemy_goals + ["Caught"] * (total - enemy_goals)
    return {"player": player, "enemy_0": enemy}
 
print("balanced          :", passes_minimal_criterion(made_up(45, 40, 10)))
print("player too easy   :", passes_minimal_criterion(made_up(88, 40, 10)))
print("too many timeouts :", passes_minimal_criterion(made_up(45, 40, 40)))
 
# 2) The real thing: a player and an enemy agent playing the fixed layout.
# HeuristicAgent in both roles is only a stand-in until we have trained agents
# (if it cannot play the enemy role in your engine, this prints the error)
try:
    both = play_games(layout, {"player": HeuristicAgent(), "enemy_0": HeuristicAgent()}, n_games=100)
    print("player outcomes   :", dict(Counter(both["player"])))
    print("enemy outcomes    :", dict(Counter(both["enemy_0"])))
    print("criterion         :", passes_minimal_criterion(both))
except Exception as error:
    print("stand-in enemy failed:", type(error).__name__, error)
 
###############################################################################
# Piece: Learner
###############################################################################

# The simplest possible learner: it never learns. It shows that any algorithm fits, and that
# poet only needs agent(), train() and clone()
class FixedLearner:
    def __init__(self, agent):
        self._agent = agent
 
    def agent(self):
        return self._agent
 
    def train(self, layout, opponent, rng):
        pass                                    # nothing to learn, so nothing to do
 
    def clone(self):
        return FixedLearner(self._agent.clone())
 
fixed = FixedLearner(HeuristicAgent())
print("is a Learner      :", isinstance(fixed, Learner))          # Expected: True
print("object() is one   :", isinstance(object(), Learner))       # Expected: False
 
twin = fixed.clone()
print("clone is another  :", twin is not fixed and twin.agent() is not fixed.agent())   # Expected: True
 
# A learner plugs into what we already built: play with whatever agent() gives
outcomes = play_games(layout, {"player": fixed.agent()}, n_games=100)
print("goal rate         :", goal_rate(outcomes["player"]))

###############################################################################
# Piece: Entry
###############################################################################

# One entry: the fixed layout and two learners that never learn (see FixedLearner above)
entry = Entry(layout, {"player": FixedLearner(HeuristicAgent()), "enemy_0": FixedLearner(HeuristicAgent())})
print("roles of the entry :", list(entry.learners))                # Expected: ['player', 'enemy_0']
 
# The agents to play with, ready for play_games. Expected: the same numbers as the criterion section
agents = entry_agents(entry)
outcomes = play_games(entry.layout, agents, n_games=100)
print("criterion          :", passes_minimal_criterion(outcomes))
 
# One round of training. These learners learn nothing, so nothing should change.
# The agents keep their own random generator between games (reset() does not reseed it), so the
# comparison must start from agents with the same seed: two fresh entries, one of them trained.
def fresh_entry():
    return Entry(layout, {"player": FixedLearner(HeuristicAgent()), "enemy_0": FixedLearner(HeuristicAgent())})
 
untouched, trained = fresh_entry(), fresh_entry()
train_entry(trained, random.Random(0))
same = play_games(untouched.layout, entry_agents(untouched), n_games=100) \
    == play_games(trained.layout, entry_agents(trained), n_games=100)
print("nothing changed    :", same)                                   # Expected: True
 
# And the reason of the False you saw: the SAME agents keep drawing from their random generator,
# so a second run with them is a different run. Expected: False (almost surely)
first = play_games(untouched.layout, entry_agents(untouched), n_games=100)
second = play_games(untouched.layout, entry_agents(untouched), n_games=100)
print("same agents, twice :", first == second)
 
# Both roles are needed. Expected: ValueError
try:
    Entry(layout, {"player": FixedLearner(HeuristicAgent())})
except ValueError as error:
    print("one role only ->", error)
    
###############################################################################
# Piece: is_eligible
###############################################################################
 
# Fresh seeds, the ones the learners will never train on (230_000 is the range planned for this)
entry = fresh_entry()
print("eligible (HeuristicAgent in both roles):", is_eligible(entry, n_games=100, first_seed=230_000))
# Expected: (False, {'player': ~0.8, 'enemy_0': ~0.16, 'timeout': ~0.03}), like the criterion section
 
# The limits of the criterion can be changed. Expected: the same report, now True
print("with low=0.1, high=0.9:", is_eligible(fresh_entry(), n_games=100, first_seed=230_000, low=0.1, high=0.9))
 
# The check does not train anything: the agents are the same objects before and after
agents_before = entry_agents(entry)
is_eligible(entry, n_games=10, first_seed=230_000)
print("same agents after   :", all(entry_agents(entry)[r] is agents_before[r] for r in agents_before))   # Expected: True
 
###############################################################################
# Piece: make_child
###############################################################################
 
parent = fresh_entry()
 
# 1) With the real limits. HeuristicAgent is too good as the player and too weak as the enemy, so
#    the parent's agents do not pass on the child layout either. Expected: (None, {report})
child, report = make_child(parent, mutation_config, mutation_seed=1, n_games=100, first_seed=230_000)
print("default limits  :", child, report)
 
# 2) Loose limits, to see an accepted child. Expected: an Entry and its report
loose = dict(low=0.05, high=0.95, timeout_cap=0.5)
child, report = make_child(parent, mutation_config, mutation_seed=1, n_games=100, first_seed=230_000, **loose)
print("loose limits    :", type(child).__name__, report)
 
# What an accepted child is. Expected: True / True / True
print("other layout    :", child.layout != parent.layout)
print("own learners    :", all(child.learners[r] is not parent.learners[r] for r in child.learners))
print("same agent kind :", all(type(child.learners[r].agent()) is type(parent.learners[r].agent()) for r in child.learners))
 
# 3) Ten mutation seeds with the loose limits: how many children are accepted?
accepted = [make_child(parent, mutation_config, mutation_seed=s, n_games=50, first_seed=230_000, **loose)[0] is not None
            for s in range(10)]
print("accepted of 10  :", sum(accepted))

###############################################################################
# Diagnosis: why could the agent not be copied?
###############################################################################
 
# Which attribute of an agent that has played holds engine observations?
played = HeuristicAgent()
play_games(layout, {"player": played}, n_games=3)
print({name: type(value).__name__ for name, value in vars(played).items()})
print("BaseAgent defines __getstate__:", "__getstate__" in vars(type(played).__mro__[1]))
 
# The copy that BaseAgent offers for this. Expected: a HeuristicAgent with an empty history
copy_of_played = played.clone()
print(type(copy_of_played).__name__, len(copy_of_played.history))
 
###############################################################################
# Piece: layout_distance
###############################################################################
 
# The picture a distance is measured on: plain text, one character per cell
print(layout_picture(layout))
 
# A layout against itself. Expected: 0
print("itself            :", layout_distance(layout, layout))
 
# Against its mutations (1 new wall and 1 reposition each). Expected: small numbers, none 0
print("to 10 children    :", [layout_distance(layout, mutate_layout(layout, mutation_config, seed=s)) for s in range(10)])
 
# Symmetric. Expected: True
child_layout = mutate_layout(layout, mutation_config, seed=1)
print("symmetric         :", layout_distance(layout, child_layout) == layout_distance(child_layout, layout))
 
# Children of the same parent are close to each other, a layout from scratch is far from all.
siblings = [mutate_layout(layout, mutation_config, seed=s) for s in range(3)]
print("siblings          :", [layout_distance(siblings[i], siblings[j]) for i in range(3) for j in range(i + 1, 3)])
# (the same config and seed always gives the same layout, so the stranger needs another seed)
print("same config again :", layout_distance(layout, generate_valid_layout(GenerationConfig(**generation_config))))   # Expected: 0
stranger = generate_valid_layout(GenerationConfig(**generation_config, seed=12345))
print("to a new layout   :", layout_distance(layout, stranger))                                                     # Expected: large

###############################################################################
# Piece: select_novel and admit
###############################################################################

pool = [fresh_entry()]
 
# Children of the pool's entry that pass the (loose) criterion: the candidates
candidates = []
for s in range(10):
    c, _ = make_child(pool[0], mutation_config, mutation_seed=s, n_games=50, first_seed=230_000, **loose)
    if c is not None:
        candidates.append(c)
print("candidates          :", len(candidates))
 
# Distance of every candidate to the pool, and between candidates
print("to the pool         :", [entry_distance(c, pool[0]) for c in candidates])
 
# Pick up to 3 of them, the most different from the pool and from each other
picked = select_novel(candidates, pool, n_keep=3, distance=entry_distance)
print("picked              :", len(picked), "| distances to the pool:", [entry_distance(c, pool[0]) for c in picked])
print("among the picked    :", [entry_distance(picked[i], picked[j]) for i in range(len(picked)) for j in range(i + 1, len(picked))])
print("picked are candidates:", all(any(p is c for c in candidates) for p in picked))        # Expected: True
 
# A copy of the pool's entry adds nothing. Expected: [] (its distance is 0)
print("copy of the parent  :", select_novel([pool[0]], pool, n_keep=3, distance=entry_distance))
 
# Admit them into a pool with room for 3: the oldest entries leave first
bigger = admit(pool, picked, max_size=3)
print("pool size           :", len(pool), "->", len(bigger))
print("parent still there  :", pool[0] in bigger)                                         # True if the pool had room
print("oldest leave first  :", pool[0] in admit(bigger, picked, max_size=3))               # Expected: False once it overflows

###############################################################################
# Piece: reproduce
###############################################################################
 
# One round of reproduction from a pool of one entry. With the real limits nobody is eligible
# (the heuristic agents are too unbalanced), so loose limits are used to see the mechanics.
pool = [fresh_entry()]
 
# Real limits first. Expected: no eligible parent, nothing tried, no children
children, report = reproduce(pool, mutation_config, mutation_seed=1000, n_children=4, max_new=3,
                             n_games=50, first_seed=230_000)
print("real limits  :", len(children), report)
 
# Loose limits. Expected: 1 eligible, 4 attempts, a few accepted, up to 3 picked
children, report = reproduce(pool, mutation_config, mutation_seed=1000, n_children=4, max_new=3,
                             n_games=50, first_seed=230_000, **loose)
print("loose limits :", len(children), report)
print("distances to the parent:", [entry_distance(c, pool[0]) for c in children])
 
# The same arguments give the same children. Expected: True
again, _ = reproduce(pool, mutation_config, mutation_seed=1000, n_children=4, max_new=3,
                     n_games=50, first_seed=230_000, **loose)
print("same arguments, same children:", [layout_distance(a.layout, b.layout) for a, b in zip(children, again)] == [0] * len(children))
 
# The next round: the pool grows, the seeds move on (the caller advances them), and a child must
# also differ from every entry already in the pool
pool = admit(pool, children, max_size=10)
children2, report2 = reproduce(pool, mutation_config, mutation_seed=2000, n_children=2, max_new=3,
                               n_games=50, first_seed=230_000, **loose)
print("round 2      :", len(pool), "entries ->", len(children2), report2)
print("nearest entry of each new child:", [min(entry_distance(c, e) for e in pool) for c in children2])    # Expected: all > 0

###############################################################################
# Piece: run_poet
###############################################################################
 
# The learners here never learn, so the only thing that can change is the pool. What to see:
# the pool grows, the oldest entries leave when it is full, and the report of every generation.
def show(generation, pool, report):
    print(f"gen {generation}: eligible {report['eligible']} | attempts {report['attempts']} | accepted {report['accepted']} "
          f"| picked {report['picked']} | dropped {report['dropped']} | pool {report['pool']}")
    if report["eligible"] == 0:
        print("   WARNING: no eligible parent, nothing was born")
 
# 1) Real limits. Nobody is eligible, so the pool stays as it is. Expected: 3 lines with the warning
pool, history = run_poet([fresh_entry()], mutation_config, n_generations=3, max_pool=5, n_children=3, max_new=2,
                         n_games=50, first_seed=230_000, mutation_seed=1000, on_generation=show)
print("pool size:", len(pool))
 
# 2) Loose limits. Expected: the pool grows by up to 2 per generation until 5, then the oldest leave
pool, history = run_poet([fresh_entry()], mutation_config, n_generations=5, max_pool=5, n_children=3, max_new=2,
                         n_games=50, first_seed=230_000, mutation_seed=1000, on_generation=show, **loose)
print("pool size:", len(pool))
print("every layout different from the others:", min(entry_distance(a, b) for i, a in enumerate(pool) for b in pool[i + 1:]) > 0)
 
# 3) The same arguments give the same run. Expected: True
again, history_again = run_poet([fresh_entry()], mutation_config, n_generations=5, max_pool=5, n_children=3, max_new=2,
                                n_games=50, first_seed=230_000, mutation_seed=1000, **loose)
print("same run twice:", history == history_again and [entry_distance(a, b) for a, b in zip(pool, again)] == [0] * len(pool))

###############################################################################
# NEAT learner, piece 1: its settings
###############################################################################
 
train_config = TrainConfig(population_size=20, n_generations=5, n_games=10, n_valid_games=20, k_neighbours=5)
champion_config = ChampionConfig()
check_seed_ranges(train_config, champion_config)                  # no error: the ranges do not overlap
print(train_config)
print("training may go up to generation", train_config.last_generation())     # Expected: 999 (200_000 ... 210_000, 10 games each)
 
try:
    TrainConfig(population_size=10, n_generations=5, n_games=10, n_valid_games=20)       # k_neighbours = 10 is not < 10
except ValueError as error:
    print("rejected ->", error)
 
###############################################################################
# NEAT learner, piece 2: scoring a population
###############################################################################
 
# One game, to see what is collected: the outcomes and the cells each role stood on
outcomes, visits = play_with_visits(layout, {"player": HeuristicAgent(), "enemy_0": HeuristicAgent()}, n_games=3, first_seed=230_000)
print("outcomes:", outcomes)
print("cells per game (player):", [len(v) for v in visits["player"]])
 
# Four HeuristicAgents with different seeds stand in for the genomes of a population
rates, novelty, fitness, cells = score_population(
    [1, 2, 3, 4], "player", {"enemy_0": HeuristicAgent()}, layout, n_games=10, first_seed=200_000,
    novelty_weight=0.2, k_neighbours=2, make_agent=lambda seed: HeuristicAgent(seed=seed))
print("rates  :", rates)
print("novelty:", [round(x, 3) for x in novelty])
print("fitness:", [round(x, 3) for x in fitness])
print("cells  :", cells)
 
# The same call again gives the same numbers (every genome meets a clone of the enemy). Expected: True
again = score_population(
    [1, 2, 3, 4], "player", {"enemy_0": HeuristicAgent()}, layout, n_games=10, first_seed=200_000,
    novelty_weight=0.2, k_neighbours=2, make_agent=lambda seed: HeuristicAgent(seed=seed))
print("repeatable:", again == (rates, novelty, fitness, cells))
 
# With real genomes: a random population of NEAT networks, scored by the default make_agent (NeatAgent).
# The import paths are the ones I found in my copy of your project; adjust them if something moved.
try:

    from neat.evolution import init_population, evolve_one_generation
    from agents.neat_agent.agent import N_INPUTS
    from core.actions import N_ACTIONS
except ImportError as error:
    print("adjust the imports of the NEAT library:", error)
else:
    neat_config = {**default_config(), "population_size": 20}
    state = init_population(N_INPUTS, N_ACTIONS, np.random.default_rng(0), neat_config)
    enemy = {"enemy_0": HeuristicAgent()}
 
    rates, novelty, fitness, cells = score_population(
        state["genomes"], "player", enemy, layout, n_games=10, first_seed=200_000, novelty_weight=0.2, k_neighbours=5)
    print("random population, goal rates:", rates)
    print("random population, novelty   :", [round(x, 2) for x in novelty])
 
    ###########################################################################
    # NEAT learner, piece 3: choosing the champion
    ###########################################################################
 
    # The champion starts as the first genome. Stage 1 picks finalists, stage 2 plays them on fresh games.
    champion_config = ChampionConfig(n_finalists=3, n_score_games=10, n_final_games=30)
    champion = state["genomes"][0]
    genome, rate = pick_champion(state["genomes"], champion, "player", enemy, layout, first_seed=240_000,
                                 novelty_weight=0.2, k_neighbours=5, config=champion_config)
    print("champion kept :", genome is champion, "| goal rate on 30 fresh games:", round(rate, 2))
 
    # Repeatable: the opponent is cloned for every finalist and NeatAgent plays without randomness. Expected: True
    again = pick_champion(state["genomes"], champion, "player", enemy, layout, first_seed=240_000,
                          novelty_weight=0.2, k_neighbours=5, config=champion_config)
    print("same choice twice:", again[0] is genome and again[1] == rate)
 
    # After some evolution the champion should not be worse than the first genome. Expected: True (usually)
    rng = np.random.default_rng(1)
    for _ in range(10):
        rates, novelty, fitness, cells = score_population(
            state["genomes"], "player", enemy, layout, 10, 200_000, 0.2, 5)
        state, _ = evolve_one_generation(state, fitness, rng, neat_config)
    evolved, evolved_rate = pick_champion(state["genomes"], champion, "player", enemy, layout, 240_000, 0.2, 5, champion_config)
    print("after 10 generations: champion changed:", evolved is not champion, "| rate:", round(evolved_rate, 2), "(was", round(rate, 2), ")")

###############################################################################
# NEAT learner, piece 4: the learner (append this to pipeline_v2.py)
###############################################################################
#
# New imports for the top of the file:
#
#   import copy
#   import time
#   import numpy as np
#   from neat.config import default_config
#   from experiments.poet.neat_learner import NeatLearner
#
# (random, Entry, train_entry, is_eligible, play_games, Learner, TrainConfig, ChampionConfig and
#  check_seed_ranges are already imported there.)

neat_config = default_config()

# The numbers of your old grow_entry: a population of 50, 10 generations per call, 20 games per genome
learner_train = TrainConfig(population_size=50, n_generations=10, n_games=20, n_valid_games=50, k_neighbours=10)
learner_champion = ChampionConfig(n_finalists=5, n_score_games=20, n_final_games=100)
check_seed_ranges(learner_train, learner_champion)

print("calls of train() before the training seeds reach the validation games:",
      (learner_train.last_generation() + 1) // learner_train.n_generations)         # Expected: 50

###############################################################################
# 1) What a NeatLearner is, and what an untrained pair does
###############################################################################

player_learner = NeatLearner("player", neat_config, learner_train, learner_champion, seed=0)
enemy_learner = NeatLearner("enemy_0", neat_config, learner_train, learner_champion, seed=1)

print("is a Learner       :", isinstance(player_learner, Learner))                  # Expected: True
print("population         :", len(player_learner.state["genomes"]), "| generation", player_learner.state["generation"])

# Nothing is trained yet. Expected: both goal rates close to 0, almost every game a timeout or a catch
untrained = play_games(layout, {"player": player_learner.agent(), "enemy_0": enemy_learner.agent()},
                       n_games=100, first_seed=230_000)
print("untrained pair     :", {role: dict(Counter(o)) for role, o in untrained.items()})
print("goal rates         :", {role: goal_rate(o) for role, o in untrained.items()})

###############################################################################
# 2) The learner keeps its promises (with a small, quick budget)
###############################################################################

quick = TrainConfig(population_size=10, n_generations=2, n_games=5, n_valid_games=20, k_neighbours=3)
quick_champion = ChampionConfig(n_finalists=2, n_score_games=5, n_final_games=10)

learner = NeatLearner("player", neat_config, quick, quick_champion, seed=0)
opponent = HeuristicAgent()

start = time.time()
learner.train(layout, opponent, random.Random(0))
print(f"one quick train() : {time.time() - start:.1f} s | generation {learner.state['generation']} | "
      f"champion goal rate {learner.champion_rate:.2f}")

# One row per generation scored, like the table of the old run_generations
print("history            :", [(row["generation"], round(row["goal_mean"], 3)) for row in learner.history])

# Same learner, same rng, same result. Expected: True
again = NeatLearner("player", neat_config, quick, quick_champion, seed=0)
again.train(layout, HeuristicAgent(), random.Random(0))
print("same training twice:", again.history == learner.history and again.champion == learner.champion)

# A clone is the same learner, and training one never changes the other. Expected: True True True
twin = learner.clone()
print("clone is equal     :", twin.champion == learner.champion and twin.state == learner.state and twin is not learner)
before = copy.deepcopy((learner.state, learner.champion))
twin.train(layout, HeuristicAgent(), random.Random(1))
print("original untouched :", (learner.state, learner.champion) == before)
print("clone moved on     :", twin.state["generation"] == learner.state["generation"] + 2)

# The agent of a learner that already played can still be cloned: it holds no engine observation
played = learner.agent()
play_games(layout, {"player": played, "enemy_0": HeuristicAgent()}, n_games=3, first_seed=230_000)
print("clone after playing:", learner.clone().champion == learner.champion)

# The seed guard: asking for generations whose training games would run into the validation games
guarded = NeatLearner("player", neat_config, quick, quick_champion, seed=0)
guarded.state["generation"] = quick.last_generation()              # a call of 2 generations would reach one beyond
try:
    guarded.train(layout, HeuristicAgent(), random.Random(0))
except ValueError as error:
    print("seed guard         ->", error)                          # Expected: the message, and the learner unchanged
print("learner unchanged  :", guarded.state["generation"] == quick.last_generation())

###############################################################################
# 3) The first end-to-end run: an untrained pair grows until it can be a parent
###############################################################################
#
# Both roles start untrained. One round (train_entry) is: the player evolves 10 generations
# against the enemy's champion, then the enemy evolves 10 generations against the NEW player.
# After every round the pair plays the minimal criterion on games nobody trained on.
#
# About 6 minutes per round on the machine of the first experiments (1000 games per generation).
# Time the first round before you leave it running.

MAX_ROUNDS = 20

entry = Entry(layout, {"player": player_learner, "enemy_0": enemy_learner})
rng = random.Random(0)

def learner_line(role):
    learner = entry.learners[role]
    rows = learner.history[-learner_train.n_generations:]
    return (f"{role}: population goal rate {sum(r['goal_mean'] for r in rows) / len(rows):.3f}"
            f" (best {max(r['goal_best'] for r in rows):.2f}) | champion {learner.champion_rate:.2f}")

for round_number in range(1, MAX_ROUNDS + 1):
    start = time.time()
    train_entry(entry, rng)
    eligible, report = is_eligible(entry, n_games=100, first_seed=230_000)

    print(f"round {round_number:>2} | eligible: {eligible} | "
          + "  ".join(f"{k}={v:.2f}" for k, v in report.items()) + f" | {time.time() - start:.0f} s")
    print("         ", learner_line("player"))
    print("         ", learner_line("enemy_0"))

    if eligible:
        break

###############################################################################
# 4) The POET loop on the grown entry
###############################################################################
#
# Reproduction needs an eligible parent, so this only makes sense when the loop above stopped with
# eligible = True. Every generation trains EVERY entry of the pool (about 6 minutes each), so the
# time of a generation grows with the pool: 5 entries are about half an hour.

def show_poet(generation, pool, report):
    print(f"gen {generation}: eligible {report['eligible']} | attempts {report['attempts']} | accepted {report['accepted']} "
          f"| picked {report['picked']} | dropped {report['dropped']} | pool {report['pool']}")
    for number, member in enumerate(pool):
        ok, check = is_eligible(member, n_games=100, first_seed=230_000)
        print(f"   entry {number}: eligible {ok} | " + "  ".join(f"{k}={v:.2f}" for k, v in check.items()))

pool, history = run_poet(
    [entry], mutation_config, n_generations=5, max_pool=5, n_children=3, max_new=2,
    n_games=100, first_seed=230_000, mutation_seed=1000, train_seed=0, on_generation=show_poet)

###############################################################################
# Piece: transfer (append this to pipeline_v2.py, after the NEAT learner block)
###############################################################################
#
# New import for the top of the file:
#
#   from poet.transfer import role_rate, transfer_into, transfer_round
#
# It uses `quick` and `quick_champion` (the small TrainConfig / ChampionConfig of the learner
# block), `fresh_entry` and `FixedLearner` (the helpers), and `mutation_config`.
 
# The seeds of the transfer games: far from the training games (200_000 ...), the validation games
# (210_000 ...), the eligibility games (230_000 ...) and the champion games (240_000 ...)
TRANSFER_SEED = 500_000
 
other_layout = mutate_layout(layout, mutation_config, seed=1)       # a neighbour of the base layout
 
 
def weak_entry():
    """Lives on the neighbour layout. Its player is an UNTRAINED NEAT network (goal rate ~0), its
    enemy a fixed heuristic."""
    return Entry(other_layout, {
        "player": NeatLearner("player", neat_config, quick, quick_champion, seed=0),
        "enemy_0": FixedLearner(HeuristicAgent()),
    })
 
 
def strong_entry():
    """Lives on the base layout. Its player is a heuristic (goal rate ~0.9)."""
    return Entry(layout, {"player": FixedLearner(HeuristicAgent()), "enemy_0": FixedLearner(HeuristicAgent())})
 
 
###############################################################################
# 1) role_rate: how well does an agent play a role in THIS entry?
###############################################################################
 
weak, strong = weak_entry(), strong_entry()
 
# The entry's own player, and the strong entry's player, both measured on the weak entry's layout and
# against the weak entry's own enemy. Same 50 games for both.
own = role_rate(weak, "player", weak.learners["player"].agent(), n_games=50, first_seed=TRANSFER_SEED)
guest = role_rate(weak, "player", strong.learners["player"].agent(), n_games=50, first_seed=TRANSFER_SEED)
print("own player on its layout          :", own)          # Expected: close to 0
print("strong entry's player, same games :", guest)        # Expected: much higher (~0.8-0.9)
 
# Measuring does not move anybody (both play as clones). Expected: True
print("repeatable                        :", role_rate(weak, "player", strong.learners["player"].agent(), 50, TRANSFER_SEED) == guest)
 
###############################################################################
# 2) transfer_into: the weak entry adopts what is better, role by role
###############################################################################
 
weak, strong = weak_entry(), strong_entry()
strong_player = strong.learners["player"]
 
# margin=0.1: a candidate must beat the entry's own agent by more than 0.1 to be adopted
report = transfer_into(weak, [strong], n_games=50, first_seed=TRANSFER_SEED, margin=0.1)
for role, row in report.items():
    print(f"{role:8}: own {row['own']:.2f} | best candidate {row['best']:.2f} (entry {row['from']}) | adopted {row['adopted']}")
# Expected: the player is adopted (own ~0, best ~0.9). The enemy is not: both are heuristics, the
# difference is within the margin.
 
print("player is now a NeatLearner?      :", type(weak.learners["player"]).__name__)             # Expected: FixedLearner
print("a clone, not the strong entry's   :", weak.learners["player"] is not strong_player)        # Expected: True
print("the strong entry is untouched     :", strong.learners["player"] is strong_player)          # Expected: True
print("layout of the weak entry unchanged:", weak.layout == other_layout)                         # Expected: True
 
# Once adopted, the same call finds nothing better. Expected: adopted False in both roles
again = transfer_into(weak, [strong], n_games=50, first_seed=TRANSFER_SEED, margin=0.1)
print("second call                       :", {role: row["adopted"] for role, row in again.items()})
 
# With no candidates nothing is played. Expected: all None
print("no candidates                     :", transfer_into(weak_entry(), [], 50, TRANSFER_SEED))
 
###############################################################################
# 3) transfer_round: the whole pool
###############################################################################
 
pool = [weak_entry(), strong_entry(), strong_entry()]
reports = transfer_round(pool, n_games=50, first_seed=TRANSFER_SEED, margin=0.1)
 
for number, entry_report in enumerate(reports):
    print(f"entry {number}:", {role: (round(row["own"], 2), round(row["best"], 2), row["from"], row["adopted"])
                               for role, row in entry_report.items()})
# Expected: entry 0 adopts a player (from = 0: the first of its candidates, entry 1, since the two strong
# ones are equal); entries 1 and 2 adopt nothing.
print("types after the round :", [type(e.learners["player"]).__name__ for e in pool])   # Expected: all FixedLearner
print("layouts unchanged     :", [e.layout for e in pool] == [other_layout, layout, layout])

###############################################################################
# Piece: run_poet with transfer (append this to pipeline_v2.py)
###############################################################################
#
# It uses weak_entry() and strong_entry() from the transfer block, `loose` and `mutation_config`
# from the helpers, and `entry` (the grown NEAT entry) from the NEAT learner block.

def show_with_transfer(generation, pool, report):
    moves = ", ".join(f"entry {i} took the {role} of entry {k}" for i, role, k in report["transfers"]) or "none"
    print(f"gen {generation}: eligible {report['eligible']} | attempts {report['attempts']} | accepted {report['accepted']} "
          f"| picked {report['picked']} | dropped {report['dropped']} | pool {report['pool']} | transfers: {moves}")

###############################################################################
# 1) The mechanics, fast: learners that never learn, a weak entry next to a strong one
###############################################################################

# The weak entry's player is an untrained NEAT network, the strong entry's player a heuristic.
# n_transfer=1: a round of transfer after every generation.
pool, history = run_poet(
    [weak_entry(), strong_entry()], mutation_config, n_generations=3, max_pool=4, n_children=2, max_new=2,
    n_games=50, first_seed=230_000, mutation_seed=1000, on_generation=show_with_transfer,
    n_transfer=1, n_transfer_games=50, transfer_margin=0.1, **loose)
# Expected: generation 0 has the move "entry 0 took the player of entry 1". Later generations may
# show moves for the children too: a child lives on another layout, where a neighbour can do better.

# Without n_transfer nothing is transferred. Expected: [] in every generation
pool, history = run_poet(
    [weak_entry(), strong_entry()], mutation_config, n_generations=2, max_pool=4, n_children=2, max_new=2,
    n_games=50, first_seed=230_000, mutation_seed=1000, **loose)
print("transfers without n_transfer:", [h["transfers"] for h in history])

# The same arguments give the same run, transfers included. Expected: True
args = dict(mutation_config=mutation_config, n_generations=2, max_pool=4, n_children=2, max_new=2, n_games=50,
            first_seed=230_000, mutation_seed=1000, n_transfer=1, n_transfer_games=50, transfer_margin=0.1, **loose)
print("same run twice:", run_poet([weak_entry(), strong_entry()], **args)[1] == run_poet([weak_entry(), strong_entry()], **args)[1])

###############################################################################
# 2) The full run: NEAT learners, reproduction, training, transfer
###############################################################################
#
# `entry` is the pair that grew until it was eligible (the NEAT learner block). Every generation trains
# EVERY entry of the pool (about 6 minutes each), so a generation with 5 entries is about half an hour.
# The pool is kept in `latest` after every generation: if you stop the run (interrupt the console),
# the last finished generation is still there.

latest = {}

def show_and_keep(generation, pool, report):
    show_with_transfer(generation, pool, report)
    latest["pool"], latest["generation"] = list(pool), generation
    for number, member in enumerate(pool):
        ok, check = is_eligible(member, n_games=100, first_seed=230_000)
        print(f"   entry {number}: eligible {ok} | " + "  ".join(f"{k}={v:.2f}" for k, v in check.items()))

pool, history = run_poet(
    [entry], mutation_config, n_generations=10, max_pool=5, n_children=3, max_new=2,
    n_games=100, first_seed=230_000, mutation_seed=1000, train_seed=0, on_generation=show_and_keep,
    n_transfer=2, n_transfer_games=100, transfer_margin=0.1)

###############################################################################
# 2) Watch game
###############################################################################

def outcomes_by_role(entry, seed):
    """Play one game with clones of the entry's agents; return {role: outcome}."""
    agents = {role: agent.clone() for role, agent in entry_agents(entry).items()}
    result, role_of = run_game(entry.layout, seed, agents)
    return {role_of[i]: outcome for i, outcome in result.agent_outcomes.items()}

def watch_entry(entry, seed, delay=0.15):
    """Draw the same game, frame by frame, with fresh clones of the entry's agents."""
    agents = {role: agent.clone() for role, agent in entry_agents(entry).items()}
    watch_game(entry.layout, seed, agents, delay=delay)
    
entry_to_watch = latest["pool"][0]
print(outcomes_by_role(entry_to_watch, 600_000))
watch_entry(entry_to_watch, random.choice(range(600_000,700_000)) )

###############################################################################
# Evaluation
###############################################################################

from collections import Counter

def failed_conditions(report, low=0.2, high=0.7, timeout_cap=0.3):
    """The names of the conditions of the minimal criterion that this report breaks."""
    failed = []
    for role in ("player", "enemy_0"):
        if report[role] < low:
            failed.append(f"{role} < {low}")
        if report[role] > high:
            failed.append(f"{role} > {high}")
    if report["timeout"] > timeout_cap:
        failed.append(f"timeout > {timeout_cap}")
    return failed

parents = [(number, e) for number, e in enumerate(latest["pool"]) if is_eligible(e, 100, 230_000)[0]]
print("eligible parents (positions in the pool):", [number for number, _ in parents])

reasons, accepted, no_mutation, attempts = Counter(), 0, 0, 0
for number, parent in parents:
    mine = 0
    for seed in range(8000, 8008):
        child, report = make_child(parent, mutation_config, seed, 100, 230_000)
        attempts += 1
        if report is None:
            no_mutation += 1
        elif child is not None:
            accepted += 1
            mine += 1
        else:
            reasons.update(failed_conditions(report))
            reasons["(rejected)"] += 1
    print(f"parent {number}: {mine} of 8 accepted")

print(f"\naccepted {accepted} of {attempts} | no valid mutation: {no_mutation}")
print("conditions broken by the rejected children:")
for name, count in reasons.most_common():
    print(f"  {name:18s} {count}")

###############################################################################
# 1) Layouts none of the agents ever saw
###############################################################################

def make_family(config):
    """seed -> a new valid layout from this generation config."""
    return lambda seed: generate_valid_layout(GenerationConfig(**config, seed=seed))

# The density below is my reading of your base layout: 26 interior walls at wall_density 0.1 on
# 16 x 16 = 256 cells. The check after the loop prints the real wall counts so you can confirm it.
families = {
    "the base layout itself":               [layout],
    "new layouts like the base (no traps)": held_out_layouts(make_family(generation_config), 20, 10_000),
    "new layouts like the late pool":       held_out_layouts(
        make_family({**generation_config, "wall_density": 0.17, "num_traps": 1}), 20, 20_000),
}

for name, layouts in families.items():
    pictures = [layout_picture(l) for l in layouts]
    print(f"{name:38s} | {len(layouts):2d} layouts | walls {sum(p.count('#') for p in pictures) / len(pictures):5.1f}"
          f" | traps {sum(p.count('T') for p in pictures) / len(pictures):.1f}")
print("walls of the pool's layouts (for comparison):", [layout_picture(e.layout).count("#") for e in latest["pool"]],
      "| traps", [layout_picture(e.layout).count("T") for e in latest["pool"]])
# Expected: the third family has about as many walls as the pool (98-106) and about one trap

# None of them is a layout of the run. Expected: every distance well above 0
nearest = [min(layout_distance(l, e.layout) for e in latest["pool"]) for l in families["new layouts like the late pool"]]
print("distance of the new layouts to the nearest pool layout:", min(nearest), "...", max(nearest))

###############################################################################
# 2) The skill of every agent, against opponents that do not learn
###############################################################################

GAMES, GAME_SEED = 20, 700_000        # games per layout; the same seeds for every agent

untrained_player = NeatLearner("player", neat_config, quick, quick_champion, seed=0).agent()
untrained_enemy = NeatLearner("enemy_0", neat_config, quick, quick_champion, seed=0).agent()

players = {f"pool entry {i} player": e.learners["player"].agent() for i, e in enumerate(latest["pool"])}
players["untrained (the floor)"] = untrained_player
players["heuristic (a reference)"] = HeuristicAgent()

enemies = {f"pool entry {i} enemy": e.learners["enemy_0"].agent() for i, e in enumerate(latest["pool"])}
enemies["untrained (the floor)"] = untrained_enemy
enemies["heuristic (a reference)"] = HeuristicAgent()

def report(role, agents, opponent, opponent_name):
    for family, layouts in families.items():
        print(f"\n{role} against {opponent_name} | {family}")
        for name, agent in agents.items():
            s = summarize(skill_on_layouts(role, agent, layouts, opponent, GAMES, GAME_SEED))
            print(f"  {name:26s} mean {s['mean']:.2f} | worst layout {s['worst']:.2f} | best {s['best']:.2f}")

report("player", players, None, "the engine's enemy")
report("enemy_0", enemies, HeuristicAgent(), "a heuristic player")


###############################################################################
# Piece: enemy pressure (append this to pipeline_v2.py, AFTER the generalization block)
###############################################################################
#
# Change to the import of the generalization block:
#
#   from poet.generalization import held_out_layouts, pressure_on_layouts, skill_on_layouts, summarize
#
# It reuses `families`, `enemies` (pool enemies, the untrained floor, the heuristic reference) and
# GAME_SEED from the generalization block. About 5 minutes.
#
# Why a new measure: the enemy's own catch rate is squeezed by the player. Against a player that
# wins 85% of the time an enemy can catch at most ~15%, so the rates above sat between 0.05 and
# 0.20 whatever the enemy had learned (2 games out of 20). Here the enemy is judged by what it
# does to a FIXED player: the share of games the player does NOT reach its goal in. Higher means
# a stronger enemy. An enemy that only blocks or delays the player now counts.

PRESSURE_GAMES = 50          # per layout. The family means below rest on 1000 games each

def pressure_report(fixed_player, player_name):
    for family, layouts in families.items():
        print(f"\nenemy pressure on {player_name} | {family}")
        for name, agent in enemies.items():
            s = summarize(pressure_on_layouts(agent, layouts, fixed_player, PRESSURE_GAMES, GAME_SEED))
            print(f"  {name:26s} mean {s['mean']:.2f} | easiest layout {s['worst']:.2f} | hardest {s['best']:.2f}")

pressure_report(HeuristicAgent(), "a heuristic player")

# How to read it:
#  - "untrained (the floor)" is the pressure of an enemy that learned nothing (its walk, the
#    timeouts the player suffers anyway). An enemy has learned something only if it is clearly ABOVE it.
#  - "heuristic (a reference)" is a hand-made enemy.
#  - Read the MEAN of the 20-layout families (about 1000 games each: a difference of 0.05 is
#    meaningful there). For the single base layout (50 games) a difference under 0.1 is noise.
#  - "easiest layout" is the layout where this enemy pressed the player least, "hardest" where it
#    pressed most.