"""
Created on Mon Oct 19 13:33:51 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import copy
import math
 
from neat.check_config import check_config
from neat.config import default_config
from neat.checkpoint import save_checkpoint
from neat.genome import init_genome
from neat.innovations import init_innovations
from neat.speciation import init_species, speciate
from neat.stagnation import update_stagnation, find_stagnant, remove_species
from neat.offspring import compute_offspring
from neat.reproduction import reproduce

###############################################################################
# Initialize population
###############################################################################

def init_population(n_inputs, n_outputs, rng, cfg=None):
    if cfg is None:
        cfg = default_config()
    check_config(cfg)
    genomes = []
    for _ in range(cfg["population_size"]):
        genomes.append(init_genome(n_inputs, n_outputs, rng, cfg))
    return {
        "generation": 0,
        "genomes": genomes,
        "species_set": init_species(),
        "stagnation": {},
        "innovations": init_innovations(n_inputs, n_outputs),
        "best": None,
    }


def evolve_one_generation(state, fitnesses, rng, cfg=None):
    if cfg is None:
        cfg = default_config()
    check_config(cfg)
    genomes = state["genomes"]
    fitnesses = [float(f) for f in fitnesses]
    if len(fitnesses) != len(genomes):
        raise ValueError("need one fitness per genome")
    for fitness in fitnesses:
        if math.isnan(fitness):
            raise ValueError("fitness is NaN")
 
    generation = state["generation"]
    innovations = copy.deepcopy(state["innovations"])
 
    species_set = speciate(genomes, state["species_set"], cfg)
    n_species = len(species_set["species"])
    stagnation = update_stagnation(state["stagnation"], species_set, fitnesses, generation)
    stagnant = find_stagnant(stagnation, generation, cfg)
    species_set = remove_species(species_set, stagnant)
    for species_id in stagnant:
        del stagnation[species_id]
 
    best_index = 0
    for index in range(len(fitnesses)):
        if fitnesses[index] > fitnesses[best_index]:
            best_index = index
    best = state["best"]
    if best is None or fitnesses[best_index] > best["fitness"]:
        best = {"fitness": fitnesses[best_index],
                "genome": copy.deepcopy(genomes[best_index]),
                "generation": generation}
    else:
        best = copy.deepcopy(best)
 
    counts = compute_offspring(species_set, fitnesses, cfg)
    new_genomes = reproduce(genomes, fitnesses, species_set, counts, innovations, rng, cfg)
 
    new_state = {
        "generation": generation + 1,
        "genomes": new_genomes,
        "species_set": species_set,
        "stagnation": stagnation,
        "innovations": innovations,
        "best": best,
    }
    stats = {
        "generation": generation,
        "best_fitness": fitnesses[best_index],
        "mean_fitness": sum(fitnesses) / len(fitnesses),
        "best_ever_fitness": best["fitness"],
        "n_species": n_species,
        "removed_species": stagnant,
    }
    return new_state, stats

def run_evolution(state, evaluate, rng, cfg=None, on_generation=None,
                  checkpoint_path=None, checkpoint_every=None):
    """Run generations until max_generations or fitness_threshold.
 
    Resumable: it continues from state["generation"], so a state loaded with
    load_checkpoint carries on where the saved run stopped.
 
    Checkpointing is off unless both checkpoint_path and checkpoint_every are given.
    The state is then saved (see save_checkpoint) every checkpoint_every generations
    and once more when the run ends.
    """
    if cfg is None:
        cfg = default_config()
    check_config(cfg)
    if (checkpoint_path is None) != (checkpoint_every is None):
        raise ValueError("checkpoint_path and checkpoint_every must be given together")
    if checkpoint_every is not None:
        if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every < 1:
            raise ValueError("checkpoint_every must be an integer of at least 1")
    history = []
    saved_at = None
    while state["generation"] < cfg["max_generations"]:
        fitnesses = evaluate(state["genomes"])
        state, stats = evolve_one_generation(state, fitnesses, rng, cfg)
        history.append(stats)
        if on_generation is not None:
            on_generation(stats)
        if checkpoint_path is not None and state["generation"] % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, state, rng, cfg)
            saved_at = state["generation"]
        threshold = cfg["fitness_threshold"]
        if threshold is not None and state["best"]["fitness"] >= threshold:
            break
    if checkpoint_path is not None and saved_at != state["generation"]:
        save_checkpoint(checkpoint_path, state, rng, cfg)
    return state, history