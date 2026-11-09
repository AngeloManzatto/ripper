"""
Created on Thu Oct 22 21:21:01 2026

@author: Angelo Antonio Manzatto

Diagnostics step 1: record what happens inside the population, generation by generation.

Nothing here changes the evolution. Everything is read from the state and the fitnesses.
A "record" is a plain dict, one per generation. The history is a plain list of records.

"""

###############################################################################
# libraries
###############################################################################

from neat.evolution import evolve_one_generation


def genome_sizes(genomes):
    """Size summary of a list of genomes (mean and max of nodes, hidden, connections, enabled)."""
    if not genomes:
        raise ValueError("genome_sizes needs at least one genome")
    nodes, hidden, conns, enabled = [], [], [], []
    for g in genomes:
        nodes.append(len(g["nodes"]))
        hidden.append(len(g["nodes"]) - g["n_inputs"] - g["n_outputs"])
        conns.append(len(g["connections"]))
        enabled.append(sum(1 for c in g["connections"].values() if c["enabled"]))
    n = len(genomes)
    return {
        "mean_nodes": sum(nodes) / n, "max_nodes": max(nodes),
        "mean_hidden": sum(hidden) / n, "max_hidden": max(hidden),
        "mean_connections": sum(conns) / n, "max_connections": max(conns),
        "mean_enabled": sum(enabled) / n, "max_enabled": max(enabled),
    }


def species_rows(genomes, fitnesses, species_set, stagnation, generation, removed=()):
    """One row per surviving species: id, size, best/mean fitness, generations since it last improved,
    and members (indices into the evaluated genomes).

    evolve_one_generation removes stagnant species before it returns, so the species_set
    it hands back no longer holds their members. `removed` is the list of removed species
    ids (stats["removed_species"]). Genomes not covered by any species are only allowed
    when something was removed; otherwise ValueError.

    Returns (rows, n_removed_genomes).
    """
    if len(genomes) != len(fitnesses):
        raise ValueError(f"{len(genomes)} genomes but {len(fitnesses)} fitnesses")
    seen = sorted(i for s in species_set["species"].values() for i in s["members"])
    if len(set(seen)) != len(seen) or any(i < 0 or i >= len(genomes) for i in seen):
        raise ValueError(
            "species members must be distinct indices of the evaluated genomes; "
            "the species_set seems to describe a different generation"
        )
    n_removed = len(genomes) - len(seen)
    if n_removed and not removed:
        raise ValueError(
            f"{n_removed} genomes belong to no species although no species was removed"
        )
    rows = []
    for sid in sorted(species_set["species"]):
        members = species_set["species"][sid]["members"]
        values = [fitnesses[i] for i in members]
        record = stagnation.get(sid)
        since = None if record is None else generation - record["last_improved"]
        rows.append({
            "id": sid, "size": len(members),
            "best": max(values), "mean": sum(values) / len(values),
            "since_improved": since,
            "members": list(members),
        })
    return rows, n_removed


def make_record(genomes, fitnesses, new_state, stats):
    """Build the record of one generation.

    genomes, fitnesses : what was evaluated this generation
    new_state, stats   : what evolve_one_generation returned for it
    """
    rows, n_removed = species_rows(genomes, fitnesses, new_state["species_set"],
                                   new_state["stagnation"], stats["generation"],
                                   stats["removed_species"])
    return {
        **stats,
        **genome_sizes(genomes),
        "species": rows,
        "removed_genomes": n_removed,
        "next_innovation": new_state["innovations"]["next_innovation"],
        "next_node_id": new_state["innovations"]["next_node_id"],
    }


def run_recorded(state, evaluate, rng, cfg, keep_genomes=False):
    """Same loop as run_evolution, but also returns one full record per generation.

    Returns (state, history) where history is a list of records (see make_record).
    Like run_evolution it continues from state["generation"] and stops at
    cfg["max_generations"] or when the best fitness reaches cfg["fitness_threshold"].

    keep_genomes=True also stores the evaluated genomes in each record under "genomes"
    (needed for plot_distances on a past generation; costs memory).
    """
    history = []
    while state["generation"] < cfg["max_generations"]:
        genomes = state["genomes"]
        fitnesses = list(evaluate(genomes))
        new_state, stats = evolve_one_generation(state, fitnesses, rng, cfg)
        record = make_record(genomes, fitnesses, new_state, stats)
        if keep_genomes:
            record["genomes"] = genomes
        history.append(record)
        state = new_state
        threshold = cfg["fitness_threshold"]
        if threshold is not None and state["best"]["fitness"] >= threshold:
            break
    return state, history