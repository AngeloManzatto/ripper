"""
Created on Sat Oct 17 20:57:11 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import copy
import numpy as np

from neat.config import default_config
from neat.graph import creates_cycle
from neat.innovations import get_connection_innovation

###############################################################################
# Mutate weights
###############################################################################

def mutate_weights(genome, rng, cfg=None):
    """
    Return a copy of the genome with some connection weights changed.

    Every connection, enabled or not, is visited once. With probability
    "weight_mutate_prob" it is touched; a touched weight is then either
    nudged (probability "weight_perturb_prob": the old weight plus Gaussian
    noise with std "weight_perturb_power") or replaced by a fresh value drawn
    like the starting weights. The result is clipped to
    ["weight_min_value", "weight_max_value"]. Nodes, biases and the structure
    of the network are never changed.

    Parameters
    ----------
    genome : dict
        A genome as built by init_genome. It is not modified.
    rng : numpy.random.Generator
        Source of all randomness.
    cfg : dict, optional
        NEAT configuration; if omitted, default_config() is used. Reads
        "weight_mutate_prob", "weight_perturb_prob", "weight_perturb_power",
        "weight_init_mean", "weight_init_stdev", "weight_min_value" and
        "weight_max_value".

    Returns
    -------
    dict
        A new genome with the same structure and mutated weights.
    """

    if cfg is None:
        cfg = default_config()

    # Work on a copy so the parent genome is never changed.
    new_genome = copy.deepcopy(genome)

    for conn in new_genome["connections"].values():

        # First dice: is this weight touched at all?
        if rng.random() >= cfg["weight_mutate_prob"]:
            continue

        # Second dice: small nudge around the old value, or a fresh value?
        if rng.random() < cfg["weight_perturb_prob"]:

            delta = rng.normal(0.0, cfg["weight_perturb_power"])

            new_w = float(np.clip(
                conn["weight"] + delta,
                cfg["weight_min_value"],
                cfg["weight_max_value"],
            ))

        else:
            new_w = float(np.clip(
                rng.normal(cfg["weight_init_mean"], cfg["weight_init_stdev"]),
                cfg["weight_min_value"],
                cfg["weight_max_value"],
            ))

        # conn is the dictionary inside new_genome, so this updates the copy.
        conn["weight"] = new_w

    return new_genome

###############################################################################
# Mutate bias
###############################################################################

def mutate_biases(genome, rng, cfg=None):
    """
    Return a copy of the genome with some node biases changed.

    Every node except the input nodes (which have no bias) is visited once.
    With probability "bias_mutate_prob" it is touched; a touched bias is then
    either nudged (probability "bias_perturb_prob": the old bias plus Gaussian
    noise with std "bias_perturb_power") or replaced by a fresh value drawn
    like the starting biases. The result is clipped to
    ["bias_min_value", "bias_max_value"]. Connections, node types and
    activations are never changed.

    Parameters
    ----------
    genome : dict
        A genome as built by init_genome. It is not modified.
    rng : numpy.random.Generator
        Source of all randomness.
    cfg : dict, optional
        NEAT configuration; if omitted, default_config() is used. Reads
        "bias_mutate_prob", "bias_perturb_prob", "bias_perturb_power",
        "bias_init_mean", "bias_init_stdev", "bias_min_value" and
        "bias_max_value".

    Returns
    -------
    dict
        A new genome with the same structure and mutated biases.
    """

    if cfg is None:
        cfg = default_config()

    # Work on a copy so the parent genome is never changed.
    new_genome = copy.deepcopy(genome)

    for node in new_genome["nodes"].values():

        # Input nodes have no bias.
        if node["type"] == "input":
            continue

        # First dice: is this bias touched at all?
        if rng.random() >= cfg["bias_mutate_prob"]:
            continue

        # Second dice: small nudge around the old value, or a fresh value?
        if rng.random() < cfg["bias_perturb_prob"]:

            delta = rng.normal(0.0, cfg["bias_perturb_power"])

            new_b = float(np.clip(
                node["bias"] + delta,
                cfg["bias_min_value"],
                cfg["bias_max_value"],
            ))

        else:

            new_b = float(np.clip(
                rng.normal(cfg["bias_init_mean"], cfg["bias_init_stdev"]),
                cfg["bias_min_value"],
                cfg["bias_max_value"],
            ))

        # node is the dictionary inside new_genome, so this updates the copy.
        node["bias"] = new_b

    return new_genome

###############################################################################
# Mutate activation
###############################################################################

def mutate_activation(genome, rng, cfg=None):
    """
    Return a copy of the genome with some node activations changed.

    Every node except the input nodes is visited once. With probability
    "activation_mutate_prob" it is touched, and a touched node switches to a
    DIFFERENT function chosen uniformly from "activation_options". A node
    with nothing else to switch to (its function is the only option) is left
    as it is. Connections, biases and node types are never changed.

    Parameters
    ----------
    genome : dict
        A genome as built by init_genome. It is not modified.
    rng : numpy.random.Generator
        Source of all randomness.
    cfg : dict, optional
        NEAT configuration; if omitted, default_config() is used. Reads
        "activation_mutate_prob" and "activation_options".

    Returns
    -------
    dict
        A new genome with the same structure and mutated activations.
    """

    if cfg is None:
        cfg = default_config()

    # Work on a copy so the parent genome is never changed.
    new_genome = copy.deepcopy(genome)

    for node in new_genome["nodes"].values():

        # Input nodes have no activation.
        if node["type"] == "input":
            continue

        # Dice: is this node touched at all?
        if rng.random() >= cfg["activation_mutate_prob"]:
            continue

        # A touched node must change, so the current function is not a choice.
        choices = []
        for name in cfg["activation_options"]:
            if name != node["activation"]:
                choices.append(name)

        # Nothing to switch to: skip THIS node, but keep visiting the others.
        if not choices:
            continue

        # str() keeps NumPy's string type out of the genome.
        node["activation"] = str(rng.choice(choices))

    return new_genome

def add_connection(genome, rng, innovations, cfg=None):
    """Maybe add one new connection between two existing nodes.

    With probability cfg["conn_add_prob"] one new connection is added,
    otherwise nothing changes. The new connection is chosen at random among
    all valid pairs (src, dst):

    - src can be any node (input, hidden or output);
    - dst can be any node except an input;
    - src != dst (no self loops);
    - the pair is not already in the genome, enabled or disabled;
    - the pair does not create a cycle (disabled connections count).

    If at least one valid pair exists, one is always added. If none exists
    the genome is returned unchanged.

    The new connection is enabled, its weight comes from the initial weight
    distribution (weight_init_mean, weight_init_stdev) and it is stored under
    its innovation number.

    The input genome is never modified. The registry `innovations` IS
    modified in place, but only when a connection is really added (through
    get_connection_innovation).

    Parameters
    ----------
    genome : dict
        Genome to mutate (not modified).
    rng : np.random.Generator
        Source of randomness (required).
    innovations : dict
        Shared innovation registry from init_innovations.
    cfg : dict, optional
        Config from default_config(). Uses conn_add_prob, weight_init_mean
        and weight_init_stdev. Defaults to default_config().

    Returns
    -------
    dict
        A new genome, with one more connection or identical to the input.
    """
    if cfg is None:
        cfg = default_config()

    new_genome = copy.deepcopy(genome)

    # roll the dice once
    if rng.random() >= cfg["conn_add_prob"]:
        return new_genome

    # pairs that already exist (enabled or disabled)
    existing = set()
    for conn in new_genome["connections"].values():
        existing.add((conn["src"], conn["dst"]))

    # every pair that is not a duplicate, self loop or input destination
    node_ids = list(new_genome["nodes"].keys())
    candidates = []
    for src in node_ids:
        for dst in node_ids:
            if new_genome["nodes"][dst]["type"] == "input":
                continue
            if src == dst:
                continue
            if (src, dst) in existing:
                continue
            candidates.append((src, dst))

    # try them in random order, take the first that does not close a loop
    for index in rng.permutation(len(candidates)):
        src, dst = candidates[int(index)]
        if creates_cycle(new_genome, src, dst):
            continue

        weight = float(rng.normal(cfg["weight_init_mean"], cfg["weight_init_stdev"]))
        innovation = get_connection_innovation(innovations, src, dst)
        new_genome["connections"][innovation] = {
            "src": src,
            "dst": dst,
            "weight": weight,
            "enabled": True,
        }
        return new_genome

    return new_genome

def add_node(genome, rng, innovations, cfg=None):
    """Maybe add one hidden node by splitting an existing connection.

    With probability cfg["node_add_prob"] one connection A -> B is split,
    otherwise nothing changes. The connection to split is chosen at random
    among the candidates:

    - the connection must be enabled;
    - a connection whose recorded split node (innovations["splits"]) already
      exists in this genome is skipped, so node ids never repeat.

    Splitting a connection A -> B means:

    - the old connection stays in the genome but is disabled;
    - a new hidden node N is added, with bias 0.0 and activation
      cfg["hidden_activation"];
    - A -> N is added with weight 1.0;
    - N -> B is added with the OLD weight, so the network behaves almost like
      before the split.

    Every genome that splits the same connection gets the same node id N,
    because the id is recorded in innovations["splits"].

    The input genome is never modified. The registry `innovations` IS
    modified in place, but only when a node is really added: a first split of
    a connection stores its node id in "splits" and moves "next_node_id" up
    by one; the two new connections are numbered through
    get_connection_innovation.

    Parameters
    ----------
    genome : dict
        Genome to mutate (not modified).
    rng : np.random.Generator
        Source of randomness (required).
    innovations : dict
        Shared innovation registry from init_innovations.
    cfg : dict, optional
        Config from default_config(). Uses node_add_prob and
        hidden_activation. Defaults to default_config().

    Returns
    -------
    dict
        A new genome, with one more node and two more connections (and one
        connection disabled), or identical to the input.
    """
    if cfg is None:
        cfg = default_config()

    new_genome = copy.deepcopy(genome)

    # roll the dice once
    if rng.random() >= cfg["node_add_prob"]:
        return new_genome

    # connections that can be split
    candidates = []
    for key, conn in new_genome["connections"].items():
        if not conn["enabled"]:
            continue
        known_node = innovations["splits"].get(key)
        if known_node is not None and known_node in new_genome["nodes"]:
            continue
        candidates.append(key)

    if not candidates:
        return new_genome

    old_key = candidates[int(rng.integers(len(candidates)))]
    old = new_genome["connections"][old_key]

    # node id: reuse the one recorded for this connection, or take a new one
    if old_key in innovations["splits"]:
        node_id = innovations["splits"][old_key]
    else:
        node_id = innovations["next_node_id"]
        innovations["splits"][old_key] = node_id
        innovations["next_node_id"] = node_id + 1

    old["enabled"] = False

    new_genome["nodes"][node_id] = {
        "type": "hidden",
        "bias": 0.0,
        "activation": cfg["hidden_activation"],
    }

    in_key = get_connection_innovation(innovations, old["src"], node_id)
    new_genome["connections"][in_key] = {
        "src": old["src"],
        "dst": node_id,
        "weight": 1.0,
        "enabled": True,
    }

    out_key = get_connection_innovation(innovations, node_id, old["dst"])
    new_genome["connections"][out_key] = {
        "src": node_id,
        "dst": old["dst"],
        "weight": old["weight"],
        "enabled": True,
    }

    return new_genome

def mutate(genome, rng, innovations, cfg=None):
    """Apply every mutation once, in a fixed order, and return a new genome.

    Order:
        1. mutate_weights      (existing connections)
        2. mutate_biases       (existing nodes)
        3. mutate_activation   (existing nodes)
        4. add_node            (maybe split one connection)
        5. add_connection      (maybe add one connection)

    The value-changing mutations run FIRST on purpose. A node or connection
    created in steps 4 and 5 keeps exactly the starting values that add_node
    and add_connection gave it (weight 1.0 into a new node, the old weight out
    of it, bias 0.0, the configured hidden activation), instead of being
    nudged in the same call. Step 5 runs after step 4, so a new connection
    can attach to a node that was just created.

    Each step rolls its own dice with its own probability from cfg, so a
    single call usually changes only a few things. The input genome is never
    modified. The registry `innovations` IS modified in place, and only by
    steps 4 and 5 when they really add something.

    Parameters
    ----------
    genome : dict
        Genome to mutate (not modified).
    rng : np.random.Generator
        Source of randomness (required).
    innovations : dict
        Shared innovation registry from init_innovations.
    cfg : dict, optional
        Config from default_config(). Defaults to default_config(). The same
        cfg is given to every step.

    Returns
    -------
    dict
        A new, mutated genome.
    """
    if cfg is None:
        cfg = default_config()

    new_genome = mutate_weights(genome, rng, cfg)
    new_genome = mutate_biases(new_genome, rng, cfg)
    new_genome = mutate_activation(new_genome, rng, cfg)
    new_genome = add_node(new_genome, rng, innovations, cfg)
    new_genome = add_connection(new_genome, rng, innovations, cfg)

    return new_genome