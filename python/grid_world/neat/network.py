"""
Created on Mon Oct 19 14:57:21 2026

@author: Angelo Antonio Manzatto

Compiled feed-forward network: turn a genome into something fast to run.

    network = compile_genome(genome)      # once per genome
    outputs = activate(network, inputs)   # many times (once per game step)

The network is plain data: n_inputs, n_outputs, n_slots, output_slots and a
list of steps (slot, bias, activation function, [(source slot, weight), ...])
in an order where every node comes after the nodes that feed it. Only enabled
connections are used, and nodes that cannot reach an output are left out.
Slots 0..n_inputs-1 hold the inputs, the others the computed nodes.

Recurrent connections are not supported yet: a cycle raises ValueError.
Activation functions are module-level (not lambdas) so a network can be
pickled and sent to worker processes.

"""

###############################################################################
# libraries
###############################################################################

import math

from neat.activations import ACTIVATIONS, get_activation

###############################################################################
# Compile genome
###############################################################################

def compile_genome(genome):
    nodes = genome["nodes"]
    input_ids = sorted(n for n, node in nodes.items() if node["type"] == "input")
    output_ids = sorted(n for n, node in nodes.items() if node["type"] == "output")

    for node_id, node in nodes.items():
        if node["type"] != "input" and node["activation"] not in ACTIVATIONS:
            raise ValueError(f"unknown activation {node['activation']!r} on node {node_id}")

    enabled = []
    for key in sorted(genome["connections"]):
        c = genome["connections"][key]
        if c["src"] not in nodes or c["dst"] not in nodes:
            raise ValueError(f"connection {key} uses a node that does not exist")
        if nodes[c["dst"]]["type"] == "input":
            raise ValueError(f"connection {key} goes into an input node")
        if c["enabled"]:
            enabled.append(c)

    # keep only the nodes that can reach an output
    needed = set(output_ids)
    changed = True
    while changed:
        changed = False
        for c in enabled:
            if c["dst"] in needed and c["src"] not in needed:
                needed.add(c["src"])
                changed = True

    incoming = {n: [] for n in needed}
    waiting = {n: 0 for n in needed}
    for c in enabled:
        if c["dst"] in needed:
            incoming[c["dst"]].append(c)
            if nodes[c["src"]]["type"] != "input":
                waiting[c["dst"]] += 1

    # topological order; inputs have no incoming connections so they start ready
    slot_of = {}
    for node_id in input_ids:
        slot_of[node_id] = len(slot_of)
    ready = sorted(n for n in needed if nodes[n]["type"] != "input" and waiting[n] == 0)
    order = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        slot_of[node_id] = len(slot_of)
        for c in enabled:
            if c["src"] == node_id and c["dst"] in needed:
                waiting[c["dst"]] -= 1
                if waiting[c["dst"]] == 0:
                    ready.append(c["dst"])
        ready.sort()
    if len(order) != sum(1 for n in needed if nodes[n]["type"] != "input"):
        raise ValueError("genome has a cycle")

    steps = []
    for node_id in order:
        links = [(slot_of[c["src"]], c["weight"]) for c in incoming[node_id]]
        steps.append((slot_of[node_id], nodes[node_id]["bias"],
                      get_activation(nodes[node_id]["activation"]), links))

    return {
        "n_inputs": len(input_ids),
        "n_outputs": len(output_ids),
        "n_slots": len(input_ids) + len(order),
        "output_slots": [slot_of[n] for n in output_ids],
        "steps": steps,
    }

###############################################################################
# Activate
###############################################################################

def activate(network, inputs):
    n_inputs = network["n_inputs"]
    if len(inputs) != n_inputs:
        raise ValueError(f"expected {n_inputs} inputs, got {len(inputs)}")
    values = [0.0] * network["n_slots"]
    for i in range(n_inputs):
        x = float(inputs[i])
        if not math.isfinite(x):
            raise ValueError(f"input {i} is not finite")
        values[i] = x
    for slot, bias, function, links in network["steps"]:
        total = bias
        for src, weight in links:
            total += values[src] * weight
        values[slot] = function(total)
    return [values[slot] for slot in network["output_slots"]]