"""
Created on Sat Oct 17 18:40:48 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from neat.config import default_config

###############################################################################
# libraries
###############################################################################

def init_genome(n_inputs, n_outputs, rng, cfg=None):
    """
    Build the smallest valid genome: inputs wired directly to outputs.

    Node ids follow a fixed convention: inputs are 0 .. n_inputs-1, outputs
    follow, and hidden nodes (added later) get higher ids. Every input is
    connected to every output, and the connection keys are the innovation
    numbers, fixed by position: src * n_outputs + (dst - n_inputs).

    Parameters
    ----------
    n_inputs : int
        Number of input nodes.
    n_outputs : int
        Number of output nodes.
    rng : numpy.random.Generator
        Source of all randomness (starting weights and biases). Required, so
        a forgotten rng is an error and not a silent set of identical genomes.
    cfg : dict, optional
        NEAT configuration; if omitted, default_config() is used. Reads
        "weight_init_mean", "weight_init_stdev", "bias_init_mean",
        "bias_init_stdev" and "output_activation".

    Returns
    -------
    dict
        {"n_inputs", "n_outputs",
         "nodes":       {node_id: {"type": "input"}
                         or {"type": "output", "bias", "activation"}},
         "connections": {innovation: {"src", "dst", "weight", "enabled"}}}
    """
    
    if cfg is None:
        cfg = default_config()
    
    genome = {
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
    }

    nodes = {}
    inputs = []
    outputs = []

    for node_id in range(n_inputs):
        nodes[node_id] = {"type": "input"}
        inputs.append(node_id)

    for node_id in range(n_inputs, n_inputs + n_outputs):
        
        bias = float(rng.normal(cfg["bias_init_mean"], 
                                cfg["bias_init_stdev"]))
        
        activation = cfg["output_activation"]
        
        nodes[node_id] = {"type": "output", "bias": bias, "activation": activation}
        outputs.append(node_id)

    # Inputs in the outer loop, outputs in the inner loop: this order is what
    # makes the innovation numbers depend only on the two sizes.
    connection_id = 0
    connections = {}

    for input_node_id in inputs:
        for output_node_id in outputs:
            
            weight = float(rng.normal(cfg["weight_init_mean"], 
                                      cfg["weight_init_stdev"]))
            
            connections[connection_id] = {
                "src": input_node_id,
                "dst": output_node_id,
                "weight": weight,    # plain float, no NumPy type
                "enabled": True,
            }
            connection_id += 1

    genome["nodes"] = nodes
    genome["connections"] = connections

    return genome