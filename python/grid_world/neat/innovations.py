"""
Created on Sun Oct 18 08:51:04 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

def init_innovations(n_inputs, n_outputs):
    """Create the registry for a population of minimal genomes.

    The initial input -> output connections are registered with the same
    numbers that init_genome gives them: src * n_outputs + (dst - n_inputs).

    Parameters
    ----------
    n_inputs : int
        Number of input nodes (ids 0 .. n_inputs - 1).
    n_outputs : int
        Number of output nodes (ids n_inputs .. n_inputs + n_outputs - 1).

    Returns
    -------
    dict
        The registry. For (3, 2) the connections are (0,3):0, (0,4):1,
        (1,3):2, (1,4):3, (2,3):4, (2,4):5, next_innovation is 6,
        next_node_id is 5 and splits is empty.
    """
    connections = {}
    for src in range(n_inputs):
        for dst in range(n_inputs, n_inputs + n_outputs):
            connections[(src, dst)] = src * n_outputs + (dst - n_inputs)

    return {
        "connections": connections,
        "next_innovation": n_inputs * n_outputs,
        "next_node_id": n_inputs + n_outputs,
        "splits": {},
    }


def get_connection_innovation(innovations, src, dst):
    """Return the innovation number of the connection src -> dst.

    A pair already in the registry returns its existing number and nothing
    changes. A new pair takes next_innovation, is stored, and the counter
    moves up by one. Direction matters: (a, b) and (b, a) are different.

    WARNING: the registry is modified IN PLACE (this is the one function in
    the library that does not return a new copy), because every genome in
    the population must see the same numbers.

    Parameters
    ----------
    innovations : dict
        The registry from init_innovations.
    src : int
        Id of the source node.
    dst : int
        Id of the destination node.

    Returns
    -------
    int
        The innovation number for the pair.
    """
    pair = (src, dst)

    if pair in innovations["connections"]:
        return innovations["connections"][pair]

    innovation = innovations["next_innovation"]
    innovations["connections"][pair] = innovation
    innovations["next_innovation"] = innovation + 1
    return innovation