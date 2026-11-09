"""
Created on Mon Oct 19 10:10:20 2026

@author: Angelo Antonio Manzatto

Crossover: build one child genome from two parents.

Genes are matched by identity: connections by innovation number, nodes by id.
Only the ATTRIBUTES of genes that both parents share are mixed. The structure
(which genes exist) always comes from the fitter parent, so the child can never
contain a cycle that the fitter parent does not have.
"""

###############################################################################
# libraries
###############################################################################

import copy

def crossover(parent_a, parent_b, rng):
    """Return a new child genome made from two parents.

    parent_a must be the fitter parent (or equal). The caller decides the
    order, because fitness is not stored in the genome; for equal fitness the
    caller should pick the order at random.

    - The child has exactly the genes of parent_a: the same connection keys and
      the same node ids. Genes that only parent_b has (disjoint or excess) are
      NOT inherited; genes that only parent_a has are copied from parent_a.
    - A gene that BOTH parents have is copied whole from parent_a or from
      parent_b, with probability 1/2 each, independently for every gene.
      A connection keeps its weight and enabled flag from the same parent; a
      node keeps its bias and activation from the same parent. The enabled
      flag is simply inherited (no extra chance to disable it).
    - n_inputs and n_outputs come from parent_a, input nodes are copied as
      they are.

    The child shares nothing with the parents (deep copies) and the parents
    are not modified. The same seed gives the same child.

    Parameters
    ----------
    parent_a : dict
        The fitter parent (or equal).
    parent_b : dict
        The other parent.
    rng : np.random.Generator
        Source of randomness (required).

    Returns
    -------
    dict
        The child genome.

    Raises
    ------
    ValueError
        If the parents cannot come from the same registry: different
        n_inputs or n_outputs, a node id with different types, or an
        innovation number that joins different nodes.
    """
    if (parent_a["n_inputs"] != parent_b["n_inputs"]
            or parent_a["n_outputs"] != parent_b["n_outputs"]):
        raise ValueError("parents have different input/output sizes")

    # node genes: the ids of parent_a, shared ones come from either parent
    nodes = {}
    for node_id in sorted(parent_a["nodes"]):
        node_a = parent_a["nodes"][node_id]
        if node_id in parent_b["nodes"]:
            node_b = parent_b["nodes"][node_id]
            if node_a["type"] != node_b["type"]:
                raise ValueError(f"node {node_id} has different types in the parents")
            if rng.random() < 0.5:
                nodes[node_id] = copy.deepcopy(node_a)
            else:
                nodes[node_id] = copy.deepcopy(node_b)
        else:
            nodes[node_id] = copy.deepcopy(node_a)

    # connection genes: the keys of parent_a, shared ones come from either parent
    connections = {}
    for key in sorted(parent_a["connections"]):
        gene_a = parent_a["connections"][key]
        if key in parent_b["connections"]:
            gene_b = parent_b["connections"][key]
            if gene_a["src"] != gene_b["src"] or gene_a["dst"] != gene_b["dst"]:
                raise ValueError(f"connection {key} joins different nodes in the parents")
            if rng.random() < 0.5:
                connections[key] = copy.deepcopy(gene_a)
            else:
                connections[key] = copy.deepcopy(gene_b)
        else:
            connections[key] = copy.deepcopy(gene_a)

    return {
        "n_inputs": parent_a["n_inputs"],
        "n_outputs": parent_a["n_outputs"],
        "nodes": nodes,
        "connections": connections,
    }