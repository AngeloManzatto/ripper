"""
Created on Sun Oct 18 08:36:49 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

def creates_cycle(genome, src: int, dst: int) -> bool:
    """
    Tell whether adding a connection src -> dst would create a loop.

    A loop appears when dst can already reach src by following existing
    connections forward (a self-loop, src == dst, is the shortest case).
    All connections count, enabled or not, because a disabled one may be
    re-enabled later. The function does not check for duplicates and does
    not modify the genome.

    Parameters
    ----------
    genome : dict
        A genome with a "connections" dictionary.
    src : int
        Id of the node the new connection would start from.
    dst : int
        Id of the node the new connection would point to.

    Returns
    -------
    bool
        True if the new connection would close a loop, False otherwise.
    """
    if src == dst:
        return True

    # Walk forward from dst; if we ever reach src, the new edge closes a loop.
    seen = set()
    stack = [dst]

    while stack:
        node = stack.pop()

        if node == src:
            return True

        if node in seen:
            continue
        seen.add(node)

        for conn in genome["connections"].values():
            if conn["src"] == node:
                stack.append(conn["dst"])

    return False