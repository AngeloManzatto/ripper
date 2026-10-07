"""
Created on Sat Jun  6 18:57:37 2026

@author: Angelo Antonio Manzatto
------------------
Innovation number registry for the NEAT algorithm.

What innovation numbers do
--------------------------
When two genomes independently add the same structural connection
(src → dst), they must receive the same innovation number. This shared
number allows crossover to align corresponding genes across genomes.

Without shared innovation numbers, crossover would be arbitrary and
offspring would have broken topologies.

Two structural event types are tracked:

    AddConnection(src, dst)  → one innovation number (the connection)
    AddNode(connection_innov) → two innovation numbers (the two new
                                connections) + one new node id

The registry maps event keys to their assigned numbers so the same
structural event always gets the same number within a single run.

Scope
-----
One InnovationRegistry instance lives for the lifetime of one evolution
run. It is created by the Population and passed to mutation operators.
Never share a registry across independent runs — that would corrupt the
historical gene alignment.

Generation reset (optional)
----------------------------
The original NEAT paper resets the registry each generation so that the
same mutation in different generations gets different innovation numbers.
In practice most implementations reuse numbers across generations (simpler
and rarely harmful). We support both modes via reset_every_generation.

Thread safety
-------------
InnovationRegistry is not thread-safe. Mutation must be serialised if
running in parallel. Parallelise fitness evaluation instead.
"""

###############################################################################
# libraries
###############################################################################

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


###############################################################################
# Event keys
###############################################################################

# A connection event is identified by (src_node_id, dst_node_id).
ConnectionKey = Tuple[int, int]

# A node-split event is identified by the innovation number of the
# connection being split. Splitting the same connection always produces
# the same intermediate node.
NodeSplitKey = int


###############################################################################
# Registry
###############################################################################

@dataclass
class InnovationRegistry:
    """
    Tracks innovation numbers and node ids for one evolution run.

    Attributes
    ----------
    reset_every_generation:
        If True, the connection→innovation mapping is cleared at the
        start of each generation so that structurally identical mutations
        in different generations receive different innovation numbers.
        If False (default), numbers are reused across generations.
    """

    reset_every_generation: bool = False

    # Innovation counter — increments with every new structural event.
    _next_innovation: int = field(default=1, init=False, repr=False)

    # Node id counter — increments with every new hidden node.
    # Caller is responsible for initialising this to
    # n_inputs + n_outputs so that new hidden node ids don't collide
    # with existing input/output ids.
    _next_node_id: int = field(default=0, init=False, repr=False)

    # Persistent maps — never cleared (historical record).
    # _node_split_map: connection_innov → (innov_a, innov_b, node_id)
    _node_split_map: Dict[NodeSplitKey, Tuple[int, int, int]] = field(
        default_factory=dict, init=False, repr=False
    )

    # Generation-scoped map — cleared each generation if
    # reset_every_generation is True.
    # _conn_map: (src, dst) → innovation_number
    _conn_map: Dict[ConnectionKey, int] = field(
        default_factory=dict, init=False, repr=False
    )

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialise(self, n_inputs: int, n_outputs: int) -> None:
        """
        Set the node id counter so that new hidden nodes don't collide
        with input/output node ids.

        Call once after creating the registry, before any mutations.

        Args:
            n_inputs:  Number of input nodes in the genome.
            n_outputs: Number of output nodes in the genome.
        """
        self._next_node_id = n_inputs + n_outputs

    # ── Generation boundary ───────────────────────────────────────────────────

    def next_generation(self) -> None:
        """
        Signal the start of a new generation.

        If reset_every_generation is True, clears the connection map so
        that the same structural mutation in this generation will receive
        a fresh innovation number.

        The node-split map is never cleared — splitting the same
        connection must always produce the same intermediate node id
        across all generations.
        """
        if self.reset_every_generation:
            self._conn_map.clear()

    # ── Add connection ────────────────────────────────────────────────────────

    def get_connection_innovation(
        self,
        src_node_id: int,
        dst_node_id: int,
    ) -> int:
        """
        Return the innovation number for a connection (src → dst).

        If this connection has been seen before in the current generation
        scope, return the existing number. Otherwise assign a new one.

        Args:
            src_node_id: Source node id.
            dst_node_id: Destination node id.

        Returns:
            Innovation number for this connection.
        """
        key = (src_node_id, dst_node_id)

        if key in self._conn_map:
            return self._conn_map[key]

        innov = self._next_innovation
        self._next_innovation += 1
        self._conn_map[key] = innov
        return innov

    # ── Add node (split connection) ───────────────────────────────────────────

    def get_node_split(
        self,
        connection_innovation: int,
    ) -> Tuple[int, int, int]:
        """
        Return (innov_a, innov_b, new_node_id) for splitting a connection.

        Splitting the same connection always produces the same result —
        the node-split map is never reset between generations.

        The split produces:
            - src  --innov_a-->  new_node  --innov_b-->  dst
            (two new enabled connections replacing one disabled connection)

        Args:
            connection_innovation:
                Innovation number of the connection being split.

        Returns:
            Tuple of (innovation_a, innovation_b, new_node_id) where:
                innovation_a = innovation for src → new_node connection
                innovation_b = innovation for new_node → dst connection
                new_node_id  = id assigned to the new hidden node
        """
        if connection_innovation in self._node_split_map:
            return self._node_split_map[connection_innovation]

        innov_a    = self._next_innovation;     self._next_innovation += 1
        innov_b    = self._next_innovation;     self._next_innovation += 1
        new_node   = self._next_node_id;        self._next_node_id    += 1

        result = (innov_a, innov_b, new_node)
        self._node_split_map[connection_innovation] = result
        return result

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def current_innovation(self) -> int:
        """The next innovation number that will be assigned."""
        return self._next_innovation

    @property
    def current_node_id(self) -> int:
        """The next node id that will be assigned to a hidden node."""
        return self._next_node_id

    @property
    def n_connection_events(self) -> int:
        """Total number of unique connection events ever registered."""
        return len(self._conn_map)

    @property
    def n_node_split_events(self) -> int:
        """Total number of unique node-split events ever registered."""
        return len(self._node_split_map)

    def has_connection(self, src_node_id: int, dst_node_id: int) -> bool:
        """Return True if this connection has been seen this generation."""
        return (src_node_id, dst_node_id) in self._conn_map

    def has_node_split(self, connection_innovation: int) -> bool:
        """Return True if this connection has been split before."""
        return connection_innovation in self._node_split_map

    def __repr__(self) -> str:
        return (
            f"InnovationRegistry("
            f"next_innov={self._next_innovation}, "
            f"next_node={self._next_node_id}, "
            f"conn_events={self.n_connection_events}, "
            f"split_events={self.n_node_split_events}, "
            f"reset_per_gen={self.reset_every_generation})"
        )