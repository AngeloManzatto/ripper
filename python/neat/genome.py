"""
Created on Sat Jun  6 18:59:22 2026

@author: Angelo Antonio Manzatto
--------------
Genome: the central data structure of NEAT.

A Genome encodes a neural network as two collections of genes:
    nodes:       Dict[node_id   → NodeGene]
    connections: Dict[innovation → ConnectionGene]

It owns:
    - Network topology (add/remove nodes and connections)
    - Forward pass (evaluate inputs → outputs)
    - Weight and bias mutations
    - Structural mutations (add_connection, add_node)
    - Compatibility distance (for speciation)
    - Crossover (class method, produces new offspring)
    - Serialization (to_dict / from_dict)

Design
------
- Genome is a regular (mutable) dataclass — genes accumulate over time.
- NodeGene and ConnectionGene are frozen — mutations return new genes.
- The forward pass uses a local value dict — no mutable state on genes.
- All randomness goes through a numpy Generator passed as argument.
- All hyperparameters come from NEATConfig passed as argument.
- InnovationRegistry is passed to structural mutations — never stored.

Forward pass
------------
Nodes are evaluated in topological order. Cycles are detected and
handled according to the allow_recurrent parameter:
    allow_recurrent=False: ValueError if cycles exist
    allow_recurrent=True:  cyclic nodes use value 0.0 for their
                           inputs (first pass approximation)

Crossover
---------
Follows the original NEAT paper:
    - Matching genes (same innovation): inherited from fitter parent,
      or randomly from either if equal fitness
    - Disjoint/excess genes: inherited from fitter parent only
    - If equal fitness: disjoint/excess inherited from both randomly
"""

###############################################################################
# libraries
###############################################################################

import json
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.neat.genes import NodeGene, ConnectionGene, NodeType
from src.neat.activations import DEFAULT_REGISTRY, ActivationRegistry
from src.neat.innovation import InnovationRegistry
from src.neat.config import NEATConfig

###############################################################################
# Genome
###############################################################################

@dataclass
class Genome:
    """
    A single individual in the NEAT population.

    Attributes
    ----------
    genome_id : int
        Unique identifier within the current run.
    n_inputs : int
        Number of input nodes. Their ids are 0..n_inputs-1.
    n_outputs : int
        Number of output nodes. Their ids are n_inputs..n_inputs+n_outputs-1.
    nodes : Dict[int, NodeGene]
        All node genes keyed by node_id.
    connections : Dict[int, ConnectionGene]
        All connection genes keyed by innovation number.
    fitness : Optional[float]
        Assigned by the fitness evaluator. None until evaluated.
    species_id : Optional[int]
        Assigned by the speciation step. None until speciated.
    """

    genome_id:   int
    n_inputs:    int
    n_outputs:   int

    nodes:       Dict[int, NodeGene]       = field(default_factory=dict)
    connections: Dict[int, ConnectionGene] = field(default_factory=dict)

    fitness:     Optional[float] = field(default=None, compare=False)
    species_id:  Optional[int]   = field(default=None, compare=False)

    # ── Node accessors ────────────────────────────────────────────────────────

    @property
    def input_node_ids(self) -> List[int]:
        """Sorted list of input node ids."""
        return list(range(self.n_inputs))

    @property
    def output_node_ids(self) -> List[int]:
        """Sorted list of output node ids."""
        return list(range(self.n_inputs, self.n_inputs + self.n_outputs))

    @property
    def hidden_node_ids(self) -> List[int]:
        """Sorted list of hidden node ids."""
        reserved = set(self.input_node_ids) | set(self.output_node_ids)
        return sorted(nid for nid in self.nodes if nid not in reserved)

    @property
    def enabled_connections(self) -> List[ConnectionGene]:
        """All enabled connection genes."""
        return [c for c in self.connections.values() if c.enabled]

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        inputs: List[float],
        *,
        allow_recurrent: bool = False,
        registry: ActivationRegistry = DEFAULT_REGISTRY,
    ) -> List[float]:
        """
        Evaluate the network on a list of input values.

        Parameters
        ----------
        inputs : List[float]
            Must have exactly n_inputs elements.
        allow_recurrent : bool
            If False, raises ValueError when cycles are detected.
            If True, cyclic nodes are initialised to 0.0.
        registry : ActivationRegistry
            Used to resolve activation function names.

        Returns
        -------
        List[float]
            Output values in output node id order.

        Raises
        ------
        ValueError
            If len(inputs) != n_inputs, or cycles detected with
            allow_recurrent=False.
        """
        if len(inputs) != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} inputs, got {len(inputs)}"
            )

        # build adjacency for enabled connections only
        # incoming[dst] = list of (src, weight)
        incoming: Dict[int, List[Tuple[int, float]]] = {
            nid: [] for nid in self.nodes
        }
        for conn in self.enabled_connections:
            incoming[conn.dst_node_id].append(
                (conn.src_node_id, conn.weight)
            )

        # topological order
        order = _topological_sort(
            list(self.nodes.keys()),
            incoming,
            allow_recurrent=allow_recurrent,
        )

        # evaluate
        values: Dict[int, float] = {}

        # seed input nodes directly — no activation, no bias
        for i, nid in enumerate(self.input_node_ids):
            values[nid] = float(inputs[i])

        for nid in order:
            node = self.nodes[nid]
            if node.is_input:
                continue  # already seeded

            raw = node.bias + sum(
                values.get(src, 0.0) * w
                for src, w in incoming[nid]
            )
            fn = registry.get_or_raise(node.activation)
            values[nid] = fn(raw)

        return [values[nid] for nid in self.output_node_ids]

    # ── Weight / bias mutations ───────────────────────────────────────────────

    def mutate_weights(
        self,
        cfg: NEATConfig,
        rng: np.random.Generator,
    ) -> None:
        """
        Mutate connection weights in place.

        For each connection, with probability weight_mutate_prob:
            - with probability weight_perturb_prob: add Gaussian noise
            - otherwise: replace with a new random weight
        """
        for innov, conn in list(self.connections.items()):
            if rng.random() >= cfg.weight_mutate_prob:
                continue
            if rng.random() < cfg.weight_perturb_prob:
                delta = rng.normal(0.0, cfg.weight_perturb_power)
                new_w = float(np.clip(
                    conn.weight + delta,
                    cfg.weight_min_value,
                    cfg.weight_max_value,
                ))
            else:
                new_w = float(np.clip(
                    rng.normal(cfg.weight_init_mean, cfg.weight_init_stdev),
                    cfg.weight_min_value,
                    cfg.weight_max_value,
                ))
            self.connections[innov] = conn.with_weight(new_w)

    def mutate_biases(
        self,
        cfg: NEATConfig,
        rng: np.random.Generator,
    ) -> None:
        """
        Mutate node biases in place.

        For each non-input node, with probability bias_mutate_prob:
            - with probability bias_perturb_prob: add Gaussian noise
            - otherwise: replace with a new random bias
        """
        for nid, node in list(self.nodes.items()):
            if node.is_input:
                continue
            if rng.random() >= cfg.bias_mutate_prob:
                continue
            if rng.random() < cfg.bias_perturb_prob:
                delta = rng.normal(0.0, cfg.bias_perturb_power)
                new_b = float(np.clip(
                    node.bias + delta,
                    cfg.bias_min_value,
                    cfg.bias_max_value,
                ))
            else:
                new_b = float(np.clip(
                    rng.normal(cfg.bias_init_mean, cfg.bias_init_stdev),
                    cfg.bias_min_value,
                    cfg.bias_max_value,
                ))
            self.nodes[nid] = node.with_bias(new_b)

    def mutate_activations(
        self,
        cfg: NEATConfig,
        rng: np.random.Generator,
    ) -> None:
        """
        Mutate node activation functions in place.

        For each non-input node, with probability activation_mutate_prob:
            replace activation with a random choice from cfg.activation_options.
        Input nodes always use identity and are never mutated.
        """
        for nid, node in list(self.nodes.items()):
            if node.is_input:
                continue
            if rng.random() >= cfg.activation_mutate_prob:
                continue
            new_act = str(rng.choice(cfg.activation_options))
            self.nodes[nid] = node.with_activation(new_act)

    # ── Structural mutations ──────────────────────────────────────────────────

    def mutate_add_connection(
        self,
        cfg: NEATConfig,
        rng: np.random.Generator,
        registry: InnovationRegistry,
        *,
        max_attempts: int = 20,
    ) -> bool:
        """
        Attempt to add a new connection gene.

        Picks a random (src, dst) pair that does not already have an
        enabled connection. Skips pairs that would connect to an input
        node (inputs never receive connections) or create a disallowed
        self-loop.

        Parameters
        ----------
        max_attempts : int
            Number of random pairs to try before giving up.

        Returns
        -------
        bool
            True if a connection was successfully added.
        """
        if rng.random() >= cfg.conn_add_prob:
            return False

        all_ids = list(self.nodes.keys())

        # existing enabled (src, dst) pairs
        existing: Set[Tuple[int, int]] = {
            (c.src_node_id, c.dst_node_id)
            for c in self.connections.values()
            if c.enabled
        }

        input_ids = set(self.input_node_ids)

        for _ in range(max_attempts):
            src = int(rng.choice(all_ids))
            dst = int(rng.choice(all_ids))

            # dst must not be an input node
            if dst in input_ids:
                continue
            # avoid duplicate enabled connections
            if (src, dst) in existing:
                continue

            innov  = registry.get_connection_innovation(src, dst)
            weight = float(np.clip(
                rng.normal(cfg.weight_init_mean, cfg.weight_init_stdev),
                cfg.weight_min_value,
                cfg.weight_max_value,
            ))
            conn = ConnectionGene(innov, src, dst, weight, enabled=True)
            self.connections[innov] = conn
            return True

        return False

    def mutate_add_node(
        self,
        cfg: NEATConfig,
        rng: np.random.Generator,
        registry: InnovationRegistry,
    ) -> bool:
        """
        Add a new hidden node by splitting an existing enabled connection.

        The original connection is disabled. Two new connections are added:
            src  --weight=1.0-->  new_node  --weight=old_weight-->  dst

        This preserves the network's behaviour at the moment of mutation.

        Returns
        -------
        bool
            True if a node was successfully added.
        """
        if rng.random() >= cfg.node_add_prob:
            return False

        enabled = self.enabled_connections
        if not enabled:
            return False

        # pick a random enabled connection to split
        idx  = int(rng.integers(len(enabled)))
        conn = enabled[idx]

        innov_a, innov_b, new_node_id = registry.get_node_split(conn.innovation)

        # disable the original connection
        self.connections[conn.innovation] = conn.disabled_copy()

        # add the new hidden node
        new_node = NodeGene(
            node_id    = new_node_id,
            node_type  = NodeType.HIDDEN,
            bias       = 0.0,
            activation = cfg.hidden_activation,
        )
        self.nodes[new_node_id] = new_node

        # add two new connections
        conn_a = ConnectionGene(innov_a, conn.src_node_id, new_node_id,
                                weight=1.0, enabled=True)
        conn_b = ConnectionGene(innov_b, new_node_id, conn.dst_node_id,
                                weight=conn.weight, enabled=True)
        self.connections[innov_a] = conn_a
        self.connections[innov_b] = conn_b

        return True

    def mutate_delete_connection(
        self,
        cfg: NEATConfig,
        rng: np.random.Generator,
    ) -> bool:
        """
        Delete a random connection gene entirely.

        Returns True if a connection was deleted.
        Note: this is a hard delete, not a disable. The gene is gone.
        """
        if rng.random() >= cfg.conn_delete_prob:
            return False

        if not self.connections:
            return False

        innov = int(rng.choice(list(self.connections.keys())))
        del self.connections[innov]
        return True

    def mutate_delete_node(
        self,
        cfg: NEATConfig,
        rng: np.random.Generator,
    ) -> bool:
        """
        Delete a random hidden node and all its connections.

        Input and output nodes are never deleted.
        Returns True if a node was deleted.
        """
        if rng.random() >= cfg.node_delete_prob:
            return False

        hidden = self.hidden_node_ids
        if not hidden:
            return False

        nid = int(rng.choice(hidden))
        del self.nodes[nid]

        # remove all connections touching this node
        to_remove = [
            innov for innov, conn in self.connections.items()
            if conn.src_node_id == nid or conn.dst_node_id == nid
        ]
        for innov in to_remove:
            del self.connections[innov]

        return True

    def mutate(
        self,
        cfg: NEATConfig,
        rng: np.random.Generator,
        registry: InnovationRegistry,
    ) -> None:
        """
        Apply all mutations in sequence.

        Order follows NEAT convention:
        1. Weight mutations (most frequent)
        2. Bias mutations
        3. Activation mutations
        4. Structural: add connection
        5. Structural: add node
        6. Structural: delete connection
        7. Structural: delete node
        """
        self.mutate_weights(cfg, rng)
        self.mutate_biases(cfg, rng)
        self.mutate_activations(cfg, rng)
        self.mutate_add_connection(cfg, rng, registry)
        self.mutate_add_node(cfg, rng, registry)
        self.mutate_delete_connection(cfg, rng)
        self.mutate_delete_node(cfg, rng)

    # ── Compatibility distance ─────────────────────────────────────────────────

    def compatibility_distance(
        self,
        other: "Genome",
        cfg: NEATConfig,
    ) -> float:
        """
        Compute the compatibility distance between this genome and another.

        Formula (NEAT paper):
            d = c1*E/N + c2*D/N + c3*W

        where:
            E = number of excess genes
            D = number of disjoint genes
            W = average weight difference of matching genes
            N = number of genes in the larger genome
              (clamped to compatibility_n_min for small genomes)
            c1, c2, c3 = coefficients from cfg

        Returns
        -------
        float
            Compatibility distance. 0.0 means identical topology.
        """
        innov_a = set(self.connections.keys())
        innov_b = set(other.connections.keys())

        matching   = innov_a & innov_b
        all_innov  = innov_a | innov_b

        max_innov_a = max(innov_a) if innov_a else 0
        max_innov_b = max(innov_b) if innov_b else 0

        disjoint = 0
        excess   = 0

        for innov in all_innov:
            if innov in matching:
                continue
            # gene is in one genome but not the other
            in_a = innov in innov_a
            in_b = innov in innov_b

            # excess: beyond the range of the smaller genome
            if (in_a and innov > max_innov_b) or \
               (in_b and innov > max_innov_a):
                excess += 1
            else:
                disjoint += 1

        # average weight difference for matching genes
        if matching:
            w_diff = sum(
                abs(self.connections[i].weight - other.connections[i].weight)
                for i in matching
            ) / len(matching)
        else:
            w_diff = 0.0

        # normalisation factor
        n_genes = max(len(innov_a), len(innov_b))
        N = max(n_genes, cfg.compatibility_n_min)

        return (
            cfg.compatibility_c1 * excess   / N +
            cfg.compatibility_c2 * disjoint / N +
            cfg.compatibility_c3 * w_diff
        )

    # ── Crossover ─────────────────────────────────────────────────────────────

    @classmethod
    def crossover(
        cls,
        parent_a: "Genome",
        parent_b: "Genome",
        offspring_id: int,
        rng: np.random.Generator,
    ) -> "Genome":
        """
        Produce an offspring genome by crossing two parents.

        Rules (NEAT paper):
            Matching genes (same innovation):
                inherited randomly from either parent (50/50)
            Disjoint/excess genes:
                inherited from the fitter parent only
                if equal fitness: inherited randomly from either

        Node genes:
            Inherited from the parent that contributed the connection gene,
            plus all nodes from the fitter parent.

        Parameters
        ----------
        parent_a, parent_b : Genome
            The two parents. Order matters only for tie-breaking.
        offspring_id : int
            The genome_id assigned to the new offspring.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        Genome
            A new genome (neither parent is modified).
        """
        fit_a = parent_a.fitness or 0.0
        fit_b = parent_b.fitness or 0.0

        # identify the fitter parent
        if fit_a > fit_b:
            dominant, recessive = parent_a, parent_b
        elif fit_b > fit_a:
            dominant, recessive = parent_b, parent_a
        else:
            # equal fitness — treat parent_a as dominant arbitrarily
            dominant, recessive = parent_a, parent_b
            equal_fitness = True

        equal_fitness = (fit_a == fit_b)

        innov_dom = set(dominant.connections.keys())
        innov_rec = set(recessive.connections.keys())
        matching  = innov_dom & innov_rec

        offspring_conns: Dict[int, ConnectionGene] = {}

        # matching genes: pick from either parent randomly
        for innov in matching:
            if rng.random() < 0.5:
                offspring_conns[innov] = dominant.connections[innov]
            else:
                offspring_conns[innov] = recessive.connections[innov]

        # disjoint / excess from dominant
        for innov in innov_dom - matching:
            offspring_conns[innov] = dominant.connections[innov]

        # disjoint / excess from recessive — only if equal fitness
        if equal_fitness:
            for innov in innov_rec - matching:
                if rng.random() < 0.5:
                    offspring_conns[innov] = recessive.connections[innov]

        # collect node ids needed by inherited connections
        needed_node_ids: Set[int] = set()
        for conn in offspring_conns.values():
            needed_node_ids.add(conn.src_node_id)
            needed_node_ids.add(conn.dst_node_id)

        # always include all input and output nodes from dominant
        needed_node_ids.update(dominant.input_node_ids)
        needed_node_ids.update(dominant.output_node_ids)

        # build node dict from dominant; fall back to recessive
        offspring_nodes: Dict[int, NodeGene] = {}
        for nid in needed_node_ids:
            if nid in dominant.nodes:
                offspring_nodes[nid] = dominant.nodes[nid]
            elif nid in recessive.nodes:
                offspring_nodes[nid] = recessive.nodes[nid]

        return cls(
            genome_id   = offspring_id,
            n_inputs    = dominant.n_inputs,
            n_outputs   = dominant.n_outputs,
            nodes       = offspring_nodes,
            connections = offspring_conns,
        )

    # ── Copying ───────────────────────────────────────────────────────────────

    def copy(self, new_id: int) -> "Genome":
        """
        Return a deep copy with a new genome_id.
        Fitness and species_id are not copied.
        """
        return Genome(
            genome_id   = new_id,
            n_inputs    = self.n_inputs,
            n_outputs   = self.n_outputs,
            nodes       = dict(self.nodes),
            connections = dict(self.connections),
        )

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id":   self.genome_id,
            "n_inputs":    self.n_inputs,
            "n_outputs":   self.n_outputs,
            "fitness":     self.fitness,
            "species_id":  self.species_id,
            "nodes":       [n.to_dict() for n in self.nodes.values()],
            "connections": [c.to_dict() for c in self.connections.values()],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Genome":
        nodes = {
            nd["node_id"]: NodeGene.from_dict(nd)
            for nd in d["nodes"]
        }
        connections = {
            cd["innovation"]: ConnectionGene.from_dict(cd)
            for cd in d["connections"]
        }
        g = cls(
            genome_id   = int(d["genome_id"]),
            n_inputs    = int(d["n_inputs"]),
            n_outputs   = int(d["n_outputs"]),
            nodes       = nodes,
            connections = connections,
        )
        g.fitness    = d.get("fitness")
        g.species_id = d.get("species_id")
        return g

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        return cls.from_dict(json.loads(s))

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Genome("
            f"id={self.genome_id}, "
            f"in={self.n_inputs}, "
            f"out={self.n_outputs}, "
            f"nodes={len(self.nodes)}, "
            f"conns={len(self.connections)}, "
            f"enabled={len(self.enabled_connections)}, "
            f"fitness={self.fitness})"
        )


###############################################################################
# Genome factory
###############################################################################

def make_minimal_genome(
    genome_id: int,
    n_inputs:  int,
    n_outputs: int,
    cfg:       NEATConfig,
    registry:  InnovationRegistry,
    rng:       np.random.Generator,
) -> Genome:
    """
    Create a minimal genome with input and output nodes fully connected.

    This is the standard NEAT starting point: no hidden nodes,
    all inputs connected to all outputs with random weights.

    Parameters
    ----------
    genome_id : int
        Id assigned to this genome.
    n_inputs, n_outputs : int
        Network dimensions.
    cfg : NEATConfig
        Hyperparameters for weight initialisation and activations.
    registry : InnovationRegistry
        Assigns innovation numbers to initial connections.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    Genome
        A minimal connected genome ready for evolution.
    """
    nodes: Dict[int, NodeGene] = {}

    # input nodes
    for i in range(n_inputs):
        nodes[i] = NodeGene(i, NodeType.INPUT, bias=0.0, activation="identity")

    # output nodes
    for i in range(n_outputs):
        nid = n_inputs + i
        nodes[nid] = NodeGene(
            nid, NodeType.OUTPUT,
            bias=float(np.clip(
                rng.normal(cfg.bias_init_mean, cfg.bias_init_stdev),
                cfg.bias_min_value, cfg.bias_max_value,
            )),
            activation=cfg.output_activation,
        )

    # fully connect inputs to outputs
    connections: Dict[int, ConnectionGene] = {}
    for i in range(n_inputs):
        for j in range(n_outputs):
            dst = n_inputs + j
            innov = registry.get_connection_innovation(i, dst)
            weight = float(np.clip(
                rng.normal(cfg.weight_init_mean, cfg.weight_init_stdev),
                cfg.weight_min_value, cfg.weight_max_value,
            ))
            connections[innov] = ConnectionGene(innov, i, dst, weight)

    return Genome(
        genome_id   = genome_id,
        n_inputs    = n_inputs,
        n_outputs   = n_outputs,
        nodes       = nodes,
        connections = connections,
    )


###############################################################################
# Topological sort (module-private)
###############################################################################

def _topological_sort(
    node_ids: List[int],
    incoming: Dict[int, List[Tuple[int, float]]],
    *,
    allow_recurrent: bool,
) -> List[int]:
    """
    Return node ids in topological evaluation order.

    Uses Kahn's algorithm (BFS-based). If cycles are detected:
        allow_recurrent=False → raises ValueError
        allow_recurrent=True  → returns a best-effort order
                                 (cyclic nodes appended at end)
    """
    # build in-degree count from enabled connections
    in_degree: Dict[int, int] = {nid: 0 for nid in node_ids}
    for nid in node_ids:
        for src, _ in incoming.get(nid, []):
            if src in in_degree:
                in_degree[nid] += 1

    # start with nodes that have no incoming connections
    from collections import deque
    queue = deque(nid for nid in node_ids if in_degree[nid] == 0)
    order: List[int] = []

    while queue:
        nid = queue.popleft()
        order.append(nid)

        # for each node that nid feeds into
        for other in node_ids:
            for src, _ in incoming.get(other, []):
                if src == nid:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

    if len(order) < len(node_ids):
        # cycle detected
        cyclic = [nid for nid in node_ids if nid not in set(order)]
        if not allow_recurrent:
            raise ValueError(
                f"Cycle detected in genome topology. "
                f"Cyclic nodes: {cyclic}. "
                f"Use allow_recurrent=True to handle cycles."
            )
        # append cyclic nodes at the end in arbitrary order
        order.extend(cyclic)

    return order