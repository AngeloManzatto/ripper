"""
Created on Sat Jun  6 18:54:04 2026

@author: Angelo Antonio Manzatto
--------------
Gene types for the NEAT algorithm.

NEAT has exactly two gene types:

    NodeGene       — represents a neuron (id, type, bias, activation)
    ConnectionGene — represents a synapse (innovation, src, dst, weight, enabled)

Design
------
- Both are frozen dataclasses: immutable value objects.
- Mutation never modifies a gene in place — it returns a new gene.
- This makes copying, crossover, and hashing trivial and thread-safe.
- Serialization (to_dict / from_dict) is supported from day one.
- NodeType is an enum: the three node roles are a closed, exhaustive set.

Innovation numbers
------------------
Innovation numbers live on ConnectionGene only. They are the historical
marker that aligns genomes during crossover. Node ids are plain sequential
integers — no innovation tracking needed for nodes.

The enabled flag
----------------
When NEAT splits a connection by adding a node, the original connection is
disabled (not deleted). Disabled genes stay in the genome as part of the
historical record and participate in crossover. This is required by the
NEAT paper and must not be optimised away.
"""

###############################################################################
# libraries
###############################################################################

import dataclasses
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict

###############################################################################
# NodeType
###############################################################################

@unique
class NodeType(Enum):
    """
    Role of a node in the network topology.

    INPUT  — receives external input; no bias, no activation applied.
    HIDDEN — intermediate computation node; has bias and activation.
    OUTPUT — produces network output; has bias and activation.

    The distinction between INPUT and the others matters during forward
    evaluation: input node values are set directly from the feature vector
    and are never computed from incoming connections.
    """
    INPUT  = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


###############################################################################
# NodeGene
###############################################################################

@dataclass(frozen=True)
class NodeGene:
    """
    Immutable gene representing a single neuron.

    Attributes
    ----------
    node_id:
        Unique integer identifier. Assigned sequentially by the genome.
        Input nodes use ids 0..n_inputs-1.
        Output nodes use ids n_inputs..n_inputs+n_outputs-1.
        Hidden nodes use ids above that range.
    node_type:
        Role of this node (INPUT, HIDDEN, OUTPUT).
    bias:
        Additive bias applied before the activation function.
        Conventionally 0.0 for input nodes (bias has no effect since
        input values are set directly, but we keep the field for
        uniformity and serialization consistency).
    activation:
        Name of the activation function as registered in
        neat.activations.DEFAULT_REGISTRY.
        Input nodes use "identity" — their value passes through unchanged.
    """

    node_id:    int
    node_type:  NodeType
    bias:       float
    activation: str

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_input(self) -> bool:
        return self.node_type is NodeType.INPUT

    @property
    def is_hidden(self) -> bool:
        return self.node_type is NodeType.HIDDEN

    @property
    def is_output(self) -> bool:
        return self.node_type is NodeType.OUTPUT

    # ── Mutation (returns new gene) ────────────────────────────────────────────

    def with_bias(self, bias: float) -> "NodeGene":
        """Return a copy with a new bias value."""
        return dataclasses.replace(self, bias=bias)

    def with_activation(self, activation: str) -> "NodeGene":
        """Return a copy with a new activation function name."""
        return dataclasses.replace(self, activation=activation)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "node_id":    self.node_id,
            "node_type":  self.node_type.value,
            "bias":       self.bias,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NodeGene":
        """Deserialize from a dictionary produced by to_dict()."""
        return cls(
            node_id    = int(d["node_id"]),
            node_type  = NodeType(d["node_type"]),
            bias       = float(d["bias"]),
            activation = str(d["activation"]),
        )

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"NodeGene("
            f"id={self.node_id}, "
            f"type={self.node_type.value}, "
            f"bias={self.bias:.3f}, "
            f"act={self.activation})"
        )


###############################################################################
# ConnectionGene
###############################################################################

@dataclass(frozen=True)
class ConnectionGene:
    """
    Immutable gene representing a directed weighted connection.

    Attributes
    ----------
    innovation:
        Global innovation number assigned by the InnovationRegistry.
        Uniquely identifies the historical origin of this connection —
        specifically the (src_node_id, dst_node_id) structural event.
        Used to align genes during crossover.
    src_node_id:
        Id of the source (pre-synaptic) node.
    dst_node_id:
        Id of the destination (post-synaptic) node.
    weight:
        Multiplicative connection weight.
    enabled:
        Whether this connection participates in the forward pass.
        A connection is disabled when a new node is added by splitting it.
        Disabled connections are retained in the genome for crossover
        alignment — they must not be removed.

    Notes
    -----
    Recurrent connections (src == dst, or cycles in the topology) are
    valid in NEAT. The forward pass handles them via topological ordering
    with cycle detection.
    """

    innovation:  int
    src_node_id: int
    dst_node_id: int
    weight:      float
    enabled:     bool = True

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_recurrent(self) -> bool:
        """True when the connection forms a self-loop."""
        return self.src_node_id == self.dst_node_id

    # ── Mutation (returns new gene) ────────────────────────────────────────────

    def with_weight(self, weight: float) -> "ConnectionGene":
        """Return a copy with a new weight."""
        return dataclasses.replace(self, weight=weight)

    def enabled_copy(self) -> "ConnectionGene":
        """Return a copy with enabled=True."""
        return dataclasses.replace(self, enabled=True)

    def disabled_copy(self) -> "ConnectionGene":
        """Return a copy with enabled=False."""
        return dataclasses.replace(self, enabled=False)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "innovation":  self.innovation,
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "weight":      self.weight,
            "enabled":     self.enabled,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConnectionGene":
        """Deserialize from a dictionary produced by to_dict()."""
        return cls(
            innovation  = int(d["innovation"]),
            src_node_id = int(d["src_node_id"]),
            dst_node_id = int(d["dst_node_id"]),
            weight      = float(d["weight"]),
            enabled     = bool(d["enabled"]),
        )

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "on" if self.enabled else "off"
        return (
            f"ConnectionGene("
            f"innov={self.innovation}, "
            f"{self.src_node_id}->{self.dst_node_id}, "
            f"w={self.weight:.3f}, "
            f"{status})"
        )