"""
Created on Tue Oct 20 08:08:43 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import json
import os
import pickle
import tempfile

import numpy as np

from neat.check_config import check_config

###############################################################################
# helpers
###############################################################################

CHECKPOINT_FORMAT = "neat-checkpoint"
GENOME_FORMAT = "neat-genome"
VERSION = 1


def write_atomically(path, data):
    """Write bytes to path through a temporary file in the same folder."""
    folder = os.path.dirname(os.path.abspath(path))
    handle, temp_path = tempfile.mkstemp(dir=folder, prefix=".tmp-neat-")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

###############################################################################
# checkpoint
###############################################################################

def save_checkpoint(path, state, rng, cfg):
    check_config(cfg)
    contents = {
        "format": CHECKPOINT_FORMAT,
        "version": VERSION,
        "state": state,
        "rng_state": rng.bit_generator.state,
        "cfg": cfg,
    }
    write_atomically(path, pickle.dumps(contents))


def load_checkpoint(path):
    with open(path, "rb") as file:
        contents = pickle.load(file)
    if not isinstance(contents, dict) or contents.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a NEAT checkpoint")
    if contents.get("version") != VERSION:
        raise ValueError(f"checkpoint version {contents.get('version')!r} is not supported "
                         f"(this code reads version {VERSION})")
    rng_state = contents["rng_state"]
    bit_generator = getattr(np.random, rng_state["bit_generator"])()
    bit_generator.state = rng_state
    return contents["state"], np.random.Generator(bit_generator), contents["cfg"]

###############################################################################
# genome
###############################################################################

def save_genome(path, genome):
    as_text = {
        "n_inputs": genome["n_inputs"],
        "n_outputs": genome["n_outputs"],
        "nodes": {str(node_id): node for node_id, node in genome["nodes"].items()},
        "connections": {str(key): connection for key, connection in genome["connections"].items()},
    }
    contents = {"format": GENOME_FORMAT, "version": VERSION, "genome": as_text}
    text = json.dumps(contents, indent=2, allow_nan=False)
    write_atomically(path, text.encode("utf-8"))


def load_genome(path):
    with open(path, "r", encoding="utf-8") as file:
        contents = json.load(file)
    if not isinstance(contents, dict) or contents.get("format") != GENOME_FORMAT:
        raise ValueError(f"{path} is not a NEAT genome file")
    if contents.get("version") != VERSION:
        raise ValueError(f"genome file version {contents.get('version')!r} is not supported "
                         f"(this code reads version {VERSION})")
    as_text = contents["genome"]
    for key in ("n_inputs", "n_outputs", "nodes", "connections"):
        if key not in as_text:
            raise ValueError(f"genome file is missing {key!r}")
    return {
        "n_inputs": as_text["n_inputs"],
        "n_outputs": as_text["n_outputs"],
        "nodes": {int(node_id): node for node_id, node in as_text["nodes"].items()},
        "connections": {int(key): connection for key, connection in as_text["connections"].items()},
    }