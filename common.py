"""Commonly used data structures and functions."""

import enum
import jax.numpy as jnp


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 7


def triangles_to_edges(faces):
    """Computes mesh edges from triangles."""
    edges = jnp.concatenate([faces[:, 0:2],
                             faces[:, 1:3],
                             jnp.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)

    receivers = jnp.min(edges, axis=1)
    senders = jnp.max(edges, axis=1)
    packed_edges = jnp.stack([senders, receivers], axis=1)

    # Remove duplicates and unpack
    unique_edges = jnp.unique(packed_edges, axis=0)
    senders, receivers = unique_edges[:, 0], unique_edges[:, 1]

    # two-way connectivity
    return(jnp.concatenate([senders, receivers], axis=0),
           jnp.concatenate([receivers, senders], axis=0))
