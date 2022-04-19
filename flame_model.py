"""Model for flame dataset."""

import jax
import jax.numpy as jnp
import jraph
import common


class Model():
	"""Model for CFD simulation."""
	def __init__(self, name='Model'):
		self.name = name

	def _build_graph(self, inputs) -> jraph.GraphsTuple:
		"""Builds input graph."""
		n_edge = inputs['cells'].shape[1]
		n_node = inputs['node_type'].shape[1]
		traj_len = inputs['temperature'].shape[0]
		one_hot_size = common.NodeType.SIZE

		# Tile node_type for trajectory length
		node_type = jnp.tile(inputs['node_type'], [traj_len, 1, 1])

		# Convert node_type to one-hot encoding
		node_type = jax.nn.one_hot(node_type, one_hot_size).reshape(traj_len, n_node, one_hot_size)

		# node_features is concatenation of temperature and node type
		node_features = jnp.concatenate([inputs['temperature'], node_type], axis = -1) 

		'''
		TO DO
		graph = jraph.GraphsTuple(
			nodes=node_features,
			edges=edges,
			senders=senders,
			receivers=receivers,
			n_node=n_node,
			n_edge=n_edge,
			globals=global_context
			)
		return graph
		'''
		
