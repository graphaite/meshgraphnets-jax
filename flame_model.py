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

		# Calculate the relative mesh positions
		senders, receivers = common.triangles_to_edges(inputs['cells'][0])
		relative_mesh_pos = (inputs['mesh_pos'][0, senders] - inputs['mesh_pos'][0, receivers])

		# edge_features is concatenation of relative mesh positions and norm of these
		edge_features = jnp.concatenate([
			relative_mesh_pos,
			jnp.linalg.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)
		
		# Create graph
		graph = jraph.GraphsTuple(
			nodes=node_features,
			edges=edge_features,
			senders=senders,
			receivers=receivers,
			n_node=n_node,
			n_edge=n_edge,
			globals=None
			)
		return graph
		
