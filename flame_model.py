"""Model for flame dataset."""

import jax
import jax.numpy as jnp
import jraph
import common
import normalization


class Model():
	"""Model for CFD simulation."""
	def __init__(self, name='Model'):
		self.name = name
		self._output_normalizer = normalization.Normalizer(size=1, name='output_normalizer') # Output is scalar (temperature)
		self._node_normalizer = normalization.Normalizer(size=1+common.NodeType.SIZE, name='node_normalizer') # Node comprises temperature and one-hot of node_type
		self._edge_normalizer = normalization.Normalizer(size=3, name='edge_normalizer') # 2D coord + length

	def _build_graph(self, inputs, is_training) -> jraph.GraphsTuple:
		"""Builds input graph."""
		n_edge = inputs['cells'].shape[0]
		n_node = inputs['node_type'].shape[0]
		one_hot_size = common.NodeType.SIZE

		# Convert node_type to one-hot encoding
		node_type = jax.nn.one_hot(inputs['node_type'], one_hot_size).reshape(-1, one_hot_size)

		# node_features is concatenation of temperature and node type
		node_features = jnp.concatenate([inputs['temperature'], node_type], axis = -1)
		node_features_norm = self._node_normalizer.normalize(node_features, is_training)

		# Calculate the relative mesh positions
		senders, receivers = common.triangles_to_edges(inputs['cells'])
		relative_mesh_pos = inputs['mesh_pos'][senders] - inputs['mesh_pos'][receivers]

		# edge_features is concatenation of relative mesh positions and norm of these
		edge_features = jnp.concatenate([
			relative_mesh_pos,
			jnp.linalg.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)
		edge_features_norm = self._edge_normalizer.normalize(edge_features, is_training)
		
		# Create graph
		graph = jraph.GraphsTuple(
			nodes=node_features_norm,
			edges=edge_features_norm,
			senders=senders,
			receivers=receivers,
			n_node=n_node,
			n_edge=n_edge,
			globals=None
			)
		return graph
