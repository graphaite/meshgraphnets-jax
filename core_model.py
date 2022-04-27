"""Core learned graph net model."""

import jax.numpy as jnp
import jax.ops as jop
from jax import random, vmap


class MLP():
	"""Multi-layer perceptron."""
	def __init__(self, cfg, seed=0, name='default_mlp'):
		self.n_layers = cfg.NUM_LAYERS
		self.size = cfg.LATENT_SIZE
		self.seed = seed
		self.name = name
		self.params = self._init_params()
		self.batched_predict = vmap(self.single_predict)

	def _random_layer_params(self, m, n, key, scale=1e-2):
		"""Randomly samples weights and biases."""
		w_key, b_key = random.split(key)
		return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

	def _init_params(self):
		"""Initialises the weights and biases."""
		keys = random.split(random.PRNGKey(self.seed), self.n_layers)
		return [self._random_layer_params(self.size, self.size, k) for k in keys]

	def _relu(self, x):
		"""ReLU activation function."""
		return jnp.maximum(0, x)

	def single_predict(self, x):
		"""Calculates output from single input."""
		# Per-example predictions
		for w, b in self.params[:-1]:
			outputs = jnp.dot(w, x) + b
			x = self._relu(outputs)
		
		# Final layer
		final_w, final_b = self.params[-1]
		outputs = jnp.dot(final_w, x) + final_b
		return outputs

	def predict(self, x):
		"""Calcalates output from batched inputs."""
		return self.batched_predict(x)


class GraphNetBlock():
	"""Multi-Edge Interaction Network with residual connections."""

	def __init__(self, cfg, name='GraphNetBlock'):
		self.name = name
		self._edge_mlp = MLP(cfg, seed=0, name='edge_mlp') # Manual seeding. Is there a better way of doing this? 
		self._node_mlp = MLP(cfg, seed=1, name='node_mlp') 

	def _update_edge_features(self, node_features, edge_features, senders, receivers):
		"""Aggregates node features, and applies edge function."""
		sender_features = node_features[senders]	
		receiver_features = node_features[receivers]	
		features = [sender_features, receiver_features, edge_features]
		return self._edge_mlp.predict(jnp.concatenate(features, axis=-1))

	def _update_node_features(self, node_features, edge_features, receivers):
		"""Aggregates edge features, and applies node function."""
		num_nodes = node_features.shape[0]
		features = [node_features]
		features.append(jop.segment_sum(edge_features, receivers, num_nodes))
		return self._node_mlp.predict(jnp.concatenate(features, axis=-1))

	def _build(self, graph):
		"""Applies GraphNetBlock and returns updated graph."""

		# Apply edge functions
		updated_features = self._update_edge_features(graph.nodes, graph.edges, graph.senders, graph.receivers)
		new_edge_features = graph.edges._replace(features=updated_features)
		#new_edge_sets.append(edge_set._replace(features=updated_features))

		# Apply node function
		new_node_features = self._update_node_features(graph.nodes, new_edge_features, graph.receivers)

		# Add residual connection
		new_node_features += graph.nodes
		new_edge_features += graph.edges

		updated_graph = jraph.GraphsTuple(
			nodes=new_node_features,
			edges=new_edge_features,
			senders=graph.senders,
			receivers=graph.receivers,
			n_node=graph.n_node,
			n_edge=graph.n_edge,
			globals=graph.globals
			)
		return updated_graph

