"""Utility functions for reading the datasets."""

import jax.numpy as jnp
import numpy as onp

def load_dataset(path="datasets/flame_minimal/"):
	"""Load dataset."""
	data = onp.load(path + "flame_minimal.npz")
	cells = data['cells']
	mesh_pos = data['mesh_pos']
	node_type = data['node_type']
	temperature = data['temperature']

	ds = {
		'cells' : jnp.array(cells),
		'mesh_pos' : jnp.array(mesh_pos),
		'node_type' : jnp.array(node_type),
		'temperature' : jnp.array(temperature)
	     }

	return ds
