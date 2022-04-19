"""Online data normalization."""

import jax.numpy as jnp

class Normalizer():
	"""Feture normalizer that accumulates statistics online."""

	def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8, name='Normalizer'):
		self.name = name
		self._max_accumulations = max_accumulations
		self._std_epsilon = std_epsilon
		self._acc_count = 0
		self._num_accumulations = 0
		self._acc_sum = jnp.zeros(size)
		self._acc_sum_squared = jnp.zeros(size)

	def normalize(self, batched_data, accumulate=True):
		"""Normalizes input data and accumulates statistics."""
		if accumulate and self._num_accumulations < self._max_accumulations:
			self._accumulate(batched_data)
				
		return (batched_data - self._mean()) / self._std_with_epsilon()

	def inverse(self, normalized_batch_data):
		"""Inverse transformation of the normalizer."""
		return normalized_batch_data * self._std_with_epsilon() + self._mean()

	def _accumulate(self, batched_data):
		"""Function to perform the accumulation of the batched_data statistics."""
		count = batched_data.shape[0]
		data_sum = jnp.sum(batched_data, axis=0)
		squared_data_sum = jnp.sum(batched_data**2, axis=0)

		self._acc_sum += data_sum
		self._acc_sum_squared += squared_data_sum
		self._acc_count += count
		self._num_accumulations += 1
		return

	def _mean(self):
		"""Calculates the mean."""
		safe_count = jnp.maximum(self._acc_count, 1)
		return self._acc_sum / safe_count

	def _std_with_epsilon(self):
		"""Calculates the std with std_epsilon."""
		safe_count = jnp.maximum(self._acc_count, 1)
		std = jnp.sqrt(self._acc_sum_squared / safe_count - self._mean()**2)
		return jnp.maximum(std, self._std_epsilon)
