import numpy as np

class Optimizer:
	def __init__(self, method="Adam", learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.m = None
		self.v = None
		self.t = 0
		methods = {
			"SGD": self._sgd,
			"Momentum": self._momentum,
			"NAG": self._nag,
			"Adagrad": self._adagrad,
			"RMSprop": self._rmsprop,
			"Adam": self._adam,
			"Adadelta": self._adadelta,
			"Adamax": self._adamax,
			"Nadam": self._nadam
		}
		if method not in methods:
			raise ValueError("Invalid optimisation method. Please choose one of the following: " + ", ".join(methods.keys()))
		self._method = methods[method]

	def apply(self, weights, gradients):
		return self._method(weights, gradients)

	def _sgd(self, weights, gradients):
		return weights - self.learning_rate * gradients

	def _momentum(self, weights, gradients):
		if self.m is None:
			self.m = np.zeros_like(weights)
		self.m = self.beta1 * self.m + self.learning_rate * gradients
		return weights - self.m

	def _nag(self, weights, gradients):
		if self.m is None:
			self.m = np.zeros_like(weights)
		prev_m = self.m
		self.m = self.beta1 * self.m + self.learning_rate * gradients
		return weights - (self.beta1 * prev_m + (1 + self.beta1) * self.m)

	def _adagrad(self, weights, gradients):
		if self.v is None:
			self.v = np.zeros_like(weights)
		self.v += gradients ** 2
		return weights - (self.learning_rate / (np.sqrt(self.v) + self.epsilon)) * gradients

	def _rmsprop(self, weights, gradients):
		if self.v is None:
			self.v = np.zeros_like(weights)
		self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2
		return weights - (self.learning_rate / (np.sqrt(self.v) + self.epsilon)) * gradients

	def _adam(self, weights, gradients):
		if self.m is None:
			self.m = np.zeros_like(weights)
		if self.v is None:
			self.v = np.zeros_like(weights)
		self.t += 1
		self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
		self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2
		m_hat = self.m / (1 - self.beta1 ** self.t)
		v_hat = self.v / (1 - self.beta2 ** self.t)
		return weights - (self.learning_rate / (np.sqrt(v_hat) + self.epsilon)) * m_hat

	def _adadelta(self, weights, gradients):
		if self.v is None:
			self.v = np.zeros_like(weights)
		if self.m is None:
			self.m = np.zeros_like(weights)
		self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2
		update = (np.sqrt(self.m + self.epsilon) / np.sqrt(self.v + self.epsilon)) * gradients
		self.m = self.beta2 * self.m + (1 - self.beta2) * update ** 2
		return weights - update

	def _adamax(self, weights, gradients):
		if self.m is None:
			self.m = np.zeros_like(weights)
		if self.v is None:
			self.v = np.zeros_like(weights)
		self.t += 1
		self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
		self.v = np.maximum(self.beta2 * self.v, np.abs(gradients))
		return weights - (self.learning_rate / (self.v + self.epsilon)) * self.m

	def _nadam(self, weights, gradients):
		if self.m is None:
			self.m = np.zeros_like(weights)
		if self.v is None:
			self.v = np.zeros_like(weights)
		self.t += 1
		self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
		m_hat = self.m / (1 - self.beta1 ** self.t)
		self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2
		v_hat = self.v / (1 - self.beta2 ** self.t)
		return weights - (self.learning_rate / (np.sqrt(v_hat) + self.epsilon)) * (self.beta1 * m_hat + (1 - self.beta1) * gradients)
# Optimizer