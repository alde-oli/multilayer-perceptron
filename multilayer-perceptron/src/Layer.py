import numpy as np
import pandas as pd

from Activation import Activation
from Initializer import Initializer


class Layer:
	def _forward(self, input):
		raise NotImplementedError("this must be implemented")
	
	def _backward(self, grads, learning_rate):
		raise NotImplementedError("this must be implemented")
# Layer


class Input(Layer):
	def __init__(self, in_shape):
		super().__init__()
		self.in_shape = in_shape
	
	def _forward(self, input):
		if not isinstance(input, np.ndarray):
			raise TypeError("input must be a numpy array")
		if input.shape != (self.in_shape,):
			raise ValueError(f"input shape must be {self.in_shape}")
		return input
	
	def _backward(self, grads):
		# to implement
		pass
# Input


class Dense(Layer):
	def __init__(self, in_shape, out_shape, activation="linear", alpha=None, w_init="zero", b_init="zero"):
		super().__init__()
		self.in_shape = in_shape
		self.out_shape = out_shape
		self.activation = Activation(activation, alpha)
		self.weights = Initializer.apply(
			w_init,
			in_shape, out_shape
		)
		self.biases = Initializer.apply(
			b_init,
			1, out_shape
		)
	
	def _forward(self, input):
		self.input = input
		return self.activation(np.dot(input, self.weights) + self.biases)
	
	def _backward(self):
		# to implement
		pass
# Dense



if __name__ == "__main__":
	tab = Initializer("he", 3, 2)
	print(tab)
# main