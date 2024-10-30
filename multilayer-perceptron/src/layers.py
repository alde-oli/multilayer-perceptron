import numpy as np
import pandas as pd

from src.activation import Linear, Sigmoid, ReLU, LeakyReLU, ELU, Tanh, Softmax
from src.initialization import zero, random, xavier, he 


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
		# 	
		pass
# Input


class Normalization(Layer):
	def __init__(self, method="standard"):
		super().__init__()
		self.method = method
		self.mean = None
		self.std = None
		self.min = None
		self.max = None
		self.eps = np.finfo(float).eps

	def _forward(self, input):
		if self.method == "standard":
			self.mean = np.mean(input, axis=0)
			self.std = np.std(input, axis=0) + self.eps
			return (input - self.mean) / self.std
		elif self.method == "minmax":
			self.min = np.min(input, axis=0)
			self.max = np.max(input, axis=0)
			return (input - self.min) / (self.max - self.min + self.eps)
		elif self.method == "mean":
			self.mean = np.mean(input, axis=0)
			return input - self.mean
		else:
			raise ValueError("Invalid normalization method specified")

	def _backward(self):
		pass
# Normalization


class Dense(Layer):
	def __init__(self, in_shape, out_shape, activation=Linear, w_init=zero, b_init=zero):
		super().__init__()
		self.in_shape = in_shape
		self.out_shape = out_shape
		self.activation = activation
		self.weights = w_init(in_shape, out_shape)
		self.biases = b_init(1, out_shape)
	
	def _forward(self, input):
		self.input = input
		return self.activation(np.dot(input, self.weights) + self.biases)
	
	def _backward(self):
		pass
# Dense