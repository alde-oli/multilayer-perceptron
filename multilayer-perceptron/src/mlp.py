from src.layers import Layer

import numpy as np


class MLP:
	def __init__(self, layers):
		self.layers = layers

	def forward(self, input):
		for layer in self.layers:
			input = layer._forward(input)
		return input

	def backward(self, grads, learning_rate):
		# to implement
		pass
	
	def train(self, X, y, epochs, learning_rate):
		for epoch in range(epochs):
			for i in range(len(X)):
				prediction = self.forward(X[i])
				# next steps to implement
				pass
	
	def predict(self, X):
		return [self.forward(x) for x in X]
	
	def evaluate(self, X, y):
		predictions = self.predict(X)
		return sum([np.argmax(predictions[i]) == np.argmax(y[i]) for i in range(len(y))]) / len(y)