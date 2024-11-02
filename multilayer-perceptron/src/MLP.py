from Layer import Layer, Input, Dense
from Normalizer import Normalizer
from Loss import Loss
from Optimizer import Optimizer

import numpy as np


class MLP:
	def __init__(self, layers, normalization="none", loss="MSE"):
		self.layers = layers
		self.normalizer = Normalizer(normalization)
		self.loss = Loss(loss)

	def forward(self, input):
		for layer in self.layers:
			input = layer._forward(input)
		return input

	def backward(self, grads, learning_rate):
		pass

	def train(self, X, y, epochs, loss_function="MSE", optimizer="SGD", batch_size="full", learning_rate=0.01, validation=0.2):
		self.normalizer.fit(X)
		X_normalized = self.normalizer.apply(X)
		optimizer = Optimizer(method=optimizer, learning_rate=learning_rate)

	
	def predict(self, X):
		X_normalized = self.normalizer.apply(X)
		return [self.forward(x, train=False) for x in X_normalized]
	
	def evaluate(self, X, y):
		predictions = self.predict(X)
		return sum([np.argmax(predictions[i]) == np.argmax(y[i]) for i in range(len(y))]) / len(y)
	

	'''
	La méthode backward est généralement responsable du calcul des gradients des poids et des biais, mais la mise à jour des poids est souvent déléguée à un optimiseur pour plusieurs raisons :

Séparation des responsabilités : La méthode backward se concentre sur le calcul des gradients, tandis que l'optimiseur gère la mise à jour des poids en fonction de ces gradients. Cela permet de garder le code plus modulaire et plus facile à maintenir.

Flexibilité : En utilisant un optimiseur, vous pouvez facilement changer la méthode de mise à jour des poids (par exemple, passer de la descente de gradient stochastique à Adam) sans modifier la logique de calcul des gradients.

Complexité des optimisateurs : Certains optimisateurs, comme Adam ou RMSprop, nécessitent des calculs supplémentaires (comme le suivi des moments) qui ne sont pas directement liés au calcul des gradients. En séparant ces responsabilités, vous pouvez implémenter des optimisateurs plus complexes sans alourdir la méthode backward.

Plan
Mettre à jour la méthode backward pour qu'elle calcule uniquement les gradients.
Utiliser l'optimiseur pour mettre à jour les poids après avoir calculé les gradients.
Code
Mettre à jour MLP.py
Mettre à jour Layer.py
Notes
La méthode backward calcule maintenant uniquement les gradients.
Les poids sont mis à jour dans la méthode train en utilisant l'optimiseur.
Assurez-vous que les classes Normalizer, Loss, Activation, et Initializer ont les méthodes nécessaires.

from Optimizer import Optimizer
import numpy as np

class MLP:
    def __init__(self, layers, normalization="none", loss="MSE"):
        self.layers = layers
        self.normalizer = Normalizer(normalization)
        self.loss = Loss(loss)

    def forward(self, input):
        for layer in self.layers:
            input = layer._forward(input)
        return input

    def backward(self, grads):
        for layer in reversed(self.layers):
            grads = layer._backward(grads)

    def train(self, X, y, epochs, loss_function="MSE", optimizer="SGD", batch_size="full", learning_rate=0.01, validation=0.2):
        self.normalizer.fit(X)
        X_normalized = self.normalizer.apply(X)
        optimizer = Optimizer(method=optimizer, learning_rate=learning_rate)
        loss_fn = Loss(loss_function)

        for epoch in range(epochs):
            for i in range(0, len(X_normalized), batch_size):
                X_batch = X_normalized[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Forward pass
                output = self.forward(X_batch)

                # Compute loss and gradients
                loss = loss_fn.apply(output, y_batch)
                grads = loss_fn.derivative(output, y_batch)

                # Backward pass
                self.backward(grads)

                # Update weights using optimizer
                for layer in self.layers:
                    if isinstance(layer, Dense):
                        layer.weights = optimizer.apply(layer.weights, layer.weight_grad)
                        layer.biases = optimizer.apply(layer.biases, layer.bias_grad)

    def predict(self, X):
        X_normalized = self.normalizer.apply(X)
        return [self.forward(x, train=False) for x in X_normalized]
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        return sum([np.argmax(predictions[i]) == np.argmax(y[i]) for i in range(len(y))]) / len(y)

	class Dense(Layer):
    def __init__(self, in_shape, out_shape, activation="linear", alpha=None, w_init="zero", b_init="zero"):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.activation = Activation(activation, alpha)
        self.weights = Initializer.apply(w_init, in_shape, out_shape)
        self.biases = Initializer.apply(b_init, 1, out_shape)
    
    def _forward(self, input, train=True):
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases
        return self.activation(self.z)
    
    def _backward(self, grads):
        activation_grad = self.activation.derivative(self.z) * grads
        self.weight_grad = np.dot(self.input.T, activation_grad)
        self.bias_grad = np.sum(activation_grad, axis=0, keepdims=True)
        input_grad = np.dot(activation_grad, self.weights.T)
        return input_grad
	'''