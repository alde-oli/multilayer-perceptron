from src.activation import Linear, Sigmoid, ReLU, LeakyReLU, ELU, Tanh, Softmax
from src.initialization import zero, random, xavier, he 
from src.layers import Layer, Input, Normalization, Dense
from src.mlp import MLP

import numpy as np


model = MLP([
	Input(2),
	Normalization("standard"),
	Dense(2, 3, ReLU(), random),
	Dense(3, 2, Softmax(), random)
])

additions = np.array([[0, 0, 0], [2, 2, 4], [0, 2, 2], [2, 0, 2], [1, 1, 2], [1, 0, 1], [0, 1, 1], [2, 1, 3], [1, 2, 3], [6, 6, 12]])
X = additions[:, :-1]
y = additions[:, -1]

model.train(X, y, 1000, 0.01)
print(model.predict(X))