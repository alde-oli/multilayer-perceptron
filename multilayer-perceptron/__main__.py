from src.Activation import Activation
from src.Initializer import initialize
from src.Layer import Layer, Input, Dense
from src.MLP import MLP

import numpy as np


model = MLP(
	layers=[
		Input(2),
		Dense(2, 3, "ReLU", "he"),
		Dense(3, 2, "softmax", "orthogonal")
	],
	normalization=""
)

additions = np.array([[0, 0, 0], [2, 2, 4], [0, 2, 2], [2, 0, 2], [1, 1, 2], [1, 0, 1], [0, 1, 1], [2, 1, 3], [1, 2, 3], [6, 6, 12]])
X = additions[:, :-1]
y = additions[:, -1]

model.train(X, y, 1000, 0.01)
print(model.predict(X))