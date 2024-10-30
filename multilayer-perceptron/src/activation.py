import numpy as np


class Linear:
	# simply returns the input
	def __call__(self, x):
		return x
	
	def derivative(self, x):
		return np.ones_like(x)
# Linear


class Sigmoid:
	# squeeze the input between 0 and 1
	def __call__(self, x):
		return 1 / (1 + np.exp(-x))
	
	def derivative(self, x):
		return x * (1 - x)
# Sigmoid


class ReLU:
	# returns 0 if negative, else returns the input
	def __call__(self, x):
		return np.maximum(0, x)
	
	def derivative(self, x):
		return np.where(x > 0, 1, 0)
# ReLU


class LeakyReLU:
	# returns alpha * x if negative to avoid the dying ReLU problem, else returns the input
	def __init__(self, alpha=0.01):
		self.alpha = alpha

	def __call__(self, x):
		return np.where(x < 0, self.alpha * x, x)
	
	def derivative(self, x):
		return np.where(x > 0, 1, self.alpha)
# LeakyReLU


class ELU:
	# returns alpha * (exp(x) - 1) if negative to avoid the dying ReLU problem, else returns the input
	def __init__(self, alpha=1.0):
		self.alpha = alpha

	def __call__(self, x):
		return np.where(x < 0, self.alpha * (np.exp(x) - 1), x)
	
	def derivative(self, x):
		return np.where(x > 0, 1, self.alpha * np.exp(x))
# ELU


class Tanh:
	# squeeze the input between -1 and 1
	def __call__(self, x):
		return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
	
	def derivative(self, x):
		return 1 - x ** 2
# Tanh


class Softmax:
	# returns a probability distribution which sums to 1. ex: [3, 2, 4] -> [0.24472847, 0.09003057, 0.66524096]
	def __call__(self, x):
		exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
		return exps / np.sum(exps, axis=-1, keepdims=True)
	
	def derivative(self, x):
		return x * (1 - x)
# Softmax



if __name__ == "__main__":
	x = np.array([-3, -1, 0, 1, 3])
	print(f"""initial array: {x}
   linear: {Linear()(x)}
  sigmoid: {Sigmoid()(x)}
     ReLU: {ReLU()(x)}
LeakyReLU: {LeakyReLU()(x)}
      ELU: {ELU()(x)}
     Tanh: {Tanh()(x)}
  softmax: {Softmax()(x)}""")
# main
