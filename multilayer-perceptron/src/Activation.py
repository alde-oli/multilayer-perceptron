import numpy as np


class Activation:
	def __init__(self, method="linear", alpha=None):
		methods = {
			"linear": self.Linear,
			"sigmoid": self.Sigmoid,
			"ReLU": self.ReLU,
			"leaky ReLU": self.LeakyReLU,
			"ELU": self.ELU,
			"tanh": self.Tanh,
			"softmax": self.Softmax
		}
		if method not in methods:
			raise ValueError("Invalid activation method. Please choose one of the following: 'linear', 'sigmoid', 'ReLU', 'leaky ReLU', 'ELU', 'tanh', 'softmax'")
		self._method = methods[method]
		self.alpha = alpha
		if alpha == None:
			if method == "leaky ReLU":
				self.alpha = 0.01
			elif method == "ELU":
				self.alpha = 1.0
	
	def __call__(self, x):
		return self._method.apply(x, self.alpha)
	
	def derivative(self, x):
		return self._method.derivative(x, self.alpha)


	class Linear:
		# simply returns the input
		@staticmethod
		def apply(x, alpha=None):
			return x
		
		@staticmethod
		def derivative(x, alpha=None):
			return np.ones_like(x)
	# Linear


	class Sigmoid:
		# squeeze the input between 0 and 1
		@staticmethod
		def apply(x, alpha=None):
			return 1 / (1 + np.exp(-x))
		
		@staticmethod
		def derivative(x, alpha=None):
			return x * (1 - x)
	# Sigmoid


	class ReLU:
		# returns 0 if negative, else returns the input
		@staticmethod
		def apply(x, alpha=None):
			return np.maximum(0, x)
		
		@staticmethod
		def derivative(x, alpha=None):
			return np.where(x > 0, 1, 0)
	# ReLU


	class LeakyReLU:
		# returns alpha * x if negative to avoid the dying ReLU problem, else returns the input
		@staticmethod
		def apply(x, alpha=None):
			return np.where(x < 0, alpha * x, x)
		
		@staticmethod
		def derivative(x, alpha=None):
			return np.where(x > 0, 1, alpha)
	# LeakyReLU


	class ELU:
		# returns alpha * (exp(x) - 1) if negative to avoid the dying ReLU problem, else returns the input
		@staticmethod
		def apply(x, alpha=None):
			return np.where(x < 0, alpha * (np.exp(x) - 1), x)
		
		@staticmethod
		def derivative(x, alpha=None):
			return np.where(x > 0, 1, alpha * np.exp(x))
	# ELU


	class Tanh:
		# squeeze the input between -1 and 1
		@staticmethod
		def apply(x, alpha=None):
			return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
		
		@staticmethod
		def derivative(x, alpha=None):
			return 1 - x ** 2
	# Tanh


	class Softmax:
		# returns a probability distribution which sums to 1. ex: [3, 2, 4] -> [0.24472847, 0.09003057, 0.66524096]
		@staticmethod
		def apply(x, alpha=None):
			exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
			return exps / np.sum(exps, axis=-1, keepdims=True)
		
		@staticmethod
		def derivative(x, alpha=None):
			return x * (1 - x)
# Softmax



if __name__ == "__main__":
	x = np.array([-3, -1, 0, 1, 3])

	print("\tValid test:")
	print(f"""initial array: {x}
after activation:
   linear: {Activation("linear")(x)}
  sigmoid: {Activation("sigmoid")(x)}
     ReLU: {Activation("ReLU")(x)}
LeakyReLU: {Activation("leaky ReLU")(x)}
      ELU: {Activation("ELU")(x)}
     Tanh: {Activation("tanh")(x)}
  softmax: {Activation("softmax")(x)}\n""")

	print(f"""derivatives:
   linear: {Activation("linear").derivative(x)}
  sigmoid: {Activation("sigmoid").derivative(x)}
     ReLU: {Activation("ReLU").derivative(x)}
LeakyReLU: {Activation("leaky ReLU").derivative(x)}
      ELU: {Activation("ELU").derivative(x)}
     Tanh: {Activation("tanh").derivative(x)}
  softmax: {Activation("softmax").derivative(x)}\n""")
	print("\tInvalid test:")
	try:
		Activation("pouet")
	except Exception as e:
		print(e)
# main
