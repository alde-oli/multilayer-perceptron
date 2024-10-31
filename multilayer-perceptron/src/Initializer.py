import numpy as np


class Initializer:
	@staticmethod
	def apply(method="zero", x=1, y=1):
		if method == "zero":
			return Initializer.zero(x, y)
		elif method == "random":
			return Initializer.random(x, y)
		elif method == "xavier":
			return Initializer.xavier(x, y)
		elif method == "he":
			return Initializer.he(x, y)
		elif method == "lecun":
			return Initializer.lecun(x, y)
		elif method == "orthogonal":
			return Initializer.orthogonal(x, y)
		else:
			raise ValueError(f"Unknown initialization method: {method}")
	
	@staticmethod
	def zero(x, y):
		# Not recommended as it can lead to symmetry problems where all neurons learn the same features.
		return np.zeros((x, y))
	# zero

	@staticmethod
	def random(x, y):
		# Useful for breaking symmetry, but can cause gradient issues if values are too large or too small.
		return np.random.randn(x, y) * 0.01
	# random

	@staticmethod
	def xavier(x, y):
		# Recommended for networks with sigmoid or tanh activation functions.
		limit = np.sqrt(6 / (x + y))
		return np.random.uniform(-limit, limit, (x, y))
	# xavier

	@staticmethod
	def he(x, y):
		# Recommended for networks with ReLU or its variants as activation functions.
		stddev = np.sqrt(2 / x)
		return np.random.randn(x, y) * stddev
	# he

	@staticmethod
	def lecun(x, y):
		# Recommended for networks with tanh activation functions.
		stddev = np.sqrt(1 / x)
		return np.random.randn(x, y) * stddev
	# lecun

	@staticmethod
	def orthogonal(x, y):
		# Useful for deep networks to help maintain gradient stability.
		a = np.random.randn(x, y)
		u, _, v = np.linalg.svd(a, full_matrices=False)
		return u if x > y else v
	# orthogonal
# Initializer


if __name__ == "__main__":
	x = 3
	y = 5
	print(f"""zero:
{Initializer.zero(x, y)}
random:
{Initializer.random(x, y)}
xavier:
{Initializer.xavier(x, y)}
he:
{Initializer.he(x, y)}
lecun:
{Initializer.lecun(x, y)}
orthogonal:
{Initializer.orthogonal(x, y)}""")
# main
