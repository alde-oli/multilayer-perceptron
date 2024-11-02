import numpy as np


class Normalizer:
	def __init__(self, method="none"):
		methods = {
			"none": self.NotNormalized,
			"min-max": self.MinMax,
			"z-score": self.ZScore,
			"max-abs": self.MaxAbs,
			"robust": self.Robust
		}
		if method not in methods:
			raise ValueError("Invalid normalization method. Please choose one of the following: 'min-max', 'z-score', 'max-abs', 'robust'")
		self._method = methods[method]
		self.params = {}

	def fit(self, X):
		self.params = self._method.fit(X, self.params)

	def apply(self, X):
		return self._method.apply(X, self.params)
	

	class NotNormalized:
		@staticmethod
		def fit(X, params):
			pass

		@staticmethod
		def apply(X, params):
			return X


	class MinMax:
		# Scales the data to a specific range (often [0, 1]), useful for algorithms sensitive to feature scales.
		@staticmethod
		def fit(X, params):
			params["min_val"] = np.min(X, axis=0)
			params["max_val"] = np.max(X, axis=0)
			return params

		@staticmethod
		def apply(X, params):
			min_val, max_val = params["min_val"], params["max_val"]
			return (X - min_val) / (max_val - min_val)


	class ZScore:
		# Centers the data around the mean with a standard deviation of 1, ideal for algorithms assuming a normal distribution.
		@staticmethod
		def fit(X, params):
			params["mean"] = np.mean(X, axis=0)
			params["std"] = np.std(X, axis=0)
			return params

		@staticmethod
		def apply(X, params):
			mean, std = params["mean"], params["std"]
			return (X - mean) / std


	class MaxAbs:
		# Scales the data by dividing by the maximum absolute value, useful for data with both positive and negative values.
		@staticmethod
		def fit(X, params):
			params["max_abs"] = np.max(np.abs(X), axis=0)
			return params

		@staticmethod
		def apply(X, params):
			max_abs = params["max_abs"]
			return X / max_abs


	class Robust:
		# Uses the median and quartiles to scale the data, effective for data with outliers.
		@staticmethod
		def fit(X, params):
			params["median"] = np.median(X, axis=0)
			params["q1"] = np.percentile(X, 25, axis=0)
			params["q3"] = np.percentile(X, 75, axis=0)
			return params

		@staticmethod
		def apply(X, params):
			median, q1, q3 = params["median"], params["q1"], params["q3"]
			return (X - median) / (q3 - q1)
# Normalizer