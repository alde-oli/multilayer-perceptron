import numpy as np


class Loss:
	def __init__(self, method="MSE", delta=1):
		methods = {
			"MSE": self.MSE,
			"MAE": self.MAE,
			"cross entropy": self.crossEntropy,
			"binary cross entropy": self.binaryCrossEntropy,
			"hinge": self.hinge,
			"huber": self.huber,
			"KL divergence": self.KLDivergence
		}
		if method not in methods:
			raise ValueError("Invalid Loss method. Please choose one of the following: 'MSE', 'MAE', 'cross entropy', 'binary cross entropy', 'hinge', 'huber', 'KL divergence'")
		self._method = methods[method]
		self.delta = delta
	
	def apply(self, y_pred, y_true):
		return self._method(y_pred, y_true, self.delta)

	def MSE(self, y_true, y_pred):
		return np.mean((y_true - y_pred) ** 2)

	def MAE(self, y_true, y_pred):
		return np.mean(np.abs(y_true - y_pred))

	def CrossEntropy(self, y_true, y_pred):
		y_pred = np.clip(y_pred, 1e-10, 1.0)
		return -np.sum(y_true * np.log(y_pred))

	def BinaryCrossEntropy(self, y_true, y_pred):
		y_pred = np.clip(y_pred, 1e-10, 1.0)
		return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

	def Hinge(self, y_true, y_pred):
		return np.mean(np.maximum(0, 1 - y_true * y_pred))

	def Huber(self, y_true, y_pred, delta=1.0):
		error = y_true - y_pred
		is_small_error = np.abs(error) <= delta
		squared_loss = 0.5 * error ** 2
		linear_loss = delta * (np.abs(error) - 0.5 * delta)
		return np.mean(np.where(is_small_error, squared_loss, linear_loss))

	def KLDivergence(self, y_true, y_pred):
		if not (np.all(y_true >= 0) and np.all(y_pred >= 0)):
			raise ValueError("y_true and y_pred values must be positive.")
		if not (np.isclose(np.sum(y_true), 1) and np.isclose(np.sum(y_pred), 1)):
			raise ValueError("y_true and y_pred must be distribution probabilities (sum equal to 1).")
		y_pred = np.clip(y_pred, 1e-10, 1.0)
		return np.sum(y_true * np.log(y_true / y_pred))
# Loss