import numpy as np


def zero(x=1, y=1):
	return np.zeros((x, y))
# zero


def random(x=1, y=1):
	return np.random.randn(x, y) * 0.01
# random


def xavier(x, y):
	limit = np.sqrt(6 / (x + y))
	return np.random.uniform(-limit, limit, (x, y))
# xavier


def he(x, y):
	stddev = np.sqrt(2 / x)
	return np.random.randn(x, y) * stddev
# he


if __name__ == "__main__":
	x = 3
	y = 5
	print(f"""  zero:
{zero(x, y)}
random:
{random(x, y)}
xavier:
{xavier(x, y)}
	he:
{he(x, y)}""")
# main
