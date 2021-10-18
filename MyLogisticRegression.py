import numpy as np
import random as rd

class MyLogisticRegression():
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""
	def __init__(self, theta, alpha=0.001, n_cycle=500000):
		self.alpha = alpha
		self.n_cycle = n_cycle
		self.theta = theta

	def predict_(self, x):
		X = np.c_[np.ones((len(x), 1)), x]
		X = np.dot(X, self.theta)
		return 1 /(1 + np.exp(-1 * X))

	def cost_(self, x, y, eps=1e-15):
		y = np.squeeze(y)
		y_hat = self.predict_(x)
		return sum((y * np.log(y_hat + eps)) + ((1 - y) * np.log(1 - y_hat + eps))) / -len(y)

	def fit_(self, x, y):
		m = len(x)
		X = np.c_[np.ones((len(x), 1)), x]
		y_hat = self.predict_(x)
		y = np.squeeze(y)
		i = 0
		while i < self.n_cycle:
			gradient = (1 / m) * (np.dot(np.transpose(X), (np.dot(X, self.theta) - y)))
			self.theta = self.theta - self.alpha * gradient
			i += 1
		return self.theta

	def zscore(self, x):
		"""
		Fonction used to normalize the dataset, uselfull if the dataset is huge
		and you want to use the gradient descente, avoid the (nan).
		"""
		m = sum(x) / len(x)
		std = (sum((x - m) ** 2))/ len(x)
		return (x - m)/(std ** (1/2))

	def data_spliter(self, x, y, proportion):
		"""
		Fonction used to split the data in two set, one to train the model
		and one other to test it.
		"""
		x_train = []
		x_test = []
		y_train = []
		y_test = []
		prop = proportion * len(x)
		index = [i for i in range(len(x))]
		rd.shuffle(index)
		i = 0
		while i < prop:
			x_train.append(x[index[i]])
			y_train.append(y[index[i]])
			i += 1
		while i < len(x):
			x_test.append(x[index[i]])
			y_test.append(y[index[i]])
			i += 1
		return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

	def label(self, y, label):
		lst = []
		for i in y:
			if i == label:
				lst.append(1)
			else:
				lst.append(0)
		return np.array(lst).reshape(-1, 1)