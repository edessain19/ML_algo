import random as rd
import numpy as np

class MyLinearRegression():
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""
	def __init__(self, theta, alpha=0.000001, max_iter=10000, lambda_=0.5):
		self.alpha = alpha #learning rate
		self.max_iter = max_iter #number of iteration
		self.theta = theta
		self.lambda_ = lambda_

	def errno_(func):
		def inner(*args):
			if len(args) >= 2  and type(args[1]) != np.ndarray:
				print("X are not numpy array")
				exit(1)
			elif len(args) >= 3 and type(args[2]) != np.ndarray:
				print("y isn't numpy array")
				exit(1)
			elif len(args) == 4 and type(args[3]) != float:
				print("Proportion isn't a float")
				exit(1)
			elif len(args) >= 3 and (np.shape(args[2])[0] != 1 \
					| np.shape(args[2])[1] != 1):
				print("y not a vector")
				exit(1)
			return func(*args)
		return inner

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
		x_train, x_test, y_train, y_test = [], [], [], []
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

	def add_polynomial_features(self, x, power):
		if type(x) != np.mdarray | type(power) != int:
			print("Polynomial argument type invalid")
			exit(1)
		elif len(x) == 0:
			return None
		result = []
		row_ = []
		i = 1
		for row in x:
			row_ = []
			i = 1
			while i <= power:
				for xi in row:
					row_.append(xi ** i)
				i += 1
			result.append(row_)
		return np.array(result)

	def predict_(self, x):
		"""
		Fonction that return y_hat, the predicted value of y according to theta and X
		"""
		Y = []
		X = np.ones((len(x), 1))
		X = np.c_[X, x]
		Y = X.dot(self.theta)
		return Y

	def fit_(self, x, y):
		"""
		Fonction that find the optimized theta using the gradient descent
		may find only a minimum local
		"""
		m = len(x)
		X = np.c_[np.ones((len(x), 1)), x]
		y_hat = self.predict_(x)
		y = np.squeeze(y)
		i = 0
		while i < self.max_iter:
			# theta_bis= np.copy(self.theta)
			# theta_bis[0] = 0
			# gradient = (1 / m) * ((np.dot(np.transpose(X), (np.dot(X, self.theta) - y))) + (self.lambda_ * theta_bis))
			# self.theta = self.theta - self.alpha * gradient
			# i += 1
			gradient = (1 / m) * (np.dot(np.transpose(X), self.predict_(x) - y))
			self.theta = self.theta - self.alpha * gradient
			i += 1
		return self.theta

	def normal_equation(self, x, y):
		"""
		Fonction tha find the optimized theta, no need to normilize the dataset,
		but doesn't works if there is to many features (x) (more than 1 000 000)
		find the general minimum
		"""
		x = np.c_[np.ones((len(x), 1)), x]
		result_inv = np.linalg.inv(np.dot(np.transpose(x), x))
		result = np.dot(np.dot(result_inv, np.transpose(x)), y)
		result = np.squeeze(result)
		self.theta = result
		return result

	def r2score_(self, x, y):
		"""
		Cost function that evalue the prediction of the model
		"""
		solution = 0.0
		y_hat = self.predict_(x)
		y_m = sum(x)/len(x)
		y = np.squeeze(y)
		solution = 1 - (sum((y - y_hat) ** (2))/sum((y - y_m) ** (2)))
		return solution

	def cost_(self, x, y):
		"""
		Cost function that evalue the prediction of the model
		"""
		m = len(x)
		y_hat = self.predict_(x)
		y = np.squeeze(y)
		theta_p2 = self.theta * self.theta
		theta_p2[0] = 0
		return (sum((y_hat - y) ** 2) + (self.lambda_ * sum(theta_p2))) / (2 * m)