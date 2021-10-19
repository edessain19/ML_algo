import numpy as np
import random as rd

class MyNeuralNetwork():
	def __init__(self, nn_params, input_layer_size, hidden_layer_size, num_labels, lambda_=0.0):
		self.nn_params = nn_params
		self.input_layer_size = input_layer_size
		self.hidden_layer_size = hidden_layer_size
		self.num_labels = num_labels

	def sigmoid(z):
		"""
		Computes the sigmoid of z.
		"""
		return 1.0 / (1.0 + np.exp(-z))

	def nnGradientComputation(self, X, y, Theta1, Theta2):
		"""
		Implements the neural network cost function and gradient for a two layer neural
		network which performs classification.
		"""
		m = y.size
		a1 = np.concatenate([np.ones((m, 1)), X], axis=1)
		a2 = self.sigmoid(a1.dot(Theta1.T))
		a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
		a3 = self.sigmoid(a2.dot(Theta2.T))

		y_matrix = y.reshape(-1)
		y_matrix = np.eye(num_labels)[y_matrix]

		temp1 = Theta1
		temp2 = Theta2

		# Add regularization term
		reg_term = (lambda_ / (2 * m)) * (np.sum(np.square(temp1[:, 1:])) + np.sum(np.square(temp2[:, 1:])))

		J = (-1 / m) * np.sum((np.log(a3) * y_matrix) + np.log(1 - a3) * (1 - y_matrix)) + reg_term

		# Backpropogation
		delta_3 = a3 - y_matrix
		delta_2 = delta_3.dot(Theta2)[:, 1:] * sigmoidGradient(a1.dot(Theta1.T))
		Delta1 = delta_2.T.dot(a1)
		Delta2 = delta_3.T.dot(a2)

		# Add regularization to gradient
		Theta1_grad = (1 / m) * Delta1
		Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]

		Theta2_grad = (1 / m) * Delta2
		Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]

		grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
		return J

	def cost_function(self):