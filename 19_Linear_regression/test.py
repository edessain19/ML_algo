import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MyLinearRegression import MyLinearRegression as MyLR

def predict_value(lr, X, Y):
	pred =  lr.predict_(X)
	return (pred)

def train_model(lr, X, Y):
	thet = lr.fit_(X, Y)
	lr.theta = thet
	return (thet)

if __name__ == "__main__":
	data = pd.read_csv("data.csv")
	X = np.array(data[['km']]).reshape(-1,1)
	Y = np.array(data[['price']]).reshape(-1,1)

	lr = MyLR([0.0, 0.0])
	
	print("prediction without training the model :")
	pred1 = predict_value(lr, X, Y)
	print(pred1)

	print("Theta after training the model :")
	new_theta = train_model(lr, X, Y)
	print(new_theta)

	print("prediction using the new theta :")
	pred2 = predict_value(lr, X, Y)
	print(pred2)