import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
def load_data():
	print("Loading Data ...")
	df = pd.read_csv("ex1data1.txt", header=None)
	x = np.array(df[0])
	y = np.ones(x.size).reshape(x.size, 1)
	y[:, 0] = df[1]
	print("Head of data : ")
	print(df.head())
	print("Plotting scatter graph of the loaded data...")
	#plt.scatter(x, y, marker='x', color='r')
	#plt.show()
	return x, y

def compute_cost(x, y, theta):
	X = np.ones(x.size*2).reshape(x.size, 2)
	X[:, 1] = x
	m = x.size
	J = ((((X.dot(theta) - y)**2).sum())/(2*m))
	return J

def gradient_descent(x, y, theta, alpha, iteration):
	X = np.ones(x.size*2).reshape(x.size, 2)
	X[:, 1] = x
	m = x.size
	J_h = np.ones(iteration)
	for i in range(iteration):
		theta = theta - (alpha/m)*(X.T.dot(X.dot(theta)-y))
		J_h[i] = compute_cost(x, y, theta)
	return theta


if __name__ == '__main__': 
	x, y = load_data()
	compute_cost(x, y, [-1,2])
	if input("Enter to continue, q to quit ") == 'q':
		print("exiting...")
		exit()
	print("Cost for theta[0:0]: " + str(compute_cost(x, y, np.array([[0],[0]]))))	
	print("Expected value: 32.07")
	print("Cost for theta[-1;2]: " + str(compute_cost(x, y, np.array([[-1],[2]]))))
	print("Expected value: 54.24")
	f_theta = gradient_descent(x, y, np.array([[0],[0]]), 0.01, 1000)
	print("Theta after traning: " + str(f_theta))