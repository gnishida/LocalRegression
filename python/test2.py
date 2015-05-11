import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def gaussian_kernel(x, x0, c, a=1.0):
	diff = x - x0
	dot_product = diff * diff.T
	return a * np.exp(dot_product / (-2.0 * c**2))

def get_weights(inputs, datapoint, c=1.0):
	n_rows = inputs.shape[0]
	
	# Create diagonal weight matrix from identity matrix
	weights = np.mat(np.eye(n_rows))
	for i in xrange(n_rows):
		weights[i, i] = gaussian_kernel(datapoint, inputs[i], c)
	
	return weights


def lwr_predict(training_inputs, training_outputs, datapoint, c=1.0):
	x = np.mat(training_inputs).T
	y = np.mat(training_outputs).T
	
	weights = get_weights(x, datapoint, c=c)
	
	x = np.c_[np.ones((x.shape[0],1)),x-datapoint]
	
	xt = x.T * (weights * x)
	betas = xt.I * (x.T * (weights * y))

	return betas[0,0]

if __name__ == '__main__':
	data_X = []
	data_Y = []
	for l in open('data.txt').readlines():
		list = l.split(',')
		rec = [float(d) for d in list]
		data_X.append(rec[0])
		data_Y.append(rec[1])

	x_list = range(1700, 3800, 100)
	y_list = []
	for x in x_list:
		y_pred = lwr_predict(data_X, data_Y, x, c=250.0);
		print y_pred
		y_list.append(y_pred)
	
	plt.plot(data_X, data_Y, 'o', label='samples')
	plt.plot(x_list, y_list, label='prediction')
	plt.legend()
	plt.show()
	