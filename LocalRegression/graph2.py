import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



# データの読み込み
X = []
for l in open('sinX.txt').readlines():
	X.append(float(l))
X = np.array(X)

Y = []
for l in open('sinY.txt').readlines():
	Y.append(float(l))
Y = np.array(Y)

plt.xlabel('sigma')
plt.ylabel('RMSE')
plt.plot(X, Y, 'o', label='RMSE');
plt.legend(loc='upper right')
plt.show()

