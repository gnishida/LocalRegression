﻿import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



# データの読み込み
X = []
for l in open('trueX.txt').readlines():
	list = l.split(' ')
	rec = [float(d) for d in list]
	X.append(rec)
X = np.array(X)

Y = []
for l in open('trueY.txt').readlines():
	list = l.split(' ')
	rec = [float(d) for d in list]
	Y.append(rec)
Y = np.array(Y)

plt.xlabel('x')
plt.ylabel('y')
plt.plot(X[:,0], Y[:,0], 'o', label='sin');
plt.legend(loc='upper right')
plt.show()

