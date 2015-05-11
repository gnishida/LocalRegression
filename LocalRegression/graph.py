import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



# データの読み込み
X = []
for l in open('results.txt').readlines():
	list = l.split(',')
	rec = [float(d) for d in list]
	X.append(rec)

X = np.array(X)

plt.xlabel('sigma')
plt.ylabel('RMSE')
plt.plot(X[:,0], X[:,1], 'b.-', label='RMSE');
plt.legend(loc='lower right')
plt.show()

