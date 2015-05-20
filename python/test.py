from math import ceil
import numpy as np
from scipy import linalg
 
 
def lowess(x, y, f=2./3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.

    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations."""
    n = len(x)
    r = int(ceil(f*n))
    
    print x
    
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    print h

    print (x[:,None] - x[None,:]) /h
    
    w = np.clip(np.abs((x[:,None] - x[None,:]) / h), 0.0, 1.0)
    w = (1 - w**3)**3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:,i]
            print weights*y
            print np.sum(weights*y)
            print np.sum(weights*y*x)
                        
            b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
            A = np.array([[np.sum(weights), np.sum(weights*x)],
                   [np.sum(weights*x), np.sum(weights*x*x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1]*x[i]
            break
        break
        
        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2)**2
 
    return yest
 
if __name__ == '__main__':
	X = []
	for l in open('data.txt').readlines():
		list = l.split(',')
		rec = [float(d) for d in list]
		X.append(rec)

	X = np.array(X)

	yest = lowess(X[:,0], X[:,1], f=0.25, iter=3)
 
	import pylab as pl
	pl.clf()
	pl.plot(X[:,0], X[:,1], 'o', label='y noisy')
	pl.plot(X[:,0], yest, label='y pred')
	pl.legend()
	pl.show()