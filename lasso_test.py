import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn.covariance
import sklearn.linear_model
from pandas import *
import pandas.rpy.common as com
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr

np.set_printoptions(precision=2)

def compare_estimate(p1, p2):
	return (int) (p1[2] > p2[2]) 

def precision_recall(Pest, indim, outdim, cutoff=0.0):
	TP = 0
	FP = 0
	for i in xrange(indim):
		for j in xrange(indim):
			if (abs(Pest[i][j]) > cutoff):
				TP += 1
	for i in xrange(indim):
		for j in xrange(outdim):
			if (abs(Pest[i][indim + j]) > cutoff):
				TP += 1
			if (abs(Pest[indim + j][i]) > cutoff):
				TP += 1
	for i in xrange(outdim):
		for j in xrange(outdim):
			if (i == j)&(abs(Pest[indim + i][indim + j]) > cutoff):
				TP += 1
			if (i != j)&(abs(Pest[indim + i][indim + j]) > cutoff):
				FP += 1
	if (TP + FP == 0):
		return 0, 0
	return 1.0*TP/(TP + FP), 1.0*TP/(indim*(indim + 2*outdim) + outdim)

def precision_recall_curve(Pest, indim, outdim, cutoff=0.0):
	precisions = []
	recalls = []
	triplets = []
	TP = 0
	FP = 0
	for i in xrange(indim + outdim):
		for j in xrange(indim + outdim):
			if (abs(Pest[i][j]) > cutoff):
				triplets.append([i, j, abs(Pest[i][j])])
	REL = (indim*(indim + 2*outdim) + outdim)
	sorted(triplets, cmp=compare_estimate)
	for t in triplets:
		if ((min(t[0], t[1]) < indim)|(t[0] == t[1])):
			TP += 1
		else:
			FP += 1
		precisions.append(1.0*TP/(TP + FP))
		recalls.append(1.0*TP/REL)
	return precisions, recalls


def generate_linmixt(indim, outdim, samples, noisestd=0.1):
	A = npr.randn(indim, samples)
	X = npr.randn(outdim, indim)
	B = np.dot(X, A) + noisestd*npr.randn(outdim, samples)
	B = B / np.std(B, 1)[:,np.newaxis]
	return np.vstack((A, B))

def plot_invcov(X, cutoff=10.0):
	plt.figure()
	C = np.cov(X)
	Cinv = scipy.linalg.inv(C)
	plt.spy(np.abs(Cinv), precision=cutoff)
	plt.show()

def plot_invcov_2(X, indim, outdim, cutoff=10.0):
	plt.figure()
	C = np.cov(X)
	Cinv = scipy.linalg.inv(C)
	plt.spy(np.abs(Cinv), precision=cutoff)
	pr, rec = precision_recall_curve(Cinv, indim, outdim, cutoff)
	plt.figure()
	ax = plt.subplot()
	ax.set_ylabel("precision")
	ax.set_xlabel("recall")
	plt.plot(rec, pr)

def plot_glassoprec0(X, alpha1=0.1, alpha2=1.0):
	C = np.cov(X)
	alphas = np.linspace(alpha1, alpha2, 16)
	for i, alpha in enumerate(alphas):
		Cest, Pest = sklearn.covariance.graph_lasso(C, alpha)
		plt.subplot(4, 4, i+1)
		plt.spy(Pest)

def plot_glassoprec0_2(X, indim, outdim, alpha1=0.6, alpha2=0.7, colours = False):
	C = np.cov(X)
	precisions = []
	recalls = []
	alphas = np.linspace(alpha1, alpha2, 16)
	for i, alpha in enumerate(alphas):
		Cest, Pest = sklearn.covariance.graph_lasso(C, alpha)
		plt.subplot(4, 4, i+1)
		if colours:
			plt.imshow([[abs(i) for i in P] for P in Pest],interpolation='none',cmap='Reds')
		else:
			plt.spy(Pest)
		pr, rec = precision_recall_curve(Pest, indim, outdim)
		#plt.plot(pr, rec)
		precisions.append(pr)
		recalls.append(rec)
		print sum(sum(1 for i in Pe if abs(i) > 0) for Pe in Pest)
	if colours:
		plt.colorbar()
	print (indim*(indim + 2*outdim) + outdim)
	plt.show()

def plot_glassoprec0_3(X, indim, outdim, alpha = 0.8):
	C = np.cov(X)
	Cest, Pest = sklearn.covariance.graph_lasso(C, alpha)
	plt.figure()
	plt.imshow([[abs(i) for i in P] for P in Pest],interpolation='none',cmap='Reds')
	plt.colorbar()
	plt.figure()
	plt.spy(Pest)
	pr, rec = precision_recall_curve(Pest, indim, outdim)
	plt.figure()
	ax = plt.subplot()
	ax.set_ylabel("precision")
	ax.set_xlabel("recall")
	plt.plot(rec, pr)
	print sum(sum(1 for i in Pe if abs(i) > 0) for Pe in Pest)
	print (indim*(indim + 2*outdim) + outdim)

def plot_all0_3(X, indim, outdim, alpha = 0.2, cutoff = 10.0):
	C = np.cov(X)
	Cest, Pest = sklearn.covariance.graph_lasso(C, alpha)
	Cinv = scipy.linalg.inv(C)
	pr, rec = precision_recall_curve(Pest, indim, outdim)
	pr2, rec2 = precision_recall_curve(Cinv, indim, outdim, cutoff)

	plt.figure()
	ax1 = plt.subplot(2,2,1)
	ax1.set_title("Glasso")
	plt.imshow(abs(Pest),interpolation='none',cmap='Reds')
	plt.colorbar()
	plt.subplot(2,2,2)
	plt.spy(Pest)
	ax2 = plt.subplot(2,2,3)
	ax2.set_title("Naive")
	plt.imshow(abs(Cinv),interpolation='none',cmap='Reds')
	plt.colorbar()
	plt.subplot(2,2,4)
	plt.spy(np.abs(Cinv), precision=cutoff)

	plt.figure()
	ax = plt.subplot()
	ax.set_ylabel("precision")
	ax.set_xlabel("recall")
	ax.set_ylim((0, 1.1))
	ax.set_xlim((0, 1.1))
	plt.plot(rec, pr, linewidth=1.0, label="glasso")
	plt.plot(rec2, pr2, linewidth=1.0, label="naive")
	plt.legend()

	print "Glasso nonzero:", sum(sum(1 for i in Pe if abs(i) > 0) for Pe in Pest)
	print "Naive nonzero:", sum(sum(1 for i in Pe if abs(i) > cutoff) for Pe in Cinv)
	print "Real nonzero:", (indim*(indim + 2*outdim) + outdim)

	plt.show()

def find_alpha(X, indim, outdim):
	C = np.cov(X)
	alphas = np.linspace(0.8, 0.9, 100)
	for alpha in alphas:
		Cest, Pest = sklearn.covariance.graph_lasso(C, alpha)
		print alpha, sum(sum(1 for i in Pe if abs(i) > 0) for Pe in Pest)

def fastclime_test(X):
	nr,nc = X.shape
	Xr = ro.r.matrix(X, nrow=nr, ncol=nc)
	ro.r.assign("X", X)
	fastclime = importr('fastclime')

indim, outdim = 10, 50
X = generate_linmixt(indim, outdim, 1000)

fastclime_test(X)

if False:
	plot_all0_3(X, indim, outdim)
