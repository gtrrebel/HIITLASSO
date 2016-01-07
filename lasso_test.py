import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn.covariance
import sklearn.linear_model

np.set_printoptions(precision=2)

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


def generate_linmixt(indim, outdim, samples, noisestd=0.1):
	A = npr.randn(indim, samples)
	X = npr.randn(outdim, indim)
	B = np.dot(X, A) + noisestd*npr.randn(outdim, samples)
	#B = B / np.std(B, 1)[:,np.newaxis]
	print X
	print np.dot(X, X.transpose())
	print np.vstack((B))[:, 0]
	print np.vstack((A, B))[:, 0]
	return np.vstack((A, B))
	#return np.vstack((A, B))

def plot_invcov(X, cutoff=10.0):
	C = np.cov(X)
	print("Covariance matrix:")
	print(C)
	Cinv = scipy.linalg.inv(C)
	print("Inverse of covariance matrix:")
	print(Cinv)
	plt.spy(np.abs(Cinv), precision=cutoff)
	plt.show()

def plot_glassoprec0(X, alpha1=0.9, alpha2=1.0):
	C = np.cov(X)
	alphas = np.linspace(alpha1, alpha2, 16)
	for i, alpha in enumerate(alphas):
		Cest, Pest = sklearn.covariance.graph_lasso(C, alpha)
		plt.subplot(4, 4, i+1)
		plt.spy(Pest)
	plt.show()

def plot_glassoprec0_2(X, indim, outdim, alpha1=0.9, alpha2=1.0):
	C = np.cov(X)
	precisions = []
	recalls = []
	alphas = np.linspace(alpha1, alpha2, 16)
	for i, alpha in enumerate(alphas):
		Cest, Pest = sklearn.covariance.graph_lasso(C, alpha)
		plt.subplot(4, 4, i+1)
		plt.spy(Pest)
		pr, rec = precision_recall(Pest, indim, outdim)
		precisions.append(pr)
		recalls.append(rec)
	print precisions
	print recalls
	plt.show()

indim, outdim = 2, 4

X = generate_linmixt(indim, outdim, 1000000)
plot_glassoprec0_2(X, indim, outdim)
plot_invcov(X)