import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn.covariance
import sklearn.linear_model
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr

#Default parametres

default_data_type = "case"
default_noise_std = 0.1
default_sample_size = 1000
default_indim = 2
default_outdim = 10
default_alpha = 0.1
default_data_count = 10
verbose = False

#Testing

def plot_performances(data_type=default_data_type, m=default_data_count, data_params = [], methods=[]):
	if verbose:
		print "making the data"
	data = compare_performances(data_type=data_type, m=m, data_params=data_params, methods=methods)
	if verbose:
		print "data finished"
	plt.figure()
	x = range(m)
	for method in data:
		plt.plot(x, data[method], label = method)
	plt.legend()
	plt.show()

def compare_performances(data_type=default_data_type, m=default_data_count, data_params = [], methods=[]):
	if verbose:
		print "making Cs"
	Cs = [np.cov(generate_data(params=data_params, data_type=data_type)) for _ in xrange(m)]
	if verbose:
		print "making Graph"
	Graph = generate_data(params=data_params, data_type=data_type, only_shape=True)
	if verbose:
		print "evaluate methods"
	return evaluate_methods(Cs, Graph, methods=methods)

def evaluate_methods(Cs, Graph, methods = []):
	return {method: [precision_recall(use_method(params=[C, sum(sum(1 for x in row if x) for row in Graph)], method_name=method), Graph)[0] for C in Cs] for method in methods}

def precision_recall(P_est, Graph, cutoff=0.0):
	TP, FP, TPC = 0, 0, 0
	for row in np.dstack((P_est, Graph)):
		for p in row:
			if (p[1]):
				TPC += 1
				if (abs(p[0]) > cutoff):
					TP += 1
			elif (abs(p[0]) > cutoff):
				FP += 1
	if (TP + FP == 0):
		return 0, 0
	return 1.0*TP/(TP + FP), 1.0*TP/TPC

#Data generation

def generate_data(params=[], data_type=default_data_type, only_shape=False):
	return globals()["generate_" + data_type](*params, only_shape=only_shape)

def generate_case(indim = default_indim, outdim = default_outdim, samples = default_sample_size, noisestd=default_noise_std, only_shape=False):
	if (only_shape):
		return np.array([[1 if ((min(i,j) < indim) or (i==j)) else 0 for i in xrange(indim + outdim)] for j in xrange(indim + outdim)])
	A = npr.randn(indim, samples)
	X = npr.randn(outdim, indim)
	B = np.dot(X, A) + noisestd*npr.randn(outdim, samples)
	B = B / np.std(B, 1)[:,np.newaxis]
	return np.vstack((A, B))

def generate_band(dim = default_outdim, samples = default_sample_size, only_shape=False):
	if (only_shape):
		return np.array([[1 if (abs(i - j) <= 2) else 0 for i in xrange(outdim)] for j in xrange(outdim)])
	omega = np.eye(dim)
	p1, p2 = 0.6, 0.3
	for i in xrange(dim - 1):
		omega[i][i + 1] = p1
		omega[i + 1][i] = p1
	for i in xrange(dim - 2):
		omega[i][i + 2] = p2
		omega[i + 2][i] = p2
	omeinv = scipy.linalg.inv(omega)
	D = np.diag(1+4*np.random.random(dim))
	C = np.dot(D, np.dot(omeinv, D))
	A = np.linalg.cholesky(C)
	X = npr.randn(dim, samples)
	B = np.dot(A, X)
	return B

#Methods

def use_method(params = [], method_name="glasso", option="P_est", alpha = None):
	if (option == "P_est"):
		if (alpha != None):
			globals()[method_name](params[0], alpha=alpha)
		else:
			if verbose:
				print "using a method " + method_name
			TPC = params[1]
			ala = 0.1
			P = globals()[method_name](params[0], alpha=ala)
			while (sum(sum(1 for i in l if abs(i) > 0) for l in P) > TPC):
				ala *= 2
				P = globals()[method_name](params[0], alpha=ala)
			yla = ala
			ala /= 2
			if verbose:
				print "binary search"
			for _ in xrange(20):
				kes = (ala + yla)/2
				P = globals()[method_name](params[0], alpha=kes)
				TC = sum(sum(1 for i in l if abs(i) > 0) for l in P)
				if (TC >= TPC):
					ala = kes
				else:
					yla = kes
			if verbose:
				print "finished with method"
			return globals()[method_name](params[0], alpha=ala)

def naive(C, alpha = default_alpha):
	P = scipy.linalg.inv(C)
	P[P < alpha] = 0
	return P

def glasso(C, alpha = default_alpha):
	return sklearn.covariance.graph_lasso(C, alpha)[1]

def scio(C, alpha = default_alpha):
	scio = importr('scio')
	return np.array(scio.scio(C, alpha)[0])

def fastclime(C, alpha = default_alpha):
	d = {'fastclime.lambda': 'fastclime_lamb'}
	fastclime = importr('fastclime', robject_translations = d)
	out1 = fastclime.fastclime(C)
	O = fastclime.fastclime_lamb(out1[4], out1[5], alpha)
	return np.array(O[0])
