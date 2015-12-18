import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn.covariance
import sklearn.linear_model


def generate_linmixt(indim, outdim, samples, noisestd=0.1):
    A = npr.randn(indim, samples)
    X = npr.randn(outdim, indim)
    B = np.dot(X, A) + noisestd*npr.randn(outdim, samples)
    B = B / np.std(B, 1)[:,np.newaxis]
    return np.vstack((A, B))

def plot_invcov(X, cutoff=10.0):
    C = np.cov(X)
    Cinv = scipy.linalg.inv(C)
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

X = generate_linmixt(2, 10, 1000)
plot_glassoprec0(X)
plot_invcov(X)