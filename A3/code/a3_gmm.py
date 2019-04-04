from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
import numpy as np
import os, fnmatch
import random
import time
import pickle

dataDir = '../data'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    d = myTheta.mu.shape[1]
    muM, sigmaM = myTheta.mu[m], myTheta.Sigma[m]
    if len(preComputedForM) == 0:
        sq = np.square(x - muM)

        num = -0.5 * np.sum(sq / sigmaM)
        # denom = np.log(np.power(2 * np.pi, d / 2) * np.sqrt(np.multiply.reduce(sigmaM)))
        denom = np.log(np.power(2 * np.pi, d / 2)) + 0.5 * np.log(np.multiply.reduce(sigmaM))

        return num - denom
    else:
        sigmaM_2 = np.sqrt(sigmaM)
        return - np.sum((0.5 - muM) * x * sigmaM_2) - preComputedForM[0]

def pre_compute_m(m, myTheta):
    d = myTheta.mu.shape[1]
    muM, sigmaM = myTheta.mu[m], myTheta.Sigma[m]
    muM2, sigmaM2 = np.square(muM), np.square(sigmaM)

    partA = np.sum(muM2 / (2 * sigmaM2))
    partB = 0.5 * d * np.log(2 * np.pi)
    partC = 0.5 * np.log(np.multiply.reduce(sigmaM2))

    return partA + partB + partC


def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    M = myTheta.mu.shape[0]
    logbmx = [log_b_m_x(i, x, myTheta) for i in range(M)]
    num = logsumexp(logbmx[m], b=[myTheta.omega[m]])
    denom = logsumexp(logbmx, b=myTheta.omega)

    return num - denom


def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''
    M, T = log_Bs.shape[0], log_Bs.shape[1]
    lptt = [logsumexp(log_Bs[:, t], b=myTheta.omega) for t in range(T)]
    return np.sum(lptt)


def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    # Initialize theta
    myTheta = theta( speaker, M, X.shape[1] )
    myTheta.omega = np.random.random((M, 1))
    myTheta.Sigma = np.random.random((M, X.shape[1]))
    myTheta.mu = np.random.random((M, X.shape[1]))

    i = 0
    prev_L, improvement = -float('inf'), float('inf')
    T = X.shape[0]
    print(M, T)

    while i < maxIter and improvement >= epsilon:
        print("Loop", i)
        start_time = time.time()

        preCompM = [None] * M
        # Compute Intermediate Results
        lbmx, lpmx = np.zeros((M, T)), np.zeros((M, T))
        for t in range(T):
            if t % 5000 == 0:
                print(t)
            for m in range(M):
                preCompM[m] = pre_compute_m(m, myTheta) if preCompM[m] is None else preCompM[m]
                lbmx[m, t] = log_b_m_x(m, X[t], myTheta, preComputedForM=[preCompM[m]])

            logbmx = lbmx[:, t]
            denom = logsumexp(logbmx, b=myTheta.omega)
            for m in range(M):
                num = logsumexp([logbmx[m]], b=[myTheta.omega[m]])
                lpmx[m, t] = num - denom

        # Compute Likelihood (X, theta)
        L = logLik(lbmx, myTheta)

        # Update Parameters (theta, X, L)
        for m in range(M):
            myTheta.omega[m] = np.sum(np.exp(lpmx[m])) / T

            sum1, sum2, sump = np.zeros(X[0].shape), np.zeros(X[0].shape), 0.0
            for t in range(T):
                sum1 += np.exp(lpmx[m, t]) * X[t]
                sum2 += np.exp(lpmx[m, t]) * X[t] * X[t]
                sump += np.exp(lpmx[m, t])
            print(lpmx)
            print(sump, sum1, sum2)
            myTheta.mu[m] = sum1 / sump
            myTheta.Sigma[m] = sum2 / sump - np.square(myTheta.mu)[m]

        improvement = L - prev_L
        print("Improvement", improvement, "taking time", time.time() - start_time)
        prev_L = L
        i += 1

    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    X = mfcc
    T = X.shape[0]
    logLiks = []

    print("Test correctID", correctID)

    for myTheta in models:
        print("    Testing model:", myTheta.name)
        preCompM = [None] * M
        # Compute Intermediate Results
        lbmx, lpmx = np.zeros((M, T)), np.zeros((M, T))
        for t in range(T):
            for m in range(M):
                preCompM[m] = pre_compute_m(m, myTheta) if preCompM[m] is None else preCompM[m]
                lbmx[m, t] = log_b_m_x(m, X[t], myTheta, preComputedForM=[preCompM[m]])

            logbmx = lbmx[:, t]
            denom = logsumexp(logbmx, b=myTheta.omega)
            for m in range(M):
                num = logsumexp([logbmx[m]], b=[myTheta.omega[m]])
                lpmx[m, t] = num - denom

        # Compute Likelihood (X, theta)
        L = logLik(lbmx, myTheta)
        logLiks.append(L)

    print("--- Test result ---")
    print(correctID)
    idx = (-np.array(logLiks)).argsort()[:k]
    for i in range(k):
        index = idx[i]
        print(models[index].name, logLiks[index])

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )

            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate
    numCorrect = 0
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k )
    accuracy = 1.0*numCorrect/len(testMFCCs)

