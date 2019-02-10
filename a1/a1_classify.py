from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import sklearn.base
from scipy import stats
import numpy as np
import argparse
import sys
import os
import csv

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.diag(C).sum() / C.sum()

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return [C[k, k] / C[k].sum() for k in range(C.shape[0])]


def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return [C[k, k] / C[:, k].sum() for k in range(C.shape[1])]


def class31(filename):
    ''' This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    print('TODO Section 3.1')
    iBest = -1

    npzfile = np.load(filename)
    arr = npzfile[npzfile.files[0]]
    npzfile.close()

    X, y = arr[:, :-1], arr[:, -1].astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape)

    classifiers = {
        1: SVC(kernel="linear"),
        2: SVC(kernel="rbf", gamma=2),
        3: RandomForestClassifier(n_estimators=10, max_depth=5),
        4: MLPClassifier(alpha=0.05),
        5: AdaBoostClassifier()
    }

    csvdata, accu = [0 for _ in range(len(classifiers))], [0 for _ in range(len(classifiers))]

    for idx, clf in classifiers.items():
        clf.fit(X_train, y_train)
        res = clf.predict(X_test)
        print("Finish training & predicting for classifier", idx)
        cmat = np.matrix(confusion_matrix(y_test, res))
        print(cmat)
        csvdata[idx - 1] = [idx, accuracy(cmat)] + recall(cmat) + precision(cmat) + np.concatenate(cmat.tolist()).tolist()
        accu[idx - 1] = accuracy(cmat)
        print("Accuracy", accuracy(cmat))
        print("------------------------------------------------\n")

    with open("a1_3.1.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(csvdata)

    iBest = accu.index(max(accu)) + 1
    print("Best classifier", iBest)

    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    train_size = [1000, 5000, 10000, 15000, 20000]

    classifiers = {
        1: SVC(kernel="linear"),
        2: SVC(kernel="rbf", gamma=2),
        3: RandomForestClassifier(n_estimators=10, max_depth=5),
        4: MLPClassifier(alpha=0.05),
        5: AdaBoostClassifier()
    }

    accuracies = []
    X_1k, y_1k = None, None

    for size in train_size:
        X_train_batch, y_train_batch, _, _ = train_test_split(X_train, y_train, train_size=(size/X_train.shape[0]))

        if size == 1000:
            X_1k, y_1k = X_train_batch, y_train_batch

        # Clone classifier, fit and predict, generate confusion matrix and calculate accuracy
        clf = sklearn.base.clone(classifiers[iBest])
        clf.fit(X_train_batch, y_train_batch)
        res = clf.predict(X_test)
        cmat = np.matrix(confusion_matrix(y_test, res))
        accuracies.append(accuracy(cmat))

    with open("a1_3.2.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(accuracies)

    return (X_1k, y_1k)

def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')
    ks = [5, 10, 20, 30, 40, 50]
    res = []

    X_1k_best5, X_train_best5 = None, None

    for k in ks:
        selector_1k = SelectKBest(chi2, k)
        X_new_1k = selector_1k.fit_transform(X_1k, y_1k)
        pp_1k = selector_1k.pvalues_

        selector_train = SelectKBest(chi2, k)
        X_new_train = selector_train.fit_transform(X_train, y_train)
        pp_train = selector_train.pvalues_

        if k == 5:
            X_1k_best5, X_train_best5 = X_new_1k, X_new_train

        res.append([k] + np.array(pp_train).tolist())
        res.append([-1, k] + np.array(pp_train)[selector_train.get_support()].tolist())

    classifiers = {
        1: SVC(kernel="linear"),
        2: SVC(kernel="rbf", gamma=2),
        3: RandomForestClassifier(n_estimators=10, max_depth=5),
        4: MLPClassifier(alpha=0.05),
        5: AdaBoostClassifier()
    }

    accuracies = []

    clf_1k = sklearn.base.clone(classifiers[i])
    clf_1k.fit(X_1k_best5, y_1k)
    res = clf_1k.predict(X_test)
    cmat = np.matrix(confusion_matrix(y_test, res))
    accuracies.append(accuracy(cmat))

    clf_train = sklearn.base.clone(classifiers[i])
    clf_train.fit(X_train_best5, y_train)
    res = clf_train.predict(X_test)
    cmat = np.matrix(confusion_matrix(y_test, res))
    accuracies.append(accuracy(cmat))

    res.append(accuracies)

    with open("a1_3.3.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(res)


def class34( filename, i ):
    ''' This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    print('TODO Section 3.4')

    npzfile = np.load(filename)
    arr = npzfile[npzfile.files[0]]
    npzfile.close()

    X, y = arr[:, :-1], arr[:, -1].astype('int')

    classifiers = {
        1: SVC(kernel="linear"),
        2: SVC(kernel="rbf", gamma=2),
        3: RandomForestClassifier(n_estimators=10, max_depth=5),
        4: MLPClassifier(alpha=0.05),
        5: AdaBoostClassifier()
    }

    fold_accuracy = []

    kf = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        accuracies = [0 for _ in range(len(classifiers))]

        for idx, classifier in classifiers.items():
            clf = sklearn.base.clone(classifier)
            clf.fit(X_train, y_train)
            res = clf.predict(X_test)
            print("Finish training & predicting for classifier", idx)
            cmat = np.matrix(confusion_matrix(y_test, res))
            accuracies[idx] = accuracy(cmat)
            print("Accuracy", accuracy(cmat))
            print("------------------------------------------------\n")

        fold_accuracy.append(accuracies)

    # Compare accuracies between i and other classifier
    pvalues = []
    acc_mat = np.array(fold_accuracy).transpose()

    for idx in range(5):
        if idx + 1 == i:
            continue
        pv = stats.ttest_rel(acc_mat[i - 1], acc_mat[idx])
        pvalues.append(pv)

    fold_accuracy.append(pvalues)

    with open("a1_3.4.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(fold_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    np.savez_compressed('train.npz', x_train=X_train, y_train=y_train, ibest=np.array(iBest))
    np.savez_compressed('test.npz', x_train=X_test, y_train=y_test, ibest=np.array(iBest))
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    np.savez_compressed('onek.npz', xonek=X_1k, yonek=y_1k, ibest=np.array(iBest))
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.input, iBest)
