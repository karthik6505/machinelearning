#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pycluster based on: http://scikit-learn.org/stable/auto_examples/plot_classifier_comparison.html
@author: 
@url: 
@Code source: Gaël Varoquaux, Andreas Müller
@Modified for documentation by Jaques Grobler
@License: BSD 3 clause
"""
print(__doc__)


import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

h = .02  # step size in the mesh

NAMES = ["Nearest Neighbors", 
         "Linear SVM", 
         "RBF SVM", 
         "Decision Tree",
         "Random Forest", 
         "AdaBoost", 
         "Naive Bayes", 
         "LDA", 
         "QDA"]

CLASSIFIERS = [ KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='brute', metric='euclidean' ),
                SVC(kernel="linear", C=0.025),
                SVC(gamma=2, C=1),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                AdaBoostClassifier(),
                GaussianNB(),
                LDA(),
                QDA()]




# ###################################################################################
TIMESTAMPS = {}
def add_timestamp( name, ts1, ts0 ):
    global TIMESTAMPS
    t = ts1 - ts0
    TIMESTAMPS[ len(TIMESTAMPS) ] =  ( name, ts1, ts0, t )


# ###################################################################################
def fit_classifier( myclf, Xt, yt): 
    myclf.fit(Xt, yt)
    return myclf


# ###################################################################################
def test_classifier( myclf, Xp, yp): 
    score = myclf.score(Xp, yp)
    return score


# ###################################################################################
def unravel_decision_function(myclf, xx, yy):
    if hasattr(myclf, "decision_function"):
        Z = myclf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = myclf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    return ( Z, xx, yy )


# ###################################################################################
def apply_emsemble( subplot_idx, dataset ):
    # preprocess dataset, split into training and test part
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(DATASETS), len(CLASSIFIERS) + 1, subplot_idx)

    # Plot the training points # and testing points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    subplot_idx += 1
    # iterate over classifiers
    for name, clf in zip(NAMES, CLASSIFIERS):
        t0 = time.time()
        print subplot_idx, name, 'started...'
        ax = plt.subplot(len(DATASETS), len(CLASSIFIERS) + 1, subplot_idx)
        try:
            clf = fit_classifier( clf, X_train, y_train )
            score = test_classifier( clf, X_test, y_test ) 
            (Z, xx, yy ) = unravel_decision_function(clf, xx, yy)
        except Exception as exception:
            print type(exception).__name__
            print subplot_idx, name, 'exception reported, train/test...'

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        # Returns the log-probabilities of the sample for each class in the model. 
        # The columns correspond to the classes in sorted order, as they appear in the attribute classes_.


        add_timestamp( "CLASSIFIER-TRAIN/TEST[%s,%s]" % (name, subplot_idx), time.time(), t0 )
        t0 = time.time()

        try:
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points # and testing points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            ax.scatter(X_test[:, 0],  X_test[:, 1],  c=y_test,  cmap=cm_bright, alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
        except Exception as exception:
            print type(exception).__name__
            print subplot_idx, name, 'exception reported, plotting...'

        add_timestamp( "CLASSIFIER-PLOT[%s,%s]" % (name, subplot_idx), time.time(), t0 )
        subplot_idx += 1

    return subplot_idx


# ###################################################################################
if __name__ == "__main__":
    SHOW = False

    X, y = make_classification(n_samples=50, n_features=6, n_redundant=2, n_informative=4, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    DATASETS = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1), 
                linearly_separable ]

    figure = plt.figure(figsize=(27, 9))

    subplot_idx = 1
    M = len(CLASSIFIERS)
    for k, dataset in enumerate(DATASETS):
        t0 = time.time()
        apply_emsemble(subplot_idx, dataset )
        subplot_idx = subplot_idx + M + 1
        add_timestamp( "DATASET-%s" % k, time.time(), t0 )

    figure.subplots_adjust(left=.02, right=.98)
    plt.show()

    print "-"* 80
    ctimes  = [ (TIMESTAMPS[x][3]) for x in sorted(TIMESTAMPS) if "CLASSIFIER-TRAIN" in TIMESTAMPS[x][0]]
    ctimes_matrix = np.reshape( ctimes, (len(CLASSIFIERS),len(DATASETS)), order='F' )
    out = "%32s |\t %4s \t %4s \t %4s\n" % ( "CLASSIFIER", "MAKE_MOONS", "MAKE_CIRCLES", "LINEARLY_SEPARABLE" )
    for i,name in enumerate(NAMES):
        print "%24s \t" % name, "%.4f \t %.4f \t %.4f" % tuple(ctimes_matrix[i,range(len(DATASETS))])
    print "-"* 80

    if 0:
        for q in TIMESTAMPS.keys():
            print "%s \t %.4f \t %s" % ( q, TIMESTAMPS[q][3], TIMESTAMPS[q][0] )
        print "-"* 80

