#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pycluster example
@author: NRM
@url: http://gotohell
"""


# ####################################################################################
import re
import os
import sys
# ####################################################################################


# ####################################################################################
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import sklearn.neighbors as nn
from sklearn import preprocessing
from sklearn.utils import as_float_array
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import pydot
from sklearn.neighbors import KNeighborsRegressor
# ####################################################################################


# ####################################################################################
def introspection( obj ):
    return [ x for x in dir(obj) if "__" not in x ]
# ####################################################################################


# ####################################################################################
'''
from: http://scikit-learn.org/stable/modules/preprocessing.html
The preprocessing module further provides a utility class StandardScaler that implements 
the Transformer API to compute the mean and standard deviation on a training set so as 
to be able to later reapply the same transformation on the testing set. This class is 
hence suitable for use in the early steps of a sklearn.pipeline.Pipeline:
'''
# ####################################################################################
def get_scaler( X ):
    # scaler StandardScaler(copy=True, with_mean=True, with_std=True)
    X = as_float_array( X )
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler


# ####################################################################################
def transform_with_scaler( Y, scaler=None, wrt_X=[] ):
    Y = as_float_array( Y )
    if len(wrt_X) and not scaler:
        wrt_X = as_float_array( wrt_X )
        scaler = get_scaler( wrt_X )
    with_mean = scaler.mean_                                      
    with_stdv = scaler.std_                                       
    Z = scaler.transform(Y)                               
    return Z


# ####################################################################################
def print_fvec( fvec, f_format= "%.4f", v_format="%4d" ):
    features = [ f_format % x for x in fvec[:-1] ]
    value    = v_format % fvec[-1]
    print "%44s\t%s" % ( features, value )


# ####################################################################################
def fvec_loader( line, dval=0.0, debug=0 ):
    items = [ x.strip() for x in line.split() if x.strip() ]
    try:
        features = [ float(x.strip()) for x in items[:-1] ]
        value    = int(items[-1])
    except:
        print 'problem with tuple', line
        features = []
        for item in items[:-1]:
            if item.strip():
                features.append( float(item.strip()) )
            else:
                features.append( float(dval) )
        value = int(items[-1])
    features.append( value )
    if debug:
        print_fvec( features )
    return features


# ####################################################################################
def get_distance_metric(X, dtype='euclidean', normalize=True, nrows=10 ):
    X = as_float_array( X )
    dist = nn.DistanceMetric.get_metric('euclidean')
    d = dist.pairwise(X)
    print '-' * 80
    print introspection( dist )
    nrows=min(nrows,len(d))
    for row in d[:nrows]:
        print [ "%.2f" % x for x in row[:nrows] ]
    print '-' * 80
    return dist


# ####################################################################################
def knn_edgelist( nbrs ):
    vv = nbrs.kneighbors_graph(X).toarray()
    EL = {}
    for v1,row_i in enumerate(vv):
        print row_i
        print
        if not v1 in EL:
            EL[v1] = []
        for v2,col_j_val in enumerate(row_i):
            if col_j_val != 0:
                if v2 in EL[v1]:
                    print 'error, redudant node, something wrong'
                else:
                    EL[v1].append( v2 )
    return EL


# ####################################################################################
# kNN - o(M*N2) on N samples of M features when using brute force algorithm
#  @url: http://pythonhaven.wordpress.com/2009/12/09/generating_graphs_with_pydot
# ####################################################################################
def kNN(X, Y=[], k=2, algorithm="brute", radius=0.65, filename='graph-output.pdf', do_regression=True ):
    graph = pydot.Dot(graph_type='digraph')
    knn_model = NearestNeighbors(n_neighbors=k, algorithm=algorithm)
    nbrs = knn_model.fit(X)
    print '-' * 80
    indices = nbrs.kneighbors(X, 2, return_distance=False)
    radius_indices = nbrs.radius_neighbors(X, radius, return_distance=False)
    k_mapping = zip( X, indices, radius_indices )
    print '-' * 80

    nodes, misses, hits = {}, [], []
    for i, kmap in enumerate(k_mapping):
        sample = kmap[0]
        nneigh = kmap[1]
        rneigh = kmap[2]
        ypred  = Y[nneigh[1]]
        ytrue  = Y[i]
        nodes[i] = pydot.Node(str(i))
        if ypred != ytrue:
            misses.append(i)
        else:
            hits.append(i)

        for j in nneigh:
            if j not in nodes:
                nodes[j] = pydot.Node(str(j))
            if i == j:
                color = "black"
                if ytrue != ypred: color = "red"
                label = "%s: %s" % ( ypred, ytrue )
                graph.add_edge(pydot.Edge(str(i), str(j), label=label, labelfontcolor="#009933", fontsize="7.0", color=color))
            if i != j:
                color = "black"
                if ypred != Y[j]: color = "red"
                label = "%s: %s" % ( ypred, Y[j] )
                graph.add_edge(pydot.Edge(str(i), str(j), label=label, labelfontcolor="#009933", fontsize="7.0", color=color))
        print "%s : %s... \n\t%s:%s \n\t%s \n\t%s" % ( i, sample[:10], ypred, ytrue, nneigh, rneigh ) 
    print '-' * 80
    graph.write_pdf(filename)
    print "[%s] misses: %s" % ( len(misses), misses )
    print "[%s] hits:   %s" % ( len(hits), hits )

    if do_regression:
        neigh = KNeighborsRegressor(n_neighbors=2)
        neigh.fit(X, Y) 
        yp = neigh.predict(X+X*random.random()*.1)
        m = len(yp)
        mse = sum([ (x-y)*(x-y) for (x,y) in zip( Y, yp ) if x-y ]) / float(m)
        print "REGRESSION mean squared error: ", mse
        print "REGRESSION mse(i)!=0: ", [ ("%s:" % i, "%.5f" % ((x-y)*(x-y)/float(m))) for (i,x,y) in zip( range(m), Y, yp ) if x-y ]


    return nbrs, yp


# ####################################################################################
def load_dataset( filename, N=None, randomize=True ):
    with open( filename, "r" ) as fp:
        lines = fp.readlines()
        lines = [ line.strip() for line in lines if line.strip() ]
        fvecs = [ fvec_loader( line ) for line in lines ]
    m = len(fvecs[0])
    if randomize:
        random.shuffle(fvecs)
    fvecs = fvecs[:N]
    XY = np.array( fvecs )
    X = np.array( XY[:, range(m-1)])
    Y = np.array( XY[:, -1] )
    return XY, X, Y


# ####################################################################################

# ####################################################################################
def xyplot(X, Y):
    plt.figure(figsize=(8,6))
    colors = [ 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w' ] * 2
    coloring = [ colors[int(y)] for y in Y ]
    labeling = [ "%s" % int(y) for y in Y ]

    plt.scatter(X[:,0], X[:,1], color=coloring, label='class_label', alpha=0.5)

    # plt.title('XAlcohol and Malic Acid content of the wine dataset')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()



# ####################################################################################
def local_do_dbscan( X, eps=0.01, minsize=3 ):
    DB = DBSCAN(eps=eps, min_samples=minsize).fit(X)
    labels = DB.labels_
    labels_true = Y
    R = zip( labels, labels_true )
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
    return DB, n_clusters_, R


##############################################################################
# Plot result, black removed and is used for noise instead.
##############################################################################
def dbplot ( X, DB ):
    labels = DB.labels_
    core_samples_mask = np.zeros_like(DB.labels_, dtype=bool)
    core_samples_mask[DB.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    misses = 0
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            misses += 1

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    return misses


# ####################################################################################
def dbscan_accuracy(R):
    Q = {}
    for r in R:
        ypred, ytrue = r[0], r[1]
        if r in Q:
            Q[r] = Q[r] + 1
        else:
            Q[r] = 1
    for q in Q:
        print q, Q[q]

    valid, wrongs = {}, 0
    for r in Q:
        ypred = r[0] 
        if ypred in valid:
            if Q[r] > valid[ ypred ]:
                wrongs = wrongs + Q[r] - valid[ ypred ]
                valid[ ypred ] = Q[r]
            else:
                wrongs += Q[r]
        else:
            valid[ ypred ] = Q[r]
    for r in valid:
        print r, valid[r]

    qsum = sum([ Q[r] for r in Q ])
    return qsum, Q

    
# ####################################################################################
# TODO: implement adaptive stepsize with adaptive based on clusters with number of black points tradeoff and adaptive step
# ####################################################################################
def do_dbscan( X, Y, scale=True, minclusters=2, eps=0.667/2, minsize=2 ):
    if scale:
        X = StandardScaler().fit_transform(X)
    n = len(X)
    m = int(0.05 * n)
    nmin = int(n - 0.025 * n)

    # Compute DBSCAN
    nclusters = 0
    misses = 0
    qsum = 0
    # TODO: the iteration logic and accounting, is wrong here and there 
    while qsum<nmin and eps > 1E-6 and misses<m or nclusters < 2: 
        ( DB, nclusters, R ) = local_do_dbscan( X, eps=eps, minsize=minsize )
        print '-' * 80
        qsum, Q = dbscan_accuracy( R )
        print qsum
        print '-' * 80
        if nclusters >= 1:
            eps = eps/1.25
        misses = dbplot ( X, DB )


# ####################################################################################
def analysis( XY, X, Y  ):
    Xp = transform_with_scaler( X, wrt_X=X )
    Xp_distance_matrix = get_distance_metric(Xp, dtype='euclidean', normalize=True )
    # kNN(Xp, Y)
    kmodel, yp = kNN(Xp, Y, k=3, algorithm="auto", radius=0.65/2, filename='graph-output.pdf' )
    xyplot( Xp, Y )
    # do_dbscan( Xp, Y, minclusters=5, eps=1.0, minsize=10, scale=False )


# ####################################################################################
if __name__ == "__main__":
    XY, X, Y = load_dataset( 'Aggregation.txt', N=None)

    analysis( XY, X, Y )
    print len(X), len(Y)

