import numpy as np
import pylab as pl

from time import time
from copy import deepcopy

from sklearn import linear_model

# ################################################################################
# preprocessing
# ################################################################################
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

# ################################################################################
# cv
# ################################################################################
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ################################################################################
# roc
# ################################################################################
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

# ################################################################################
# svm
# ################################################################################
from sklearn import svm

# ################################################################################
# kmeans
# ################################################################################
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA






# ################################################################################
# datasets
# ################################################################################
from sklearn import datasets
from sklearn.datasets import load_digits

X = [[0, 0, 1, 2], [1, 1, 4, 1], [2, 2, 4, 3]]
Y = [0, 1, 1]

iris = datasets.load_iris()
X, Y = iris.data, iris.target

digits = datasets.load_digits()
X, Y = digits.data, digits.target


# ################################################################################
TAB    = "  "
LINE   = ""
HEADER = "*" * 80


# ################################################################################
def do_scale( X ):
    ''' scale a-la R
    '''
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
    Xs = scaler.transform(X)                               
    return ( Xs, scaler ) 


# ################################################################################
# multiplot of m/4 x 4 plots, one per binary classifier class on multiclass svm
# ################################################################################
def do_svm_roc( X, y, kernel="rbc" ):
    ''' plots a ROC for the given model
    '''
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(y, n_folds=6)
    classifier = svm.SVC(kernel=kernel, probability=True, random_state=0)

    def recode( y, K ):
        return [ 1 if t==K else -1 for t in y ] 
        # yt = y[:] # for i,t in enumerate(y): # if t != K: # yt[i] = -1 # return yt[:]

    mean_tpr = 0.0

    def plotroc( X, y, K ):
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for i, (train, test) in enumerate(cv):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            # fpr, tpr, thresholds = roc_curve(recode(y[test],K), probas_[:, 1], pos_label=K)
            fpr, tpr, thresholds = roc_curve(recode(y[test],K), probas_[:, 1], pos_label=-1)
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            # roc_auc = auc(fpr, tpr)
            # pl.plot(fpr, tpr, lw=1 )#, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        #mean_auc = auc(mean_fpr, mean_tpr)
        # pl.plot(mean_fpr, mean_tpr, 'k--') #, label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
        return ( mean_tpr, mean_fpr )

    mean_tpr = np.linspace(0, 1, 100)
    mean_fpr = np.linspace(0, 1, 100)
    mean_auc = 0.0
    for K in range(len(set(y))):
        ( tpr, fpr ) = plotroc( X, y, K )
        # mean_tpr = ((mean_tpr*K) + tpr)/float(K+1)
        # mean_fpr = ((mean_fpr*K) + fpr)/float(K+1)
        # mean_auc = ((mean_auc*K) + auc(fpr, tpr))/float(K+1)
        # mean_tpr += tpr
        # mean_fpr += fpr
        # mean_auc += auc(fpr,tpr)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr = mean_tpr/float(K+1)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    pl.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6) )# , label='Luck')
    plotroc( X, y, 7 )
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()

    return

    gmtpr = mean_tpr = 0.0
    gmfpr = mean_fpr = np.linspace(0, 1, 100)
    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for K in range(len(set(y[train]))):
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(recode(y[test],K), probas_[:, 1], pos_label=0)
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            # roc_auc = auc(fpr, tpr)
            pl.plot(fpr, tpr, lw=1 )
            gmtpr += ( mean_tpr * (len(y[train])-len([t for t in y[train] if t==K ]))/len(y[train]) )
    gmtpr /= len(cv)
    gmtpr[-1] = 1.0
    mean_auc = auc(mean_fpr, gmtpr)

    pl.plot([0, 1.1], [0, 1.1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    pl.plot(gmfpr, gmtpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()


# ################################################################################
def svm_plot( x, y, yp, eps=None):
    ''' plots a svm model (C or R) with its data mapped into 2D.
    '''
    cm = confusion_matrix( y, yp )

    # fig = pl.figure()
    fig = pl.figure(figsize=(8, 6))

    ax = fig.add_subplot(121)
    cax = ax.matshow(cm)
    pl.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    pl.xlabel('Predicted')
    pl.ylabel('True')

    ax = fig.add_subplot(122)
    if not eps: eps = float(abs(max(y)-min(y)+1))/float(len(set(y))*4.0)
    xx = [ np.sqrt(np.dot( row, row ))  for row in x ]   
    pl.scatter(xx, yp,       marker='o', color='r', label='y_predicted')
    pl.scatter(xx, y + eps,  marker='>', color='k', label='y_true')
    pl.hold('on')
    pl.xlabel('L2NORM(X)')
    pl.ylabel('PREDICTED SVM CLASS(Y)')
    pl.title('SUPPORT VECTOR CLASSIFICATION/REGRESSION')
    pl.legend()
    pl.show()


# ################################################################################
def do_confusion_matrix( y_test, y_pred, labels=[] ):
    ''' computes confusion matrix for the specified multiclass vectors;
        cross tabulation of predicted classes against training labels
    '''
    cm = confusion_matrix( y_test, y_pred )
    if labels: cm = confusion_matrix( y_test, y_pred, labels )
    return cm


# ################################################################################
def do_svm( X100, Y100, class_weight=None, C=4.0, 
            kernel='rbf', trainwith=0.6, nfolds=4, seed=0, mode="train",
            probs=True, regression=False, scale=True, debug=False ):
    ''' performs svm multiclass classification/regression of 
        X (n rows, p features) -> Y (n categorical/factor labels)
    '''

    t0 = time()

    # scaling
    X100s, scaler = X100, None
    if scale: ( X100s, scaler ) = do_scale( X100 )

    # partition testing vs. training sets
    Xt, Xp, Yt, Yp = cross_validation.train_test_split(X100s, Y100, test_size=1-trainwith, random_state=0)

    # train
    m = svm.SVC(kernel=kernel, C=C, class_weight=class_weight, probability=probs, verbose=debug )
    if regression: m = svm.SVR(kernel=kernel, C=C, class_weight=class_weight, probability=probs, verbose=debug )
    m.fit( Xt, Yt )
    training_accuracy = m.score( Xt, Yt )
    testing_accuracy  = m.score( Xp, Yp )

    #  cv 
    t1 = time()
    if nfolds < ( X.shape[0]/nfolds):
        cv_scores = cross_validation.cross_val_score(m, Xt, Yt, cv=nfolds)

    # test
    yp = m.predict( Xp )
    yp_probs = m.predict_proba( Xp )
    cm = do_confusion_matrix(Yp, yp)
    class_report = classification_report( Yp, yp )

    print LINE
    top = min(len(yp_probs),10)
    if len(yp_probs) - top: 
        print "FIRST %s FVECTORS OF %s MORE:" % ( top, len(yp_probs) - top )
    for i,x in enumerate( yp_probs[0:top] ):
        print "Fvec#: %3s, ClassProb: %s ----> P:%s vs T:%s" % ( i, str([ "%.2f" % y for y in x ]), yp[i], Yp[i] )
    print LINE

    # metrics
    t2 = time()
    print HEADER
    print "SVM:", m.get_params()
    print "NUMBER SUPPORT VECTORS: %s " % m.n_support_
    print "MULTICLASS SVM CLASSES: %s " % m.classes_
    print "SHAPE INPUT FVEC DATA : %s " % repr(X100.shape)
    print "SHAPE TRAINING SAMPLES: %s " % repr(Xt.shape)
    print "SHAPE TESTING  SAMPLES: %s " % repr(Xp.shape)
    print "TRAINING ACCUR. (%s) : %s " % ( trainwith, training_accuracy)
    print "TESTING ACCUR.  (%s) : %s " % ( 1-trainwith, testing_accuracy)
    print "REAL-TIME MODEL BLDNG : %s " % ( t1 - t0 )
    print "REAL-TIME CROSS VALID : %s " % ( t2 - t1 )
    print "CV %2s-FOLD SCORES     : %s " % ( nfolds, repr( [ "K%s: %.3f" % (i, v ) for i,v in enumerate(cv_scores) ] ) )
    print "CV %2s-FOLD SCORES     : %0.3f +/- %0.3f" % ( nfolds, np.mean( cv_scores ), np.std( cv_scores ) )
    print LINE
    print "CLASSIFICATION REPORT"
    print ( class_report )

    print "CONFUSION MATRIX"
    print cm
    print HEADER

    # plot in testing performance in 2D using L2-norm of X against true/predicted Y values 
    if scale:
        svm_plot( scaler.inverse_transform(Xp), Yp, yp )
    else:
        svm_plot( Xp, Yp, yp )

    # do_svm_roc( Xp, Yp, kernel=kernel )
    pl.figure(figsize=(8, 6))
    plot_subfigure(Xp, Yp, 1, "With unlabeled samples + CCA", "cca")
    plot_subfigure(Xp, Yp, 2, "With unlabeled samples + PCA", "pca")

    # pl.subplots_adjust(.04, .02, .97, .94, .09, .2)
    pl.show()




# ################################################################################

import numpy as np
import matplotlib.pylab as pl

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    pl.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        # Convert list of tuples to a class indicator matrix first
        Y_indicator = LabelBinarizer().fit(Y).transform(Y)
        X = CCA(n_components=2).fit(X, Y_indicator).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    pl.subplot(2, 2, subplot)
    pl.title(title)

    zero_class = np.where([0 in y for y in Y])
    one_class = np.where([1 in y for y in Y])
    pl.scatter(X[:, 0], X[:, 1], s=40, c='gray')
    pl.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
               facecolors='none', linewidths=2, label='Class 1')
    pl.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
               facecolors='none', linewidths=2, label='Class 2')

    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')
    pl.xticks(())
    pl.yticks(())

    pl.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    pl.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        pl.xlabel('First principal component')
        pl.ylabel('Second principal component')
        pl.legend(loc="upper left")



# ################################################################################
def do_cluster( X, Y ):
    pass


# ################################################################################
def do_glm( X, Y ): 
    pass


# ################################################################################
def describe( X=[] ):
    '''describe 2D variable X, intended to be a training or testing dataset'''
    print HEADER
    attrs, col_mu, col_sd, col_sum = (), (), (), ()
    if len(X):
        print "X:"

        Xp = np.asarray( X )
        if Xp.ndim == 2:
            nrows = Xp.shape[0]
            ncols = Xp.shape[1]

            print TAB, "%32s : %4s" % ( "NUMBER OF DIMENSIONS", Xp.ndim )
            print TAB, "%32s : %4s" % ( "NUMBER OF FEATURES",   ncols )
            print TAB, "%32s : %4s" % ( "NUMBER OF SAMPLES",    nrows )
            print TAB, "%32s : %4s" % ( "NUMBER OF ELEMENTS",   Xp.size )
            print

            Xt = np.transpose( Xp )
            col_mu  = map( np.mean, [ x for x in Xt ] ) 
            col_sd  = map( np.std, [ x for x in Xt ] ) 
            col_sum = map( sum, [ x for x in Xt ] ) 
            attrs = [ "x%s:" % str(i) for i in range(ncols) ]

            print TAB, "(X(i)-FEATURE: MEAN(X), STDDEV(X), SUM(X)):",
            f_stats = map( lambda a,x,y,z: ( a, x, y, z ), attrs, col_mu, col_sd, col_sum )
            for i,f in enumerate( f_stats ):
                if not i % 1: print TAB*2
                print TAB, "%-40s" % str(f), 
            print
            print

            print TAB, "(COVARIANCE MATRIX):"
            col_cov = np.cov( scale( np.ndarray.astype( Xt, float ) ) )
            # print repr( col_cov )
            print

    print HEADER
    ret = ( attrs, col_mu, col_sd, col_sum )
    return ret


# ################################################################################
def do_lm( X, Y, mode="train" ): 
    ret = ()
    if mode == "train":
        # train
        m = linear_model.LinearRegression()
        m.fit (X, Y )
        print m
        print "y = (b=%s) + (m=%s * x)" % ( m.coef_[0], m.coef_[1] )
    elif mode == "cv":
        # cross_validate
        pass
    elif mode == "predict":
        # predict
        pass
    else:
        print "unknown mode specified"
    return ret


# ################################################################################
def do_knn( X, Y ): 
    d = distance( X )
    XY = augment( X, Y )
    for x,y in map( lambda x,y: (x, y), X, Y ):
        xset = nearest( k, X ) 
        for w in xset:
            pass


# ################################################################################
def do_dbscan( X, Y ): 
    pass


# ################################################################################
def do_kmeans( data, Y=[] ): 
    n_samples, n_features = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target
    sample_size = 300
    print("n_digits: %d, \t n_samples %d, \t n_features %d" % (n_digits, n_samples, n_features))
    
    # ############################################################################
    def bench_k_means(estimator, name, data):
        t0 = time()
        estimator.fit(data)
        print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
              % (name, (time() - t0), estimator.inertia_,
                 metrics.homogeneity_score(labels, estimator.labels_),
                 metrics.completeness_score(labels, estimator.labels_),
                 metrics.v_measure_score(labels, estimator.labels_),
                 metrics.adjusted_rand_score(labels, estimator.labels_),
                 metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=sample_size)))
    # ############################################################################
    
    # ############################################################################
    def plot_kmeans( reduced_data ):
        ##########################################################################
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
        
        ##########################################################################
        # Plot the decision boundary. For that, we will assign a color to each
        ##########################################################################
        x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
        y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        ##########################################################################
        
        ##########################################################################
        # Obtain labels for each point in mesh. Use last trained model.
        ##########################################################################
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ##########################################################################
        # Put the result into a color plot
        ##########################################################################
        pl.figure(1)
        pl.clf()
        pl.imshow(Z, interpolation='nearest',
                  extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                  cmap=pl.cm.Paired,
                  aspect='auto', origin='lower')
        
        pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        pl.scatter(centroids[:, 0], centroids[:, 1],
                   marker='x', s=169, linewidths=3,
                   color='w', zorder=10)

        pl.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                 'Centroids are marked with white cross')

        pl.xlim(x_min, x_max)
        pl.ylim(y_min, y_max)
        pl.xticks(())
        pl.yticks(())
        pl.show()
    # ############################################################################

    print(HEADER)
    print( '% 9s' % 'init    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')
    print(HEADER)
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10), name="k-means++", data=data)
    bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10), name="random", data=data)
    print(HEADER)
    
    ###############################################################################
    # seed the centers deterministically with PCA SVD components, 
    # hence we run kmeans only once (n_init=1) using these seeds 
    # Then, visualize the results on the PCA-reduced 2D data space
    # Kmeans now with n_digits with 10 random starts against the transformed data
    ###############################################################################
    pca = PCA(n_components=n_digits).fit(data)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1), name="PCA-based", data=data)
    
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    if Y:
        centroids = kmeans.cluster_centers_
        Z = kmeans.predict(Y)
        print centroids
        print Z
    
    plot_kmeans( reduced_data )

    return kmeans


# ################################################################################
def do_tree( X, Y ): 
    pass


# ################################################################################
def driver(X, Y, model="lm"):
    describe( X )
    ret = do_lm( X, Y)
    ret = do_svm( X, Y )


# ################################################################################
if __name__ == "__main__":

    ret = driver(X, Y)


    '''
    np.random.seed(42)
    digits = load_digits()
    X = scale(digits.data)
    do_kmeans( X )
    '''








