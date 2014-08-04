#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pycluster based on: http://scikit-learn.org/stable/auto_examples/plot_classifier_comparison.html
@Code source: Gaël Varoquaux, Andreas Müller
@Modified for documentation by Jaques Grobler
@Modified by Nelson & Hobbes (i.e, scikit sources modified for local use)
@License: BSD 3 clause
"""
print(__doc__)


# ###################################################################################
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# ###################################################################################
import plot_roc
# ###################################################################################



# ###################################################################################
try:
    from sklearn.ensemble import BaggingClassifier
except:
    pass
# ###################################################################################


# ###################################################################################
NAMES = ["kNN K=5B",
         "kNN K=2A",
         "Linear SVM", 
         "RBF SVM", 
         "RBFSVMC=10", 
         "Decision Tree",
         "Random Forest", 
         "Rnd Forest Broad", 
         "AdaBoost", 
         "Naive Bayes", 
         "Dummy",
          #"Bagging(kNN)",
         "Gradient Boost",
         "LogistRegression"
         "LDA", 
         "QDA"]
# ###################################################################################
WHICH_VOTE = [
         "kNN K=5B",
         "RBF SVM", 
         "Rnd Forest Broad", 
         "Naive Bayes", 
         "Gradient Boost",
         "QDA"]
# ###################################################################################
CLASSIFIERS = [ KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='brute', metric='euclidean' ),
                KNeighborsClassifier(n_neighbors=2),
                SVC(kernel="linear", C=0.025, class_weight='auto'),
                SVC(gamma=2, C=1, class_weight='auto'),
                SVC(gamma=1, C=4, class_weight='auto'),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=1, verbose=0, min_density=None, compute_importances=None),
                AdaBoostClassifier(),
                GaussianNB(),
                DummyClassifier(strategy='stratified'),
                #BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5),
                GradientBoostingClassifier(loss='deviance', learning_rate=0.10, n_estimators=100, subsample=0.20, min_samples_split=2, min_samples_leaf=2, max_depth=3, init=None, random_state=1, max_features="auto", verbose=0),
                LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None),
                LDA(),
                QDA()]
# ###################################################################################
METRICS = {}
def add_metrics( cname, mname, mval ):
    global METRICS
    if cname not in METRICS:
        METRICS[ cname ] = {}
    if mname not in METRICS[cname]:
        METRICS[ cname ][ mname ] = [ mval, ]
    else:
        METRICS[ cname ][ mname ].append( mval )
# ###################################################################################




# ###################################################################################
TIMESTAMPS = {}
def add_timestamp( name, ts1, ts0 ):
    global TIMESTAMPS
    t = ts1 - ts0
    TIMESTAMPS[ len(TIMESTAMPS) ] =  ( name, ts1, ts0, t )
# ###################################################################################


# ###################################################################################
def do_scale( X ):
    X_scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
    Xs = X_scaler.transform(X)                               
    return ( Xs, X_scaler ) 
# ###################################################################################


# ###################################################################################
def fit_classifier( cname, myclf, Xt, ytrue, metrics=False): 
    # if "SVM" in cname: OneVsRestClassifier(myclf.fit(Xt, ytrue).predict(Xt))
    try:
        OneVsRestClassifier(myclf.fit(Xt, ytrue))
    except Exception as exception:
        print subplot_idx, cname, 'exception reported, fitting...', type(exception).__name__
        myclf.fit(Xt, ytrue)

    try:
        if metrics:
            if hasattr(myclf, "oob_score_"):
                print cname, "OUT-OF-BAG (AVG ACROSS ALL ITER) SCORE:", myclf.oob_score_
            if hasattr(myclf, "feature_importances_"):
                print cname, "ATTRIBUTE IMPORTANT (SHUFFLED VALUES):"
                for i,f in enumerate( myclf.feature_importances_ ):
                    print "X[%4d] : %s" % (i, f)
                print '-' * 80
    except Exception as exception:
        print subplot_idx, cname, 'exception reported, fitting(2)...', type(exception).__name__

    return myclf
# ###################################################################################


# ###################################################################################
def plot_binary_class_countour( subplot_idx, cname, CONTOURMESH, COLORMAPS, TRAIN_WITH, TEST_WITH, PERF_MSG ):
    [ cm, cm_bright ]    = COLORMAPS
    [ Z, xx, yy ]        = CONTOURMESH
    [ X_train, y_train ] = TRAIN_WITH
    [ X_test, y_test ]   = TEST_WITH
    try:
        # Put the result into a color plot; # Plot also the training points # and testing points
        ax = plt.subplot(len(DATASETS), len(CLASSIFIERS) + 1, subplot_idx)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        ax.scatter(X_test[:, 0],  X_test[:, 1],  c=y_test,  cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(cname)
        ax.text(xx.max() - .3, yy.min() + .3, PERF_MSG, size=13, horizontalalignment='right')
    except Exception as exception:
        print subplot_idx, cname, 'exception reported, plotting...', type(exception).__name__
    return
# ###################################################################################


# ###################################################################################
def plot_multiclass_basics(subplot_idx, cname, myclf, Xp, ytrue, eps=None, debug=False):
    yp = myclf.predict(Xp)
    cmat = confusion_matrix( ytrue, yp )
    if debug: print cname, cmat # [[34  1  2] [ 1 33  2] [ 0  1 26]]

    ax = plt.subplot(len(DATASETS), len(CLASSIFIERS) + 1, subplot_idx)
    cax = ax.matshow(cmat, cmap=plt.get_cmap("Reds"), interpolation='nearest' )
    minimal_ticks=[int(cmat.min()), int(cmat.mean()), int(cmat.max())]
    ticks_labels=[ "%s" % x for x in minimal_ticks ]
    plt.colorbar(cax, orientation='horizontal', ticks=minimal_ticks)
    # ax.set_xticklabels(ticks_labels)
    # plt.xlabel('PRED LABEL')
    # plt.ylabel('TRUE LABEL')
    # ax.set_title(cname)

    for i,row in enumerate(cmat): 
        for j,val in enumerate(row):
            plt.annotate(val, xy=(j, i), horizontalalignment='center', size=8, verticalalignment='center', color='blue')

    if not eps: eps = float(abs(max(yp)-min(yp)+1))/float(len(set(yp))*4.0)
    l2 = [ np.sqrt(np.dot( row, row ))  for row in Xp ]   
    ax.scatter(l2, yp,           marker='.', color='b', label='y_pred')
    ax.scatter(l2, ytrue + eps,  marker='+', color='k', label='y_true')
    plt.xlabel(cname)
    # plt.xlabel('L2NORM(X)')
    # plt.ylabel('PRED LABEL')
    # ax.set_title(cname)
    # plt.legend()
    # plt.hold('on')
# ###################################################################################


# ###################################################################################
def test_classifier( cname, myclf, Xp, ytrue, cvfolds=4, VOTER=None):              # StratifiedKFold
    if VOTER:
        yp = myclf.predict(Xp)
        VOTER.put_scores( cname, Xp, yp )

    try:
        scores = cross_val_score(myclf, Xp, ytrue, cv=cvfolds)
        sigma2 = 2.0 * scores.std() 
        score  = scores.mean()
    except Exception as exception:
        cvfolds = 1
        print subplot_idx, cname, 'exception reported, cross-validation...', type(exception).__name__
        score  = myclf.score(Xp, ytrue)
        sigma2 = 0
    add_metrics( cname, "STRAT%sKFOLD SCORE" % cvfolds, "%.3f+/-%.2f" % ( score, sigma2 ) )
    feedback = "CV [%s folds] ACCURACY: %0.2f (+/- %0.2f)" % (cvfolds, score, sigma2 )
    return (yp, score, sigma2, cvfolds, feedback) 
# ###################################################################################


# ###################################################################################
def unravel_decision_function(myclf, xx, yy):
    if hasattr(myclf, "decision_function"):
        Z = myclf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        # Returns the log-probabilities of the sample for each class in the model. 
        Z = myclf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    return ( Z, xx, yy )
# ###################################################################################


# ###################################################################################
def apply_feature_selection( X, y, k=2, dtype='classification', scoring_func=f_classif, debug=0 ):
    if debug:
        for i,x in enumerate(X):
            if sum( [ xi for xi in x if xi < 0.0 ]):
                print "%s \t %50s" % ( i, x )

    if dtype == 'classification':
        fSelector = SelectKBest(scoring_func, k=k)
        Xn = fSelector.fit_transform(X, y)

    print '-' * 80
    print "%6s \t %6s \t %8s" % ( "FEATURE", "F-SCORE", "P-VAL" )
    print '-' * 80
    print "ORIGINALLY: %s ---> TRANSFORMED INTO %s" % ( X.shape, Xn.shape )
    print '-' * 80

    X_feature_scores = zip(range(X.shape[1]), fSelector.scores_, fSelector.pvalues_ )
    X_feature_scores = sorted( X_feature_scores, key=lambda x: x[2] )
    for (f,v,p) in X_feature_scores:
        print "X[%3d] \t %8.2f \t %8.6f" % ( f, v, p )
    print '-' * 80

    return (fSelector, Xn, y)
# ###################################################################################


# ###################################################################################
# see whether training a voting classifier using binary features to predict the class given the votes
# it will also include feature selection to remove or handle the correlated voters.
# ###################################################################################
class VOTING( object ):
    def __init__( self, X, y, M, WHICH_VOTE, debug=0, method="minimal-consensus", csv="classifier_votes.csv" ):
        if method == "minimal-consensus":
            self.PIVOT = 2
        elif method == "majority":
            self.PIVOT = int(M/2)
        elif method == "significant-consensus":
            self.PIVOT = M-2
        else:
            self.PIVOT = 3
        self.N = len(y)
        self.M = M
        self.X = X
        self.y = y
        self.yp = []
        self.VOTES = {}
        self.VALID_VOTERS = WHICH_VOTE[:]
        self.debug=debug
        self.filename = csv
        return

    def makekey(self, xsample ):
        key = str(xsample)
        return key[:]

    def register( self, x, cname, vote ):
        key = self.makekey(x)
        if key not in self.VOTES:
            self.VOTES[key] = []
        self.VOTES[self.makekey(x)].append( vote )
        return

    def get_vote( self, x ):
        xrow = self.makekey( x )
        votes = self.VOTES[self.makekey(x)]
        vote_dict = {}
        for v in votes:
            if v not in vote_dict: vote_dict[ v ] = 0
            vote_dict[v] = vote_dict[v] + 1
        if self.debug:
            print xrow, "yields", vote_dict
        return ( vote_dict, votes )

    def get_agreement( self, x ):
        ( votes, voting_vector ) = self.get_vote( x ) 
        try:
            (nonzero_bias_bit, vbit) = max( [ (votes[v], v) for v in votes.keys() if v ] )
            if nonzero_bias_bit >= self.PIVOT:
                if self.debug: print vbit, "with", votes[vbit], "votes selected", votes
                return (vbit, nonzero_bias_bit, votes, voting_vector )
            else:
                if self.debug: print 0, "with", votes[0], "votes selected", votes
                return (0, votes[0], votes, voting_vector )
        except ValueError:
            if self.debug: print 0, "with", votes[0], "votes selected", votes
            return (0, votes[0], votes, voting_vector )

    def put_scores( self, cname, X, y ):
        if cname not in self.VALID_VOTERS:
            return "No votes registered"
        for i, x in enumerate(X):
            self.register( x, cname, y[i] )
        return "%d votes possibly registered" % len(X)

    def get_scores( self, Xt, Yt, method="minimal-consensus" ):
        tp, tn = 0, 0
        fp, fn = 0, 0
        hits, misses = 0, 0
        self.yp = np.zeros(len(Yt))

        #  only valid for binary class
        header = "-" * 80 + '\n'
        if self.filename: fptr = open( self.filename, "w" )
        fptr.write( header )
        for i, xt in enumerate(Xt):
            (ypred, vsum, votes, voting_vector ) =  self.get_agreement( xt )
            self.yp[i] = int(ypred)
            if self.yp[i] == Yt[i] and Yt[i]:
                tp += 1
                hits += 1
            if self.yp[i] == Yt[i] and not Yt[i]:
                tn += 1
                hits += 1
            if self.yp[i] != Yt[i] and Yt[i]:
                fn += 1
                misses += 1
            if self.yp[i] != Yt[i] and not Yt[i]:
                fp += 1
                misses += 1
            out = "%5s \t %s \t %s \t %s \t %s \t %s\n" % ( i, ypred, vsum, voting_vector, xt, votes )
            fptr.write( out )
        fptr.write( header )
        fptr.close()
        accuracy = float(tp+tn)/len(self.yp)

        cmat = confusion_matrix( Yt, self.yp )
        if self.filename: 
            with open( self.filename, "a" ) as fp:
                fp.write( header )
                for i,row in enumerate(cmat):
                    out = "%s:\t %s\n" % ( i, row )
                    fp.write( out )
                fp.write( header )
                out = "%s\n" % [(x-y, x, y) for (x,y) in zip( self.yp, Yt )]
                fp.write( out )
                fp.write( header )

        return (accuracy, cmat[:], self.yp[:])
# ###################################################################################


# ###################################################################################
def basic_mesh_plot(X, X_train, y_train, X_test, y_test, h=0.02):
    # plot the dataset first; the training points and testing points
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) # h = .02;  step size in the mesh
    ax = plt.subplot(len(DATASETS), len(CLASSIFIERS) + 1, subplot_idx)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    ax.scatter(X_test[:, 0],  X_test[:, 1],  c=y_test,  cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    return xx, yy
# ###################################################################################


# ###################################################################################
def split_dataset( X, y, split_ratio=0.4 ):
    # split into training and test part
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio )
    y_labels = determine_num_ylabels(y)
    return ( X_train, y_train, X_test, y_test, y_labels )
# ###################################################################################


# ###################################################################################
def determine_num_ylabels(Y):
    y_labels = {} 
    for i,y in enumerate(Y):
        if y not in y_labels:
            y_labels[y] = []
        y_labels[y].append( i )
    print "NUM LABELS = %s\t" % len(y_labels)
    for i in y_labels:
        print "LABEL[%s] = %d samples" % ( i, len(y_labels[i]) )
    print '-' * 80
    return y_labels
# ###################################################################################


# ###################################################################################
def compute_roc( clf, X_train, y_train, X_test, y_test, y_pred=[] ):
    return 
# ###################################################################################


# ###################################################################################
# this is NOT a emsemble at this time, it is the skeleton to later become one, as for example, 
# why the test is pulled away on its own so individual votes could be picked and an arbitrary 
# emsemble generator could be used, with the caching of the bits
def apply_emsemble( subplot_idx, dataset, h=0.02 ):
    X, y = dataset

    X = StandardScaler().fit_transform(X)

    ( X_train, y_train, X_test, y_test, y_labels ) = split_dataset( X, y )

    VOTER = VOTING( X, y, len(CLASSIFIERS), WHICH_VOTE, debug=0, csv="voting_classifier_details_%s.csv" % subplot_idx )

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    (xx, yy) = basic_mesh_plot( X, X_train, y_train, X_test, y_test, h=h )

    # pp = PdfPages('foo.pdf')

    subplot_idx += 1
    for cname, clf in zip(NAMES, CLASSIFIERS):
        t0 = time.time()

        try:
            trained_clf = fit_classifier( cname, clf, X_train, y_train )
            (yp, cscore, sigma, cvfolds,feedback) = test_classifier( cname, trained_clf, X_test, y_test, VOTER=VOTER ) 
            print "%2d %24s \t ... \t %s" % ( subplot_idx, cname, feedback )
        except Exception as exception:
            print subplot_idx, cname, 'exception reported, train/test...', type(exception).__name__

        add_timestamp( "CLASSIFIER-TRAIN/TEST[%s,%s]" % (cname, subplot_idx), time.time(), t0 )
        t0 = time.time()

        try:
            plt.figure(2)           # make figure 2 current
            roc_fig = plot_roc.roc( X, y, clf, n=len(y_labels.keys()), cname=cname, idx=subplot_idx)
            # if roc_fig: plt.savefig(roc_fig, bbox_inches='tight')
        except Exception as exception:
            # print subplot_idx, cname, 'exception reported, roc...', type(exception).__name__
            pass
        finally:
            plt.figure(1)           # make figure 1 current

        add_timestamp( "CLASSIFIER-ROC[%s,%s]" % (cname, subplot_idx), time.time(), t0 )
        t0 = time.time()

        N,M = X_train.shape
        if M<=2:
            # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, m_max]x[y_min, y_max].
            (Z, xx, yy ) = unravel_decision_function(trained_clf, xx, yy)
            Z = Z.reshape(xx.shape)
            COLORMAPS   = [ cm, cm_bright ]
            CONTOURMESH = [ Z, xx, yy ]
            TRAIN_WITH  = [ X_train, y_train ]
            TEST_WITH   = [ X_test, y_test ]
            PERF_MSG    = ('%.2f' % cscore).lstrip('0') + '+/-' + ('%.2f' % sigma).lstrip('0')
            plot_binary_class_countour( subplot_idx, cname, CONTOURMESH, COLORMAPS, TRAIN_WITH, TEST_WITH, PERF_MSG )
        else:
            # plot instead simpler confusion matrix and/or plot for class performance
            plot_multiclass_basics(subplot_idx, cname, trained_clf, X_test, y_test )
    
        add_timestamp( "CLASSIFIER-PLOT[%s,%s]" % (cname, subplot_idx), time.time(), t0 )
        subplot_idx += 1

    # pp.close()
    print '-' * 80
    accuracy, cmat, boosted_yp = VOTER.get_scores( X_test, y_test )
    print "VOTING EMSEMBLE ACCURACY (NOT A CLASSIFIER!, COMPARISON PURPOSES ONLY!)", accuracy
    print "CONFUSION MATRIX:"
    def confusion_matrix_line( row ):
        m = len(row)
        outstr = "%6d\t" * m 
        out = outstr % tuple(row)
        return out
    for i, row in enumerate(cmat):
        print "%s: \t %s" % ( i, confusion_matrix_line(row) )
    print '-' * 80

    return subplot_idx
# ###################################################################################



# ###################################################################################
def do_post_analytics( debug=0 ):
    print "-"* 80
    ctimes  = [ (TIMESTAMPS[x][3]) for x in sorted(TIMESTAMPS) if "CLASSIFIER-TRAIN" in TIMESTAMPS[x][0]]
    ctimes_matrix = np.reshape( ctimes, (len(CLASSIFIERS),len(DATASETS)), order='F' )
    print "%24s |\t %4s \t %4s \t %4s" % ( "CLASSIFIER-PERFORMANCE", "MOONS", "CIRCLES", "LIN_SEPARABLE" )
    print "-"* 80
    for i,name in enumerate(NAMES):
        print "%24s \t" % name, "%.4f \t %.4f \t %.4f" % tuple(ctimes_matrix[i,range(len(DATASETS))])

    if debug: 
        print "-"* 80
        for q in TIMESTAMPS.keys():
            print "%s \t %.4f \t %s" % ( q, TIMESTAMPS[q][3], TIMESTAMPS[q][0] )
        print "-"* 80

    print "-"* 80
    print "%2s:%2s \t %24s \t %24s \t %16s" % ( "--", "--", "CLASSIFIER NAME", "METRICS NAME", "DATASET's ESTIMATE +/- 2*SIGMA" )
    print "-"* 80
    for i, c in enumerate(sorted(METRICS.keys())):
        for j, q in enumerate(sorted(METRICS[c].keys())):
            v=" "
            if c in WHICH_VOTE: v="*"
            print "%2d:%2d \t %24s \t %24s \t %16s" % ( i, j, c+v, q, METRICS[c][q] )
    print "-"* 80
# ###################################################################################


# ###################################################################################
def do_dataset_generation( NumSamples, NumClasses=2, NumFeatures=2, multiclass=False, apply_stdscaler=True ):
    # n_classes * n_clusters_per_class mustbe smaller or equal 2 ** n_informative
    if multiclass:
        from sklearn import datasets
        iris, digits = datasets.load_iris(), datasets.load_digits()
        X1, y1 = iris.data, iris.target
        X2, y2 = digits.data, digits.target
    else:
        X1, y1 = make_moons(n_samples=NumSamples, noise=0.3, random_state=0)
        X2, y2 = make_circles(n_samples=NumSamples, factor=0.5, noise=0.2, random_state=1)

    X3, y3 = make_classification(n_samples=NumSamples, 
                                 n_classes=NumClasses, 
                                 # weights=[0.6,0.4],
                                 n_features=NumFeatures+6, 
                                 # n_redundant=NumFeatures*1,
                                 n_informative=NumFeatures+1,
                                 n_clusters_per_class=3,
                                 random_state=0,
                                 flip_y=0.01 )

    if apply_stdscaler:
        (X1, x1_scaler ) = do_scale( X1 )
        (X2, x2_scaler ) = do_scale( X2 )
        (X3, x2_scaler ) = do_scale( X3 )

    fsel1, X1n, y1 = apply_feature_selection( X1, y1, k=2, dtype='classification', scoring_func=f_classif )
    fsel2, X2n, y2 = apply_feature_selection( X2, y2, k=2, dtype='classification', scoring_func=f_classif )
    fsel3, X3n, y3 = apply_feature_selection( X3, y3, k=NumFeatures, dtype='classification', scoring_func=f_classif )

    return [ (X1n, y1), (X2n, y2), (X3n, y3) ]
# ###################################################################################


# ###################################################################################
if __name__ == "__main__":

    NumSamples, NumClasses, NumFeatures, multiclass = 1000, 3, 4, False

    DATASETS = do_dataset_generation( NumSamples, NumClasses, NumFeatures, multiclass )

    figure = plt.figure(1,figsize=(27, 12))
    subplot_idx = 1

    for k, dataset in enumerate(DATASETS):
        t0 = time.time()

        apply_emsemble(subplot_idx, dataset )

        subplot_idx = subplot_idx + len(CLASSIFIERS) + 1

        add_timestamp( "DATASET-%s" % k, time.time(), t0 )

    figure.subplots_adjust(left=.02, right=.98)
    figure.show()
    figure.savefig('comparative-performance.png')

    do_post_analytics()

    print '-' * 80
    print 'please lookup file comparative-performance.png for the comparison plot of these classifiers'
    print 'please lookup files voting_classifier_details_*.csv for details of the current emsemble voter engine'
    print '-' * 80


