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
import re
import time
import numpy as np
import pandas
import math
import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from matplotlib import mlab
from sklearn.linear_model import Ridge
from numpy import corrcoef, sum, log, arange
# ###################################################################################
import plot_roc
# ###################################################################################
try:
    from sklearn.ensemble import BaggingClassifier
except:
    pass
# ###################################################################################


# ###################################################################################
FHEADER_REGEX = re.compile( '(\d+).\s+?(\w.*)?:\s+(.*$)' )
FEATURE_DESCRIPTION_HEADER = '''
0   mislabeled_spam_or_not: 1, 0.    | spam, non-spam classes
1   word_freq_make:         continuous.
2   word_freq_address:      continuous.
3   word_freq_all:          continuous.
4   word_freq_3d:           continuous.
5   word_freq_our:          continuous.
6   word_freq_over:         continuous.
7   word_freq_remove:       continuous.
8   word_freq_internet:     continuous.
9   word_freq_order:        continuous.
10  word_freq_mail:         continuous.
11  word_freq_receive:      continuous.
12  word_freq_will:         continuous.
13  word_freq_people:       continuous.
14  word_freq_report:       continuous.
15  word_freq_addresses:    continuous.
16  word_freq_free:         continuous.
17  word_freq_business:     continuous.
18  word_freq_email:        continuous.
19  word_freq_you:          continuous.
20  word_freq_credit:       continuous.
21  word_freq_your:         continuous.
22  word_freq_font:         continuous.
23  word_freq_000:          continuous.
24  word_freq_money:        continuous.
25  word_freq_hp:           continuous.
26  word_freq_hpl:          continuous.
27  word_freq_george:       continuous.
28  word_freq_650:          continuous.
29  word_freq_lab:          continuous.
30  word_freq_labs:         continuous.
31  word_freq_telnet:       continuous.
32  word_freq_857:          continuous.
33  word_freq_data:         continuous.
34  word_freq_415:          continuous.
35  word_freq_85:           continuous.
36  word_freq_technology:   continuous.
37  word_freq_1999:         continuous.
38  word_freq_parts:        continuous.
39  word_freq_pm:           continuous.
40  word_freq_direct:       continuous.
41  word_freq_cs:           continuous.
42  word_freq_meeting:      continuous.
43  word_freq_original:     continuous.
44  word_freq_project:      continuous.
45  word_freq_re:           continuous.
46  word_freq_edu:          continuous.
47  word_freq_table:        continuous.
48  word_freq_conference:   continuous.
49  char_freq_;:            continuous.
50  char_freq_(:            continuous.
51  char_freq_[:            continuous.
52  char_freq_!:            continuous.
53  char_freq_$:            continuous.
54  char_freq_#:            continuous.
55  capital_run_length_average: continuous.
56  capital_run_length_longest: continuous.
57  capital_run_length_total:   continuous.
    '''
# ###################################################################################
HEADER = "-" * 80
NAN    = np.nan
# ###################################################################################


# ##########################################################################################
global FEATURES
global FEATURES_MAPPING
FEATURES = {} 
FEATURES_MAPPING = {} 
# ##########################################################################################


# ###################################################################################
CLASSIFIER_NAMES = ["kNN K=5B",
                    # "kNN K=2A",
                    "Linear SVM",
                    #"RBFSVMC=1",
                    "RBFSVMC=10",
                    "Decision Tree",
		    "RndForest Broad",
		    "AdaBoost",
		    "Naive Bayes",
		    "Gradient Boost",
		    "LogistRegr1",
		    "LDA",
		    # , "Dummy",
		    # , "Bagging(kNN)",
                    "QDA",
                    ]
# ###################################################################################
WHICH_VOTE = [      "kNN K=5B",
                    "RBFSVMC=10", 
                    "Rnd Forest Broad", 
                    "Naive Bayes", 
                    "Gradient Boost",
             ]
# ###################################################################################
CLASSIFIER_MODELS =  [ KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='brute', metric='euclidean' ),
                       # KNeighborsClassifier(n_neighbors=2),
                       SVC(kernel="linear", C=0.025, class_weight='auto'),
                       # SVC(gamma=.1, C=1, class_weight='auto'),
                       SVC(gamma=1, C=10, class_weight='auto'),
                       DecisionTreeClassifier(max_depth=5),
                       RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=1, verbose=0, min_density=None, compute_importances=None),
                       AdaBoostClassifier(),
                       GaussianNB(),
                       GradientBoostingClassifier(loss='deviance', learning_rate=0.10, n_estimators=100, subsample=0.20, min_samples_split=2, min_samples_leaf=2, max_depth=3, init=None, random_state=1, max_features="auto", verbose=0),
                       LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=1, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None),
                       LDA(),
                       # DummyClassifier(strategy='stratified'),
                       # BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5),
                       QDA(),
                       ]
# ###################################################################################


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
def decode_factor( factor_vector ):
    vlabels   = set( [ x for x in factor_vector ] )
    invlabels = set( [ x for x in factor_vector if x not in vlabels ] )
    print vlabels
    assert  len(invlabels) == 0, 'decoding error on factor, NaN/Missing?' 
    vencoder = dict( zip(range(len(vlabels)), vlabels) )
    vdecoder = dict( zip(vlabels, range(len(vlabels))) )
    encoded_vector = np.array([ vdecoder[ vlabel ] for vlabel in factor_vector ])
    return ( vlabels, vencoder, vdecoder, encoded_vector )
# ###################################################################################


# ###################################################################################
def encode_factor( encoded_vector, original_vector=[], vencoder=[] ):
    assert len(vencoder) > 0, "factor decoder is required"
    factor_vector = [ vencoder[ item ] for item in encoded_vector ]
    return factor_vector
# ###################################################################################


# ##########################################################################################
def describe_data( XY, Y=[], dataset_num="", heading="", debug=False ):
    print HEADER
    if heading: print " MATRIX: %s \t SHAPE: %s \t" % ( heading, XY.shape )
    print XY.dtypes
    if debug:
        print XY.describe()
        print
    print HEADER
    return
# ##########################################################################################


# ##########################################################################################
# from: http://blog.marmakoide.org/?p=94
# ##########################################################################################
def do_feature_descriptions( XY, Y=[], dataset_name="", nmax=256, fmax=100, which_cols=[], rmax=0.81, debug=False ):
    pdf_pages = PdfPages('feature_description_booklet_%s.pdf' % dataset_name)
    nmax = min( nmax, len(Y) )
    fmax = min( fmax, XY.shape[1] )

    nb_plots_per_page = 4
    nb_plots  = fmax
    nb_pages  = int(np.ceil(nb_plots / float(nb_plots_per_page))) + 1
    print "GENERATING: feature_description_booklet.pdf [%s] pages" % nb_pages

    GRID_SIZE = (nb_plots_per_page, 4)
    
    if not len(which_cols):
        which_cols = [x for x in XY]

    which_cols = which_cols[:fmax]

    def ival( f ):
        return "%.1f" % f

    for i, c in enumerate(which_cols):
        samples = get_np_col( XY, c ) # [yy for yy in XY[c]]
        which_samples = np.random.random_integers(0, high=len(Y)-1, size=nmax)
        x       = np.array( samples )
        y       = np.array( Y )
        xx      = np.array([x[k] for k in which_samples])
        yy      = np.array([y[k] for k in which_samples])
        mu      = x.mean()
        sigma   = x.std()
        xfreq = {}
        for a in x:
            aa = ival(a)
            if aa not in xfreq: xfreq[aa] = 0
            xfreq[aa] = xfreq[aa] + 1

        if i % nb_plots_per_page == 0:
            fig = plt.figure(9,figsize=(11.69, 8.27), dpi=72)

        print '.',
        ax0 = plt.subplot2grid(GRID_SIZE, (i % nb_plots_per_page, 0), rowspan=1, colspan=1)
        ax0.set_xlabel("X%s: %s" % ( c, get_fname( USECOLS[c] ) ), fontsize=8 )
        xx_buckets = np.array([ int(a*10) for a in xx ]) #if a and xfreq[ival(a)]>4 ])
        cmat = confusion_matrix( xx_buckets, yy )
        cax0 = ax0.matshow(cmat, cmap=plt.get_cmap("Reds"), interpolation='nearest' )
        minimal_ticks=[int(cmat.min()), int(cmat.mean()), int(cmat.max())]
        ticks_labels=[ "%s" % x for x in minimal_ticks ]
        plt.colorbar(cax0, orientation='horizontal', ticks=minimal_ticks )

        print '.',
        ax1 = plt.subplot2grid(GRID_SIZE, (i % nb_plots_per_page, 1), rowspan=1, colspan=1)
        yr = [ yval + (0.25 - np.random.random()) * 0.5 for yval in yy ]
        ax1.scatter(xx, yr,  color=['r' if a else 'b' for a in yy], marker='s', s=[math.sqrt(xfreq[ival(a)]+1) for a in xx])
        ax1.set_ylabel("Y=X%s: %s" % ( PREDICT_COL, get_fname( PREDICT_COL )[0:10] ), fontsize=8 )

        print '. ',
        ax2 = plt.subplot2grid(GRID_SIZE, (i % nb_plots_per_page, 2), rowspan=1, colspan=1)
        ax2.text(0,1, r'$\mu=%.2f,\ \sigma=%.2f$' % ( mu, sigma ) )
        nb, bins, patches = ax2.hist(xx, 32, normed=1, facecolor='#808080', alpha=0.75)
        # nb, bins, patches = ax2.hist(xx, 32, normed=1, facecolor='green', alpha=0.60)
        # plt.plot(bins, mlab.normpdf( bins, mu, sigma), 'r--', linewidth=2)
        ax2.set_xlabel('X-Value', fontsize=8 )
        ax2.set_ylabel('Frequency|Probability', fontsize=8)

        print '. ',
        ax3 = plt.subplot2grid(GRID_SIZE, (i % nb_plots_per_page, 3), rowspan=1, colspan=1)
        k = sorted(xfreq.keys())
        ax3.plot([log(xfreq[kk]+1) if xfreq[kk]+1 != 0 else 0 for kk in k] )
        ax3.set_ylabel('Log(Frequency)', fontsize=8)
        ax3.set_xlabel('10 * X-Value', fontsize=8 )

        # Close the page if needed
        if (i + 1) % nb_plots_per_page == 0 or (i + 1) == nb_plots:
            plt.tight_layout()
            pdf_pages.savefig(fig)

    # from: http://glowingpython.blogspot.com/2012/10/visualizing-correlation-matrices.html
    fig = plt.figure(8,figsize=(11.69, 8.27), dpi=72)
    R = compute_correlation_matrix( XY, Y, which_cols=which_cols, nmax=nmax )
    plt.pcolor(R)
    plt.colorbar()
    for i,c in enumerate(which_cols):
        plt.text( 0.33, i+0.33, "X%s:%s" % (c, get_fname(USECOLS[c])), fontsize=8 )
    pdf_pages.savefig(fig)
    pdf_pages.close()

    correlated_cols = highly_patentable_determine_in_a_very_basic_way_almost_stupidly_which_features_are_correlated( R, which_cols, rmax=rmax)

    return correlated_cols
# ##########################################################################################


# ##########################################################################################
def compute_correlation_matrix( XY, Y, which_cols=[], nmax=128):
    which_samples = np.random.random_integers(0, high=len(Y)-1, size=nmax)
    MM = XY[tuple(which_samples),:]
    MM = MM[:,tuple(which_cols)]
    R = corrcoef(np.transpose(MM))
    return R
# ##########################################################################################


# ##########################################################################################
# a most basic correlation analysis examining for multicolinearity among variables as done in 
# standard Design of Experiments (DOE) textbooks. Highly correlated variables reduce predictive 
# power and numerical stability - as done in DOE textbooks
# ##########################################################################################
def highly_patentable_determine_in_a_very_basic_way_almost_stupidly_which_features_are_correlated( R, which_cols, rmax=0.81, debug=False ):
    colinear_tagged  = {}
    print
    print HEADER
    for i, ci in enumerate(which_cols):
        if ci == PREDICT_COL: continue
        for j, cj in enumerate(which_cols):
            if j<=i or cj == PREDICT_COL: continue
            R_ij = abs( R[i,j] )
            if debug: print "X:%2s X:%2s \t\t %.3f \t %30s \t %30s" % (i, j, R_ij, get_fname(USECOLS[ci])[:10], get_fname(USECOLS[cj])[:10])
            if R_ij >= rmax:
                if ci not in colinear_tagged: colinear_tagged[ci] = []
                colinear_tagged[ci].append( cj )
                print i, j, ci, cj
                print "<<< X:%2s -- X:%2s \t\t %.3f \t %30s \t %30s" % (i, j, R_ij, get_fname(USECOLS[ci]), get_fname(USECOLS[cj]))
                print colinear_tagged[ci]

    for ck in colinear_tagged.keys():
        print ck, get_fname(USECOLS[ck]), 
        print "\t>>>", set(colinear_tagged[ck]), [get_fname(USECOLS[cj]) for cj in set(colinear_tagged[ck])]
    print

    correlated_cols = []
    for ci in colinear_tagged.keys():
        if ci == PREDICT_COL: continue
        cols = set(colinear_tagged[ci])
        if PREDICT_COL in cols: continue
        correlated_cols.extend( cols )

    correlated_cols = sorted( set( correlated_cols ) )

    return correlated_cols 
# ##########################################################################################


# ###################################################################################
def get_df_column( X, idx ):
    assert len(X[X.columns[0]]) == len(X[idx]), "BASIC PROBLEM WITH DATAFRAME STRUCTURE"
    Xi = X[idx]
    return Xi
# ###################################################################################


# ##########################################################################################
def get_np_row( x, j ):
    col = [ item for item in x[j,:]]
    return col
# ##########################################################################################


# ##########################################################################################
def get_np_slice_rows( x, indices ):
    return x[tuple(indices),:]
# ##########################################################################################


# ##########################################################################################
def get_np_slice_cols( x, indices ):
    return x[:,tuple(indices)]
# ##########################################################################################


# ##########################################################################################
def get_np_col( x, j ):
    col = [ item for item in x[:,j]]
    return col
# ##########################################################################################


# ###################################################################################
def add_df_column_to( X, data_column, debug=1 ):
    assert len(X[X.columns[0]]) == len(data_column), "COLUMNS ARE NOT EQUAL IN SIZE"
    idx_where_to = max([x for x in X.columns ]) + 1
    print "BEFORE:", X.columns
    X[idx_where_to] = data_column
    print "AFTER: ", X.columns
    return (X, idx_where_to)
# ###################################################################################


# ###################################################################################
def safe_op( op, vector_x, repval=np.NaN ):
    x = []
    for x_i in vector_x:
        try:
            if op == math.sqrt: 
                val = op( float(x_i) )
            elif op == math.log:  
                val = op( float(x_i) )
            else: 
                raise Exception
        except:
            val = repval
        x.append( val )
    return x[:]
# ###################################################################################
    

# ###################################################################################
def apply_scalar_op_to( op, a, b ):
    try:
        c = eval("float(%s).__%s__(float(%s))" %  ( a, op, b ))
    except:
        c = np.NaN
    return c
# ###################################################################################


# ###################################################################################
def get_fname( idx ):
    assert idx in FEATURES, "PROBLEM WITH NAMING OF FEATURES"
    if idx in FEATURES: fname = FEATURES[ idx ][1] 
    return fname
def get_fnum( fname ):
    if fname in FEATURES: fidx = FEATURES_MAPPING[ fname ][1] 
    return fidx
# ###################################################################################


# ###################################################################################
def generate_feature( X, i, j=None, how="interaction", OP="mul" ):
    global FEATURES
    global FEATURES_MAPPING

    what = ""

    Xcolumn_i = get_df_column( X, i )
    if j: Xcolumn_j = get_df_column( X, j )

    if "sqrt" in how:
        Xcolumn_i = safe_op(math.sqrt, Xcolumn_i)
        if j: Xcolumn_j = safe_op(math.sqrt, Xcolumn_j)
        what+="sqrt(%s)" % (get_fname(i))
    if "log" in how:
        Xcolumn_i = safe_op(math.log, Xcolumn_i)
        if j: Xcolumn_j = safe_op(math.log, Xcolumn_j)
        what+="log(%s)" % (get_fname(i))
    if "interaction" in how:
        if j: 
            Xcolumn_i = [ apply_scalar_op_to(OP, xi, xj) for (xi,xj) in zip(Xcolumn_i, Xcolumn_j) ]
            what+="%s(%s,%s)" % (OP, get_fname(i), get_fname(j))
        else:
            Xcolumn_i = [ apply_scalar_op_to(OP, xi, xj) for (xi,xj) in zip(Xcolumn_i, Xcolumn_i) ]
            what+="%s(%s,%s)" % (OP, get_fname(i), get_fname(i))
    if "divide" in how:
        Xcolumn_i = [ apply_scalar_op_to("div",  xi, xj) for (xi,xj) in zip(Xcolumn_i, Xcolumn_j) ]


    (Xn, idx) = add_df_column_to( X, Xcolumn_i )

    # generated features have index equal to the tuple's index
    FEATURES[ idx ] = [ str(idx), what[:], how[:] ]
    FEATURES_MAPPING = dict([ get_ftuple_from( key, retval="mapping" ) for key in FEATURES.keys() ])
    print "*** GENERATED FEATURE %s : %s" % ( idx , FEATURES[ idx ] )

    return (Xn, idx )
# ###################################################################################


# ###################################################################################
'''
print "%3s %3s" % (idx, i), "\t %6s \t %42s \t %48s" % tuple( FEATURES[i] )
WILL BE ANALYZING USING THESE FEATURES*: (*pandas columns starts at 0)
  0   9          10                                      wheel-base                           continuous from 86.6 120.9.
  1  *10          11                                          length                       continuous from 141.1 to 208.1.
  2  11          12                                           width                         continuous from 60.3 to 72.3.
  3  12          13                                          height                         continuous from 47.8 to 59.8.
  4  *13          14                                     curb-weight                         continuous from 1488 to 4066.
 14  63          63                         div(curb-weight,length)                                           interaction
(XY, idx ) = generate_feature( XY, 13, j=10, how="interaction", OP="div" )
'''
# ###################################################################################
def get_ftuple_from( pandas_idx, retval="mapping" ):
    ( origfile_fidx, origfile_fname, origfile_fdesc ) =  FEATURES[ pandas_idx ]
    fmapping = origfile_fname[:], pandas_idx
    ftuple   = ( pandas_idx, origfile_fidx, origfile_fname, origfile_fdesc ) 
    if "mapping" in retval: return fmapping
    return ftuple
# ###################################################################################


# ###################################################################################
def lookup_feature_by_name( fname ):
    pandas_fidx = None
    FEATURES_MAPPING = dict([ get_ftuple_from( key, retval="mapping" ) for key in FEATURES.keys() ])
    if fname in FEATURES_MAPPING:
        pandas_fidx = FEATURES_MAPPING[ fname ]
        print "FEATURE NAME: %24s \t --> \t index: %4s" % ( fname, pandas_fidx )
    return pandas_fidx
# ###################################################################################


# ###################################################################################
def generate_interaction( XY, fn1="", fn2="", how="", vector_op="mul" ):
    fidx1 = lookup_feature_by_name( fn1)
    fidx2 = lookup_feature_by_name( fn2)
    (XY, new_feature_pandas_idx ) = generate_feature( XY, fidx1, j=fidx2, how=how, OP=vector_op )
    return (XY, new_feature_pandas_idx)
# ###################################################################################


# ##########################################################################################
def clean_data( XY, method="drop", debug=False, drop_cols=False, repvals=[], y_column=0 ):
    print "BEFORE: ", XY.shape, "CLEANED DATA"

    if debug: Xin = XY.describe() 

    if "replace" in method:
        print "Replacing NAs where specified"
        if len(repvals):
            for c in XY:
                XY[c].replace( np.NaN, repvals[c] )

    if "interpolate" in method:
        print "Interpolating NAs when possible"
        for c in XY:
            if c == y_column: continue
            XY[c] = XY[c].interpolate(method='spline', order=3)

    if "drop" in method:
        print "Dropping remnant NAs across selected X-axis"
        XY = XY.dropna(axis=int(drop_cols))  #drops either rows or columns, drop_cols=False, drops rows (samples)

    print "AFTER:  ", XY.shape, "CLEANED DATA"

    if debug: Xout = XY.describe() 

    if debug and not drop_cols:
        n_out = ""
        for c in XY:
            out = ""
            impacted = False
            for (n, a, b) in zip(['count','mean','std','min','25%','50%','75%','max'], Xin[c], Xout[c] ):
                if a != b:
                    impacted = True
                    out += "%10s \t %10.3f \t %10.3f\n" % ( n, a, b )
            if impacted:
                print "COLUMN %s WAS MODIFIED, WITH IMPACT AS FOLLOWS" % c
                print out[:-1]
                print HEADER
            else:
                n_out += "%s, " % c
        print "COLUMN(S): " + n_out + "WERE NOT MODIFIED."

    return XY
# ##########################################################################################


# ###################################################################################
def split_dataset( X, y, split_ratio=0.4 ):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=2)
    return ( X_train, y_train, X_test, y_test)
# ##########################################################################################


# ##########################################################################################
def LOAD_FEATURE_HEADER( FEATURE_DESCRIPTION_HEADER="", n=None, debug=False ):
    global FEATURES

    if FEATURE_DESCRIPTION_HEADER:
        for fline in FEATURE_DESCRIPTION_HEADER.splitlines():
            if fline.strip():
                fvals = FHEADER_REGEX.findall( fline )[0]
                fnum, fname, fdesc = fvals[0],fvals[1],fvals[2]
                idx = len( FEATURES.keys() )
                FEATURES[ idx ] = [ fnum[:], fname[:], fdesc[:] ]
                if debug: print idx, FEATURES[idx]
    else:
        for i in range(n):
            FEATURES[ i ] = [ str(i+1), str(i+1), "N/A" ]

    if debug:
        for f in FEATURES:
            print f, "%8s \t %24s \t %44s" % tuple( FEATURES[f] )

    print HEADER
    return
# ##########################################################################################


# ##########################################################################################
def specialized_feature_generation_for_autos_datafile( XY, usecols ):
    '''
    10. wheel-base:               continuous from 86.6 120.9.
    11. length:                   continuous from 141.1 to 208.1.
    12. width:                    continuous from 60.3 to 72.3.
    13. height:                   continuous from 47.8 to 59.8.
    14. curb-weight:              continuous from 1488 to 4066.
    17. engine-size:              continuous from 61 to 326.
    19. bore:                     continuous from 2.54 to 3.94.
    20. stroke:                   continuous from 2.07 to 4.17.
    21. compression-ratio:        continuous from 7 to 23.
    22. horsepower:               continuous from 48 to 288.
    23. peak-rpm:                 continuous from 4150 to 6600.
    24. city-mpg:                 continuous from 13 to 49.
    '''
    ( XY, pandas_idx) = generate_interaction( XY, fn1="horsepower", fn2="", how="sqrt" )
    usecols.append(pandas_idx)
    ( XY, pandas_idx) = generate_interaction( XY, fn1="curb-weight", fn2="compression-ratio", how="interaction", vector_op="div" )
    usecols.append(pandas_idx)
    ( XY, pandas_idx) = generate_interaction( XY, fn1="sqrt(horsepower)", fn2="compression-ratio", how="interaction", vector_op="div")
    usecols.append(pandas_idx)
    XY = clean_data( XY, method="interpolation|drop", drop_cols=False )
    # interface to reach by feature number instead, useful for further derivation of new features over new features
    # (XY, idx ) = generate_feature( XY, 20, j=None, how="interaction" )
    # usecols.append(idx)
    # (XY, idx ) = generate_feature( XY, 21, j=None, how="sqrt" )
    # usecols.append(idx)
    # (XY, idx ) = generate_feature( XY, 20, j=22, how="interaction" )
    # usecols.append(idx)
    # (XY, idx ) = generate_feature( XY, 13, j=10, how="interaction", OP="div" )
    # usecols.append(idx)
    # XY = clean_data( XY, method="interpolation|drop", drop_cols=False )
    return XY, usecols
# ##########################################################################################


# ##########################################################################################
def specialized_feature_generation_for_spambase_datafile( XY, usecols ):
    ( XY, pandas_idx) = generate_interaction( XY, fn1="word_freq_remove", fn2="", how="sqrt" )
    usecols.append(pandas_idx)
    ( XY, pandas_idx) = generate_interaction( XY, fn1="word_freq_remove", fn2="word_freq_order", how="interaction", vector_op="mul" )
    usecols.append(pandas_idx)
    XY = clean_data( XY, method="interpolation|drop", drop_cols=False )
    return XY, usecols
# ##########################################################################################


# ##########################################################################################
'''
filepath_or_buffer: 
sep or delimiter:
delim_whitespace:
dtype: A data type name or a dict of column name to data type. If not specified, data types will be inferred.
header: row number(s) to use as the column names, and the start of the data. Defaults to 0 if no names passed, otherwise None. Explicitly pass header=0 to be able to replace existing names. The header can be a list of integers that specify row locations for a multi-index on the columns E.g. [0,1,3]. Intervening rows that are not specified will be skipped. (E.g. 2 in this example are skipped)
names: List of column names to use as column names. To replace header existing in file, explicitly pass header=0.
skiprows: A collection of numbers for rows in the file to skip. Can also be an integer to skip the first n rows
na_values: optional list of strings to recognize as NaN (missing values), either in addition to or in lieu of the default set.
true_values: list of strings to recognize as True
false_values: list of strings to recognize as False
keep_default_na: whether to include the default set of missing values in addition to the ones specified in na_values
parse_dates: if True then index will be parsed as dates (False by default). You can specify more complicated options to parse a subset of columns or a combination of columns into a single date column (list of ints or names, list of lists, or dict) [1, 2, 3] -> try parsing columns 1, 2, 3 each as a separate date column [[1, 3]] -> combine columns 1 and 3 and parse as a single date column {‘foo’ : [1, 3]} -> parse columns 1, 3 as date and call result ‘foo’
quotechar : string, The character to used to denote the start and end of a quoted item. Quoted items can include the delimiter and it will be ignored.
quoting : int, Controls whether quotes should be recognized. Values are taken from csv.QUOTE_* values. Acceptable values are 0, 1, 2, and 3 for QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONE, and QUOTE_NONNUMERIC, respectively.
skipinitialspace : boolean, default False, Skip spaces after delimiter
nrows: Number of rows to read out of the file. Useful to only read a small portion of a large file
chunksize: An number of rows to be used to “chunk” a file into pieces. Will cause an TextFileReader object to be returned. More on this below in the section on iterating and chunking
skip_footer: number of lines to skip at bottom of file (default 0)
converters: a dictionary of functions for converting values in certain columns, where keys are either integers or column labels
encoding: a string representing the encoding to use for decoding unicode data, e.g. 'utf-8` or 'latin-1'.
verbose: show number of NA values inserted in non-numeric columns
squeeze: if True then output with only one column is turned into Series
usecols: a subset of columns to return, results in much faster parsing time and lower memory usage.
'''
# ##########################################################################################
def LOAD_DATASET( filename, usecols=None, y_column=None, sep=',', feature_generator_func=None, debug=False):
    XY, X, Y = [[]], [[]], []

    XY = pandas.read_csv(filename, 
                         usecols=usecols, 
                         header=None, 
                         parse_dates=True, 
                         na_values='?', 
                         error_bad_lines=False, 
                         verbose=True, 
                         keep_default_na=True)

    if not usecols: usecols = [ int(x) for x in XY.columns ]

    XY = clean_data( XY, method="interpolation|drop", drop_cols=False, y_column=y_column )

    for i,col in enumerate(XY):
        if i>=100: break
        print "X COLUMN %2s: " % i, 
        for ij in XY[col][0:20]:
            print "%.1f" % ij,
        print
    print HEADER

    if feature_generator_func:
        print "FEATURE_GENERATOR BEING APPLIED"
        XY = clean_data( XY, method="interpolation|drop", drop_cols=False, y_column=y_column )
        XY, usecols = feature_generator_func( XY, usecols )
        print HEADER

    if y_column>=0:
        Y = XY[y_column]
        if debug:
            print HEADER
            print "Y", [ yy for yy in Y][0:50]
            print HEADER
        Y = np.array([ int(round(yy)) for yy in Y ])
        # print "Y", [ yy for yy in Y][0:50]
        usecols = [x for x in usecols if x not in (y_column,)]
        X = XY[[x for x in usecols if x not in (y_column,)]]
        describe_data( X, Y, heading="X" )
    else:
        describe_data( XY )

    ( X, scaler ) = do_scale( X )

    # add_timestamp( "FEATURE GENERATION")

    return X, Y, usecols
# ##########################################################################################


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
        ax = plt.subplot(len(DATASETS), len(CLASSIFIER_MODELS) + 1, subplot_idx)
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

    ax = plt.subplot(len(DATASETS), len(CLASSIFIER_MODELS) + 1, subplot_idx)
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


# ##########################################################################################
def get_feature_scores( fSelector, pmin=1E-6 ):
    indices = range( len( [ float(x) for x in fSelector.scores_] ) )
    X_feature_scores = zip(indices, fSelector.scores_, fSelector.pvalues_ )
    X_feature_scores = sorted( X_feature_scores, key=lambda x: x[2] )
    idx = None
    for i, (f,v,p) in enumerate(X_feature_scores):
        if idx==None and p>pmin: idx = i
        try:
            print "X[%3d] \t %8.2f \t %8.6f \t %s" % ( f, v, p, get_fname( USECOLS[f] ) )
        except:
            print "X[%3d] \t %8.2f \t %8.6f \t %s" % ( f, v, p, "N/A" )
    print '-' * 80
    return ( X_feature_scores, idx )
# ##########################################################################################


# ##########################################################################################
def get_ranked_features( fSelector, pmin=1E-6 ):
    indices = range( len( [ float(x) for x in fSelector.scores_] ) )
    X_feature_scores = zip(indices, fSelector.scores_, fSelector.pvalues_ )
    X_feature_scores = sorted( X_feature_scores, key=lambda x: x[2] )
    idx = None
    for i, (f,v,p) in enumerate(X_feature_scores):
        if idx==None and p>pmin: idx = i
        try:
            print "X[%3d] \t %8.2f \t %8.6f \t %s" % ( f, v, p, get_fname( USECOLS[f] ) )
        except:
            print "X[%3d] \t %8.2f \t %8.6f \t %s" % ( f, v, p, "N/A" )
    print '-' * 80
    features = [ x[0] for x in X_feature_scores ]
    return features
# ##########################################################################################


# ##########################################################################################
def APPLY_FEATURE_SELECTION( X, y, k=2, dtype='regression', scoring_func=f_classif, debug=0 ):
    if debug:
        for i,x in enumerate(X):
            if sum( [ xi for xi in x if xi < 0.0 ]):
                print "%s \t %50s" % ( i, x )

    if dtype == 'classification':
        fSelector = SelectKBest(scoring_func, k=k)
        Xn = fSelector.fit_transform(X, y)

    n = len(fSelector.scores_)

    print '-' * 80
    print "%6s \t %6s \t %8s" % ( "FEATURE", "SCORE", "P-VAL" )
    print '-' * 80
    ( features, cutoff ) = get_feature_scores( fSelector, pmin=1E-3 )
    print "ORIGINALLY: %s ---> TRANSFORMED INTO %s CUTOFF %s:%s" % ( X.shape, Xn.shape, cutoff, k )

    if cutoff and cutoff < k:
        fSelector = SelectKBest(f_classif, k=cutoff)
        Xn = fSelector.fit_transform(X, y)
        print "RETRANSFORMED: %s ---> TRANSFORMED INTO %s" % ( X.shape, Xn.shape )
    print '-' * 80

    return (fSelector, Xn, y)
# ##########################################################################################


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
    ax = plt.subplot(len(DATASETS), len(CLASSIFIER_MODELS) + 1, subplot_idx)
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
# ###################################################################################
def apply_emsemble( subplot_idx, dataset, h=0.02 ):
    X, y = dataset

    X = StandardScaler().fit_transform(X)

    ( X_train, y_train, X_test, y_test, y_labels ) = split_dataset( X, y )

    VOTER = VOTING( X, y, len(CLASSIFIER_MODELS), WHICH_VOTE, debug=0, csv="voting_classifier_details_%s.csv" % subplot_idx )

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    (xx, yy) = basic_mesh_plot( X, X_train, y_train, X_test, y_test, h=h )

    # multipage pdf handling from: http://matplotlib.org/examples/pylab_examples/multipage_pdf.html
    pdf = PdfPages('rocs-%s.pdf' % subplot_idx)
    d = pdf.infodict()
    d['Title']        = 'Classifier Emsembles ROC'
    d['Author']       = u'from: http://matplotlib.org/examples/pylab_examples/multipage_pdf.html\xe4nen'
    d['Subject']      = 'NRM'
    d['Keywords']     = "machine learning, scikit, classification, cross comparison"
    d['CreationDate'] = datetime.datetime(2014, 8, 5)
    d['ModDate']      = datetime.datetime.today()

    subplot_idx += 1
    for cname, clf in zip(CLASSIFIER_NAMES, CLASSIFIER_MODELS):
        plt.title('%s Page %s - %s' % ("ROCS", cname, subplot_idx ))

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
            pdf.savefig(roc_fig)    # add a page with this plot to the multipage pdf object
            roc_fig.clf()
            roc_fig.close()
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

    # finalize the multipage pdf object
    pdf.close()

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
    ctimes_matrix = np.reshape( ctimes, (len(CLASSIFIER_MODELS),len(DATASETS)), order='F' )
    print "%24s |\t %4s \t %4s \t %4s" % ( "CLASSIFIER-PERFORMANCE", "MOONS", "CIRCLES", "LIN_SEPARABLE" )
    print "-"* 80
    for i,name in enumerate(CLASSIFIER_NAMES):
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
def do_dataset_generation_basic( NumSamples, NumClasses=2, NumFeatures=2, multiclass=False, apply_stdscaler=True ):
    # n_classes * n_clusters_per_class mustbe smaller or equal 2 ** n_informative
    if multiclass:
        from sklearn import datasets
        iris, digits = datasets.load_iris(), datasets.load_digits()
        X1, y1 = iris.data, iris.target
        X2, y2 = digits.data, digits.target
    else:
        X1, y1 = make_moons(n_samples=NumSamples, noise=0.3, random_state=0)
        X2, y2 = make_circles(n_samples=NumSamples, factor=0.5, noise=0.2, random_state=1)

    if apply_stdscaler:
        (X1, x1_scaler ) = do_scale( X1 )
        (X2, x2_scaler ) = do_scale( X2 )

    fsel1, X1n, y1 = APPLY_FEATURE_SELECTION( X1, y1, k=2, dtype='classification', scoring_func=f_classif )
    fsel2, X2n, y2 = APPLY_FEATURE_SELECTION( X2, y2, k=2, dtype='classification', scoring_func=f_classif )

    return [ (X1n, y1), (X2n, y2) ]
# ###################################################################################


# ###################################################################################
def do_dataset_generation_multiclass( NumSamples, NumClasses=2, NumFeatures=2, multiclass=False, apply_stdscaler=True ):
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
        (X3, x2_scaler ) = do_scale( X3 )

    fsel3, X3n, y3 = APPLY_FEATURE_SELECTION( X3, y3, k=NumFeatures, dtype='classification', scoring_func=f_classif )

    return (X3n, y3)
# ###################################################################################


# ###################################################################################
def basic_check_on( X, Y ):
    print HEADER
    print "X"
    for xx in X[0:5,]:
        for xxx in xx:
            print "%.2f" % xxx,
        print
    print HEADER
    if len(Y):
        print "Y"
        for yy in Y:
            print "%.1f" % yy, 
        print
        print HEADER
# ###################################################################################


# ###################################################################################
if __name__ == "__main__":
    DEBUG = True

    # ########################################################################################
    # HANDLING OF REGRESSION TEST DATASETS
    # ########################################################################################
    NumSamples, NumClasses, NumFeatures, multiclass = 1000, 3, 4, False
    USECOLS            = range( NumFeatures )
    DATASET_BENCHMARKS = do_dataset_generation_basic( NumSamples, NumClasses, NumFeatures, multiclass )
    DATASET_MULTICLASS = do_dataset_generation_multiclass( NumSamples, NumClasses, NumFeatures, multiclass )
    DATASETS = DATASET_BENCHMARKS


    # ########################################################################################
    # HANDLING OF INCOMING DATASET
    # ########################################################################################
    LOAD_FEATURE_HEADER( FEATURE_DESCRIPTION_HEADER )

    YCOL_IS_FACTOR    = True
    FEATURE_SELECTION = True
    DATASET_FILE      = "autos/imports-85.data"
    PREDICT_COL       = 5-1
    DROP_HOW_MANY     = 7
    INVALID_COLUMNS   = [15, 16, 18, 25]
    VALID_COLS        = [PREDICT_COL+1,] + range(10,26)
    FEATURE_GENERATOR_FUNC = specialized_feature_generation_for_autos_datafile
    USECOLS           = sorted([ x-1 for x in sorted(set(VALID_COLS)) if x not in set(INVALID_COLUMNS) ])
    NUM_FEATURES      = len(VALID_COLS)

    YCOL_IS_FACTOR    = False
    FEATURE_SELECTION = True
    DATASET_FILE      = "spambase/spambase.data"
    DROP_HOW_MANY     = 54
    PREDICT_COL       = 57
    INVALID_COLUMNS   = [ ]
    VALID_COLS        = range(58)
    USECOLS           = sorted([ x for x in sorted(set(VALID_COLS)) if x not in set(INVALID_COLUMNS) ])
    FEATURE_GENERATOR_FUNC = specialized_feature_generation_for_spambase_datafile
    NUM_FEATURES      = len(VALID_COLS)


    # ########################################################################################
    # DATA LOADING AND PREPARATION FOR THE SPECIFIED DATASET 
    # ########################################################################################
    X, Y, USECOLS = LOAD_DATASET( DATASET_FILE, 
                                  usecols=USECOLS,
                                  y_column=PREDICT_COL, 
                                  feature_generator_func = FEATURE_GENERATOR_FUNC )

    if YCOL_IS_FACTOR: 
        ( VLABELS, VENCODER, VDECODER, Y ) = decode_factor( Y )

    fSelector, XX, YY = APPLY_FEATURE_SELECTION( X, Y, k=NUM_FEATURES-DROP_HOW_MANY, dtype='classification', scoring_func=f_classif )

    DROP_COLS = do_feature_descriptions( X, Y, which_cols=get_ranked_features( fSelector) )
    DROP_COLNAMES = [ get_fname(USECOLS[c]) for c in DROP_COLS ]

    M, N = XX.shape
    print HEADER
    print "WILL BE ANALYZING USING THESE FEATURES*: (*pandas columns starts at 0)"
    for idx, i in enumerate(USECOLS):
       cname = FEATURES[i][1]
       msg = ""
       if cname in DROP_COLNAMES: msg = "CORR_DROP"
        
       print "%3s %3s" % (idx, i), "\t %6s \t %42s \t %48s" % tuple( FEATURES[i] ), "\t %s" % msg
    print HEADER
    print "AND WILL BE PREDICTING THIS COLUMN:"
    print "%3s %3s" % (PREDICT_COL, PREDICT_COL+1), "\t %6s \t %20s \t %48s" % tuple( FEATURES[PREDICT_COL] )
    print HEADER


    # ########################################################################################
    # HANDLING OF REGRESSION TEST DATASETS
    # ########################################################################################
    DATASET_TO_ANALYZE = ( XX, YY )
    DATASETS.append( DATASET_TO_ANALYZE )


    # ########################################################################################
    # APPLICATION OF THE CLASSIFIERS TO THE DATASETS
    # ########################################################################################
    figure = plt.figure(1,figsize=(27, 12))
    subplot_idx = 1

    for k, dataset in enumerate(DATASETS):
        t0 = time.time()

        apply_emsemble(subplot_idx, dataset )

        subplot_idx = subplot_idx + len(CLASSIFIER_MODELS) + 1

        add_timestamp( "DATASET-%s" % k, time.time(), t0 )

    figure.subplots_adjust(left=.02, right=.98)
    figure.show()
    figure.savefig('comparative-performance.png')


    # ########################################################################################
    # ########################################################################################
    do_post_analytics()
    print '-' * 80
    print 'please lookup file comparative-performance.png for the comparison plot of these classifiers'
    print 'please lookup files voting_classifier_details_*.csv for details of the current emsemble voter engine'
    print '-' * 80


