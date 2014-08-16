#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
ORIGINALLY: from various scikit.learn code snippets 
MODIFIED: Nelson & Hobbes ( Holly Spirit Buried in Backyard And Murdered by Crimericans)
DATE: July 31, 2014
'''
print(__doc__)


# ##########################################################################################
FEATURE_DESCRIPTION_HEADER = '''
     1. symboling:                -3, -2, -1, 0, 1, 2, 3.
     2. normalized-losses:        continuous from 65 to 256.
     3. make:                     alfa-romero, audi, bmw, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercedes-benz, mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo
     4. fuel-type:                diesel, gas.
     5. aspiration:               std, turbo.
     6. num-of-doors:             four, two.
     7. body-style:               hardtop, wagon, sedan, hatchback, convertible.
     8. drive-wheels:             4wd, fwd, rwd.
     9. engine-location:          front, rear.
    10. wheel-base:               continuous from 86.6 120.9.
    11. length:                   continuous from 141.1 to 208.1.
    12. width:                    continuous from 60.3 to 72.3.
    13. height:                   continuous from 47.8 to 59.8.
    14. curb-weight:              continuous from 1488 to 4066.
    15. engine-type:              dohc, dohcv, l, ohc, ohcf, ohcv, rotor.
    16. num-of-cylinders:         eight, five, four, six, three, twelve, two.
    17. engine-size:              continuous from 61 to 326.
    18. fuel-system:              1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi.
    19. bore:                     continuous from 2.54 to 3.94.
    20. stroke:                   continuous from 2.07 to 4.17.
    21. compression-ratio:        continuous from 7 to 23.
    22. horsepower:               continuous from 48 to 288.
    23. peak-rpm:                 continuous from 4150 to 6600.
    24. city-mpg:                 continuous from 13 to 49.
    25. highway-mpg:              continuous from 16 to 54.
    26. price:                    continuous from 5118 to 45400.
    '''
# ##########################################################################################


# ##########################################################################################
import re
import time
import numpy as np
import pandas
import math
import numpy.linalg
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.metrics import explained_variance_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor
from matplotlib import mlab
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from scipy.stats.distributions import  t
# ##########################################################################################


# ##########################################################################################
HEADER = "-" * 80
NAN    = np.nan
# ##########################################################################################
global PREVTIME
METRICS={}
PREVTIME = time.time()
# ##########################################################################################
FHEADER_REGEX = re.compile( '\s(\d+)\. (\w.*)?:\s+(.*$)' )
global FEATURES
global FEATURES_MAPPING
FEATURES = {} 
FEATURES_MAPPING = {} 
# ##########################################################################################


# ###################################################################################
def add_timestamp( qname ):
    global PREVTIME

    TIMENOW = time.time()
    if qname not in METRICS:
        METRICS[qname] = [ qname[:], PREVTIME, TIMENOW, TIMENOW-PREVTIME ]
    else:
        pass
    PREVTIME = TIMENOW
# ###################################################################################


# ###################################################################################
def do_scale( X ):
    X_scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
    Xs = X_scaler.transform(X)                               
    return ( Xs, X_scaler ) 
# ###################################################################################


# ##########################################################################################
def describe_data( XY, heading="", debug=False ):
    print HEADER
    if heading: print " MATRIX: %s \t SHAPE: %s \t" % ( heading, XY.shape )
    print XY.dtypes
    if debug:
        print XY.describe()
        print
    print HEADER
# ##########################################################################################


# ###################################################################################
def get_df_column( X, idx ):
    assert len(X[X.columns[0]]) == len(X[idx]), "BASIC PROBLEM WITH DATAFRAME STRUCTURE"
    Xi = X[idx]
    return Xi
# ###################################################################################


# ###################################################################################
def add_df_column_to( X, data_column, debug=1 ):
    assert len(X[X.columns[0]]) == len(data_column), "COLUMNS ARE NOT EQUAL IN SIZE"
    idx_where_to = max([x for x in X.columns ]) + 10
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
    # FEATURES_MAPPING = dict([ get_ftuple_from( key, retval="mapping" ) for key in FEATURES.keys() ])

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
def clean_data( XY, method="drop", debug=False, drop_cols=False, repvals=[] ):
    print "BEFORE: ", XY.shape

    if debug: Xin = XY.describe(percentile_width=None, percentiles=[25,75]) 

    if "replace" in method:
        print "Replacing NAs where specified"
        if len(repvals):
            for c in XY:
                XY[c].replace( np.NaN, repvals[c] )

    if "interpolate" in method:
        print "Interpolating NAs when possible"
        XY = XY.interpolate(method='spline', order=3)

    if "drop" in method:
        print "Dropping remnant NAs across selected X-axis"
        XY = XY.dropna(axis=int(drop_cols))  #drops either rows or columns, drop_cols=False, drops rows (samples)

    print "AFTER:  ", XY.shape

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
    else:
        for i in range(n):
            FEATURES[ i ] = [ str(i+1), str(i+1), "N/A" ]

    if debug:
        for f in FEATURES:
            print f, "%s \t %s \t %s" % tuple( FEATURES[f] )

    print HEADER
    return
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
def LOAD_DATASET( filename, usecols=None, y_column=None, sep=',', debug=False ):
    XY, X, Y = [[]], [[]], []

    XY = pandas.read_csv(filename, 
                         usecols=usecols, 
                         header=None, 
                         parse_dates=True, 
                         na_values='?', 
                         error_bad_lines=False, 
                         verbose=True, 
                         keep_default_na=True)
    if debug: print XY.head(5)

    if not usecols: usecols = [ int(x) for x in XY.columns ]

    XY = clean_data( XY, method="interpolation|drop", drop_cols=False )

    add_timestamp( "DATASET LOADING")

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

    if y_column:
        Y = XY[y_column]
        usecols = [x for x in usecols if x not in (y_column,)]
        X = XY[[x for x in usecols if x not in (y_column,)]]
        describe_data( X, heading="X" )
        describe_data( Y, heading="Y" )
    else:
        describe_data( XY )

    ( X, scaler ) = do_scale( X )

    add_timestamp( "FEATURE GENERATION")

    return X, Y, usecols
# ##########################################################################################


# ##########################################################################################
def get_feature_scores( fSelector, pmin=1E-6 ):
    indices = range( len( [ float(x) for x in fSelector.scores_] ) )
    X_feature_scores = zip(indices, fSelector.scores_, fSelector.pvalues_ )
    X_feature_scores = sorted( X_feature_scores, key=lambda x: x[2] )
    idx = None
    for i, (f,v,p) in enumerate(X_feature_scores):
        if idx==None and p>pmin: idx = i
        print "X[%3d] \t %8.2f \t %8.6f \t %s" % ( f, v, p, get_fname( USECOLS[f] ) )
    print '-' * 80
    return ( X_feature_scores, idx )
# ##########################################################################################


# ##########################################################################################
def apply_feature_selection( X, y, k=2, dtype='regression', scoring_func=f_classif, debug=0 ):
    if debug:
        for i,x in enumerate(X):
            if sum( [ xi for xi in x if xi < 0.0 ]):
                print "%s \t %50s" % ( i, x )

    if dtype == 'regression':
        fSelector = SelectKBest(f_regression, k=k)
        Xn = fSelector.fit_transform(X, y)

    n = len(fSelector.scores_)

    print '-' * 80
    print "%6s \t %6s \t %8s" % ( "FEATURE", "SCORE", "P-VAL" )
    print '-' * 80
    ( features, cutoff ) = get_feature_scores( fSelector, pmin=1E-3 )
    print "ORIGINALLY: %s ---> TRANSFORMED INTO %s CUTOFF %s:%s" % ( X.shape, Xn.shape, cutoff, k )

    if cutoff < k:
        fSelector = SelectKBest(f_regression, k=cutoff)
        Xn = fSelector.fit_transform(X, y)
        print "RETRANSFORMED: %s ---> TRANSFORMED INTO %s" % ( X.shape, Xn.shape )
    print '-' * 80

    return (fSelector, Xn, y)
# ##########################################################################################


# ##########################################################################################
def get_design_matrix( X, debug=0 ):
    Xn = []
    for row in X:
        nrow = [1,]
        nrow.extend( row )
        if debug: print len(nrow), nrow
        Xn.append( nrow[:] )
    Xn = np.matrix(Xn)
    print "(M)x(N+1) Design Matrix Created", Xn.shape
    return Xn
# ##########################################################################################


# ##########################################################################################
def dot( x, y ):
    m, n = len(x), len(y) 
    if m != n:
        print m, n
        print x
        print y
        assert m == n
    xy = sum([ a*float(b) for (a,b) in zip( x, y) ])
    return xy
# ##########################################################################################



# ##########################################################################################
'''
test = numpy.array([[1, 2],  
                    [3, 4], 
                    [5, 6]])
test[:,0] -->  array([1, 3, 5]) col
test[1,:] -->  array([3, 4]) row
'''
# ##########################################################################################


# ##########################################################################################
def get_np_row( x, i ):
    row = [ item for item in x[i,:]]
    return row
# ##########################################################################################


# ##########################################################################################
def get_np_col( x, j ):
    col = [ item for item in x[:,j]]
    return col
# ##########################################################################################


# ##########################################################################################
def cost_function( Y, h0X ):
    m, n = len(Y), len(h0X) 
    if m != n:
        print m, n
        assert m == n
    J = sum( [ (yp-yt)*(yp-yt) for (yp,yt) in zip(Y,h0X) ] ) / float( 2.0 * m )
    return J
# ##########################################################################################


# ##########################################################################################
#         (N+1)xM       Mx(N+1)       (N+1)xM                Mx1
# Theta  =    (Xt * X)^-1 *                     Xt * Y
# ##########################################################################################
def apply_normal_equation( X, Y, regularization=True, alpha=1.0, debug=0 ):
    assert X.shape[1] != Y.shape

    Xn = get_design_matrix( X )
    Xt = np.transpose(Xn)
    XtX = Xt * Xn

    if regularization:
        print "REGULARIZATION BEING APPLIED, USING LAMBDA:", alpha 
        m = Xn.shape[1]
        R = np.zeros((m,m)) 
        for i in range(2,m):
            R[i,i] = 1.0
        R = alpha * R
        Xinv = np.linalg.pinv(  XtX + R ) 
    else:
        Xinv = np.linalg.pinv(  XtX ) 
    XinvXt = Xinv * Xt

    if debug: print "Xt", Xt.shape, "XtX", XtX.shape, "Xinv", Xinv.shape, "XinvXt", XinvXt.shape, "Y", Y.shape
    assert XtX.shape[0] == Xn.shape[1] 

    Theta = []
    for i, featurerow in enumerate(XinvXt):
        if debug: print "*",i
        theta_i = dot( np.reshape(get_np_row(XinvXt,i),len(Y)), Y)
        Theta.append( float(theta_i) )

    for i, t in enumerate(Theta):
        print "Theta(%4s) = %6.4f" % (i, Theta[i])
    print HEADER

    ( y_pred, J, SCORE )  = predict_linear_model( X, Xn=Xn, Y=Y, Theta=Theta )

    return ( Theta, y_pred, J )
# ##########################################################################################


# ##########################################################################################
def predict_linear_model( X, Xn=[], Y=[], Theta=[], model=None, debug=1 ):
    if not len(Xn):
        Xn = get_design_matrix( X )

    y_pred = []
    for i, col in enumerate(Xn):
        y_i = dot( Theta, np.reshape(get_np_row(Xn, i),len(Theta)) )
        y_pred.append(y_i)

    J, SCORE = np.NaN, np.NaN
    if len(Y):
        J = cost_function( Y, y_pred )
        SCORE = explained_variance_score ( Y, y_pred )

    if debug: 
        print "X shape: \t %s" % repr(Xn.shape)
        if len(Y): print "Y shape: \t %s" % repr(Y.shape)
        print "Theta:   \t %s" % Theta 
        print "J(Theta):\t %s" % J
        print "SCORE:   \t %s" % SCORE
        print HEADER
    return ( y_pred, J, SCORE )
# ##########################################################################################


# ##########################################################################################
def plot_cost_function( XY, X, Y ):
    pass
# ##########################################################################################


# ##########################################################################################
def performance_analysis( model, Theta, X_train, Y_train, debug=1 ): 
    # learning curves
    '''
    shuffle x, y
    divide x,y into p batches (size=M/p
    learnin_scores = {}
    for i in p:
        train=batches[0:i] of x
        test=batches[0:i] of y
        learnin_scores[ (i+1) * M/p ] = mean ( cross_validated scores( train, test ) )
    plot learning_scores
    '''
    y_fitted = model.predict( X_train )
    SCORE = model.score(X_train, Y_train)
    R2    = r2_score( Y_train, y_fitted )
    J     = cost_function( Y_train, y_fitted )
    if debug:
        print "X shape:  \t %s"   % repr(X_train.shape)
        print "Y shape:  \t %s"   % repr(Y_train.shape)
        print "0 theta:  \t %s"   % [ float(x) for x in Theta ]
        print "J(theta): \t %s"   % J
        print 'fitscore: \t %.2f' % SCORE
        print 'R^2    : \t %.2f' % R2
        print HEADER
    return ( model, Theta, J, SCORE )
# ##########################################################################################


# ##########################################################################################
def train_linear_regressor( X_train, Y_train, normalize=True, debug=True ):
    model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=normalize)
    model.fit (X_train, Y_train )
    Theta = [ float(model.intercept_), ]
    Theta.extend( [ float(x) for x in model.coef_])
    ( model, Theta, J, SCORE ) = performance_analysis( model, Theta, X_train, Y_train, debug=1 )
    if debug:
        import statsmodels.api as sm
        Xdm = sm.add_constant(X_train)
        stats_model = sm.OLS(Y_train, Xdm)
        detailed_results = stats_model.fit()
        print detailed_results.summary()
    return ( model, Theta, J, SCORE )
# ##########################################################################################


# ##########################################################################################
def apply_ridge( X_train, Y_train, alpha=None ):
    alphas = [ alpha ]
    if not alpha: alphas = [ x for x in sorted(set([ alpha, 0.1, 1.0/3.0, 1.0, 10.0/3.0, 10.0 ])) if x]
    ALPHA_VALS = {}
    for a in alphas:
        model = Ridge(alpha=a, 
                      fit_intercept=True, 
                      normalize=False, 
                      copy_X=True, 
                      max_iter=None, 
                      tol=0.001, 
                      solver='auto')
        # sample_weights = [ 1.0/float(len(Y)) for x in Y ]
        model.fit( X_train, Y_train )# , sample_weight=sample_weights)
        R2 = model.score(X_train, Y_train)
        L2 = dot(model.coef_,model.coef_)
        ALPHA_VALS [a ] = [ a, R2, L2, [x for x in model.coef_] ]
        print "ALPHA: %.2f \t R^2=%7.4f \t L2_NORM(THETA)=%10.2f \t THETA[1:N]=%s" % ( a, R2, L2, model.coef_ )
    # A = sorted([ ALPHA_VALS[x] for x in ALPHA_VALS [ a, R2, L2, model.coef_[:] ], key=lambda x: x[1], reversed=True )
    Theta = [ float( model.intercept_ ) , ]
    Theta.extend( [ float( x ) for x in model.coef_])
    ( model, Theta, J, SCORE ) = performance_analysis( model, Theta, X_train, Y_train, debug=1 )
    return ( model, Theta, J, SCORE )
# ##########################################################################################


# ##########################################################################################
def apply_lasso( X_train, Y_train, alpha=None ):
    alphas = [ 0.1, 0.3, 0.5 ]
    ALPHA_VALS = {}
    for a in alphas:
        model = Lasso(alpha=a, 
                    fit_intercept=True, 
                    normalize=False, 
                    precompute='auto', 
                    copy_X=True, 
                    max_iter=50000, 
                    tol=0.001, 
                    warm_start=False, 
                    positive=False)
        # sample_weights = [ 1.0/float(len(Y)) for x in Y ]
        model.fit( X_train, Y_train )# , sample_weight=sample_weights)
        R2 = model.score(X_train, Y_train)
        L1 = sum([abs(x) for x in model.coef_])
        ALPHA_VALS [a ] = [ a, R2, L1, [x for x in model.coef_] ]
        print "ALPHA: %.2f \t R^2=%7.4f \t L1(THETA)=%.2f \t THETA[1:N]=%s" % ( a, R2, L1, ", ".join(["%.4f" % x for x in model.coef_] ))
    # A = sorted([ ALPHA_VALS[x] for x in ALPHA_VALS [ a, R2, L2, model.coef_[:] ], key=lambda x: x[1], reversed=True )
    Theta = [ float( model.intercept_ ) , ]
    Theta.extend( [ float( x ) for x in model.coef_])
    ( model, Theta, J, SCORE ) = performance_analysis( model, Theta, X_train, Y_train, debug=1 )
    return ( model, Theta, J, SCORE )
# ##########################################################################################


# ##########################################################################################
def apply_sgd_( X_train, Y_train, alpha=0.0003, shuffle=True):
    n_iter = np.ceil(10**6 / len(Y_train))
    model = SGDRegressor(loss='squared_loss', 
                         penalty='l2', 
                         alpha=alpha,
                         epsilon=0.01,
                         fit_intercept=True, 
                         n_iter=n_iter, shuffle=shuffle, random_state=int(time.time()*8192)%8192, warm_start=False,
                         verbose=0, 
                         learning_rate='invscaling' )
    # model.fit_transform( X_train, Y_train )
    # model.partial_fit_transform( X_train, Y_train )
    # sample_weights = [ 1/float(m) for x in Y ]
    model.fit( X_train, Y_train, sample_weight=None )
    Theta = [ float( model.intercept_ ) , ]
    Theta.extend( [ float( x ) for x in model.coef_])
    ( model, Theta, J, SCORE ) = performance_analysis( model, Theta, X_train, Y_train, debug=1 )
    return ( model, Theta, J, SCORE )
# ##########################################################################################


# ##########################################################################################
def predict_with_parameters( X_test, Theta, model=None, Y_true=[], debug=True, n=10 ):
    if model:
        y_pred = model.predict( X_test )
        print "Y_pred: \t %s"   % y_pred[0:n]
    else:
        y_pred = np.transpose( Theta ) * X_test
        print "Y_pred: \t %s"   % y_pred[0:n]

    m = X_test.shape[0]
    if len(Y_true):
        MSE = sum( [ (yp-yt)*(yp-yt) for (yp,yt) in zip(Y_true,y_pred) ] ) / float( m )
        SCORE = model.score( X_test, Y_true )
        R2 = r2_score( Y_true, y_pred )
    else:
        MSE, SCORE, R2 = NAN, NAN, NAN

    if debug:
        print HEADER
        print "X shape: \t %s"   % X.shape
        print "Y shape: \t %s"   % Y.shape
        print "0 theta: \t %s"   % [ float(x) for x in Theta ]
        print 'MSE(0) : \t %.2f' % MSE
        print 'SCORE  : \t %.2f' % SCORE
        print 'R^2    : \t %.2f' % R2

    return y_pred
# ##########################################################################################


# ##########################################################################################
def do_diagnostics( fig, y_test, y_pred, subplot=321, label="Y_TEST v Y_PRED" ):
    # I = range(len(y_test)) 
    # xx = [ dot(row,row) for row in X_test ]
    # xx = [ float(row[0]) for row in X_test ]
    y_errors = y_test - y_pred
    y_error_abs = [ abs(yy) for yy in y_errors ]
    mu = y_errors.mean()
    sigma = y_errors.std()
    eq_xy = 1 + 1.125 * y_test.mean() + min(y_test)/2.0

    ax11 = fig.add_subplot(subplot)
    ax11.scatter(y_test, y_pred,  color='r', marker='s', s=y_error_abs, label=label)
    # ax1.plot(xx, y_pred, color='r', marker='o', linewidth=0, label=label)
    m, b = np.polyfit( y_test, y_pred, 1 ) 
    ax11.plot(y_test, m*y_test + b, '-')
    ax11.text( y_test.min(), eq_xy, "%.2f+%.2fy" % ( b, m ), fontsize=10 ) 
    ax11.grid(True)
    ax11.axhline(0, color='black', lw=2)
    ax11.set_xlabel(r'$y(i)$', fontsize=10)
    ax11.set_ylabel(r'${h_\Theta}(i)$', fontsize=10)
    # ax11.set_title('Estimator Performance')
    ax11.legend(loc='lower right')

    # http://matplotlib.org/1.2.1/examples/api/histogram_demo.html
    ax12 = fig.add_subplot( subplot+1 if subplot+1%530 else 530)
    ax12.text(0,min(.30,max(y_errors)), r'$\mu=%.2f,\ \sigma=%.2f$' % ( mu, sigma ) )
    n, bins, patches = ax12.hist(y_errors, 50, normed=1, facecolor='green', alpha=0.60)
    # bincenters = 0.5*(bins[1:]+bins[:-1])
    l = ax12.plot(bins, mlab.normpdf( bins, mu, sigma), 'r--', linewidth=2)
    ax12.set_xlabel('Error')
    ax12.set_ylabel('Probability')
    return ax11, ax12, y_errors
# ##########################################################################################


# ##########################################################################################
# http://kitchingroup.cheme.cmu.edu/blog/2013/02/12/Nonlinear-curve-fitting-with-parameter-confidence-intervals/
def compute_coefficient_intervals( y_test, Theta ):
    alpha = 0.05    # 95% confidence interval = 100*(1-alpha)
    n  = len(y_test) # number of data points
    p  = len(Theta)  # number of parameters
    df = max(0, n - p) # number of degrees of freedom
    # student-t value for the dof and confidence level
    tval = t.ppf(1.0-alpha/2., df) 
    # for i, pi in zip(range(p), Theta): print 'p{0}: {1} [{2}  {3}]'.format(i, pi, pi - sigma*tval, pi + sigma*tval)
# ##########################################################################################


# ##########################################################################################
if __name__ == "__main__":

    LOAD_FEATURE_HEADER( FEATURE_DESCRIPTION_HEADER )

    DATASET_FILE      = "autos/imports-85.data"
    FEATURE_SELECTION = True
    DROP_HOW_MANY     = 7
    PREDICT_COL       = 24-1      
    USECOLS           = [ x-1 for x in range(10,26) if x not in [15, 16, 18, 25] ]

    X, Y, USECOLS = LOAD_DATASET( DATASET_FILE, usecols=USECOLS, y_column=PREDICT_COL )
    print X.shape, Y.shape
    M, N = X.shape
    print HEADER

    print "WILL BE ANALYZING USING THESE FEATURES*: (*pandas columns starts at 0)"
    for idx, i in enumerate(USECOLS):
        print "%3s %3s" % (idx, i), "\t %6s \t %42s \t %48s" % tuple( FEATURES[i] )
    print HEADER
    print "AND WILL BE PREDICTING THIS COLUMN:"
    print "%3s %3s" % (PREDICT_COL, PREDICT_COL+1), "\t %6s \t %20s \t %48s" % tuple( FEATURES[PREDICT_COL] )
    print HEADER


    # SPLIT TRAIN AND TEST - JUST TO SELECT FEATURES FROM A RANDOM TRAINING SET, THESE SET HAS ALL COLUMNS
    ( X_train, y_train, X_test, y_test) = split_dataset( X, Y, split_ratio=0.4 )
    if FEATURE_SELECTION:
        fSelector, X, Y = apply_feature_selection( X_train, y_train, k=N-DROP_HOW_MANY, dtype='regression', scoring_func=chi2, debug=0 )
        X = fSelector.fit_transform(X, Y)

    # SPLIT TRAIN AND TEST (ACTUALLY WILL BE SAME SAMPLES AS LONG AS RANDOM SEED IS SAME, JUST WITH FEWER COLUMNS
    ( X_train, y_train, X_test, y_test) = split_dataset( X, Y, split_ratio=0.4 )
    print HEADER
    add_timestamp( "FEATURE SELECTION")


    print "TRAINING VIA NORMAL EQS"
    ( Theta, y_pred, J ) = apply_normal_equation( X_train, y_train, alpha=0.05 )
    print "TESTING"
    ( y_pred_neq, J, SCORE )  = predict_linear_model( X_test, Y=y_test, Theta=Theta )
    print HEADER
    add_timestamp( "NORMAL EQUATIONS")


    print "TRAINING VIA ORDINARY LEAST SQUARE (NE)"
    ( model, Theta, J, SCORE ) = train_linear_regressor( X_train, y_train )
    print "TESTING"
    ( y_pred_ols, J, SCORE )  = predict_linear_model( X_test, Y=y_test, Theta=Theta, model=model )
    print HEADER
    add_timestamp( "ORDINARY LEAST SQUARES")


    print "TRAINING VIA STOCHASTI GRADIENT DESCENT"
    ( model, Theta, J, SCORE ) = apply_sgd_( X_train, y_train )
    print "TESTING"
    ( y_pred_sgd, J, SCORE )  = predict_linear_model( X_test, Y=y_test, Theta=Theta, model=model )
    print HEADER
    add_timestamp( "STOCHASTIC GRADIENT DESCENT")


    print "TRAINING VIA RIDGE REGRESSION (i.e., alpha * L2(THETA)) PENALTY"
    ( model, Theta, J, SCORE ) = apply_ridge( X_train, y_train, alpha=2.0 )
    print "TESTING"
    ( y_pred_rdg, J, SCORE )  = predict_linear_model( X_test, Y=y_test, Theta=Theta, model=model )
    print HEADER
    add_timestamp( "PLACEHOLDER FOR RIDGE REGRESSION W/O ALPHA SEARCH NOR CV")


    print "TRAINING VIA LASSO REGRESSION (i.e., L1NORM(THETA) PENALTY)"
    ( model, Theta, J, SCORE ) = apply_lasso( X_train, y_train, alpha=0.25 )
    print "TESTING"
    ( y_pred_lso, J, SCORE )  = predict_linear_model( X_test, Y=y_test, Theta=Theta, model=model )
    print HEADER
    add_timestamp( "PLACEHOLDER FOR LASSO REGRESSION")


    # Plot outputs
    fig = plt.figure( 1, (9,9) )
    do_diagnostics( fig, y_test, y_pred_neq, subplot=521, label='Y v NEQ:$H_0(X)$')
    do_diagnostics( fig, y_test, y_pred_ols, subplot=523, label='Y v OLS:$H_0(X)$')
    do_diagnostics( fig, y_test, y_pred_sgd, subplot=525, label='Y v SGD:$H_0(X)$')
    do_diagnostics( fig, y_test, y_pred_rdg, subplot=527, label='Y v RDG:$H_0(X)$')
    do_diagnostics( fig, y_test, y_pred_lso, subplot=529, label='Y v LSO:$H_0(X)$')
    add_timestamp( "DIAGNOSTICS & PLOTS")

    # coarse comparative performance matrix
    for i, q in enumerate(METRICS.keys()):
        print "%s \t %.3f \t %s " % ( i, METRICS[q][3], q )
    print HEADER

    plt.show()

    # CROSS VALIDATE THE MODEL
    # FOLDING_SCHEME = KFold(M, n_folds=3, indices=None, shuffle=TrueFalse, random_state=None)
    # print HEADER

