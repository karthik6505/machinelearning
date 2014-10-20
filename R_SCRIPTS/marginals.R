# ######################################################################################################
# MACHINE LEARNING TOOLBOX FILES IN R
#           Copyright (C) Nelson R. Manohar
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# ######################################################################################################
# @AUTHOR:  Nelson R. Manohar Alers
# @EMAIL:   manohar.nelson@gmail.com
# @DATE:    September, 2014
# @URL:     http://www.bitbucket.org/nelsonmanohar/machinelearning
# ######################################################################################################
LICENSETEXT = "These R code samples (version Sep/2014), Copyright (C) Nelson R. Manohar,
comes with ABSOLUTELY NO WARRANTY.  This is free software, and you are welcome to 
redistribute it under conditions of the GNU General Public License." 
message( LICENSETEXT )
message("")
# ######################################################################################################


# ######################################################################################################
source( 'utilities.R' )
source( 'aggregate.R' )
source( 'datasets.R' )
# ######################################################################################################


# ######################################################################################################
opts = options( digits=3, width=132 )
# ######################################################################################################


# ######################################################################################################
GET_MARGINALS = function( X, Y ) {
    XM = addmargins( X, Y )
}
# ######################################################################################################


# ######################################################################################################
GET_FREQ_XY = function( X=matrix(), XM=matrix(), i ) {
    if (  nrow(XM) == 0 ) XM = GET_MARGINALS( X )
    NUM_I = XM[i,"Sum"]
    SUM_I = XM["Sum",i]
    TOTAL = XM['Sum','Sum']
    F_I = SUM_I/TOTAL
    retvals ( NUM_I, SUM_I, F_I )
    return ( retvals )
}
# ######################################################################################################


# ######################################################################################################
GET_COND_XY = function( X=matrix(), XM=matrix(), x=0, y=0 ) {
    if (  nrow(XM) == 0 ) XM = GET_MARGINALS( X )
    X_RETVALS = GET_FREQ_XY( XM=XM, i=x )
    Y_RETVALS = GET_FREQ_XY( XM=XM, i=y )
}
# ######################################################################################################


# ######################################################################################################
H_DEBUG = function( i, j, Frx, LOG=log2 ) {
    print( sprintf( "Hx[%s,%s] = -(Frx[%s,%s]=%s/Frx[%s,'Sum']=%s) * LOG(Frx[i,j]=%s/Frx[i,'Sum']=%s) ",
                  i,j,
                  i,j,Frx[i,j], i,Frx[i,'Sum'],
                  i,j,Frx[i,j], i,Frx[i,'Sum'] ) )
    val = -(Frx[i,j]/Frx[i,'Sum']) * LOG(Frx[i,j]/Frx[i,'Sum'])
    print( val )
}
# ######################################################################################################


# ######################################################################################################
#> addmargins(table( X[,1], Y ))
#            no  yes Sum
#  overcast   0    4   4
#  rainy      2    3   5
#  sunny      3    2   5
#  Sum        5    9  14
# ######################################################################################################
GET_MARGINALS_WRT = function( X=matrix(), Y=matrix(), x=0, y=0, debug=FALSE ) {
    if( class(X) != "matrix") X = as.matrix(X)
    if( class(Y) != "matrix") Y = as.matrix(Y)
    if(x==0) x=1
    if(y==0) y=1

    XX = X[,x]
    YY = Y[,y]
    Frx  = addmargins(table(XX,YY))

    return ( Frx )
}
# ######################################################################################################


# ######################################################################################################
# SHANNON ENTROPY OF A SOURCE: 
# http://math.stackexchange.com/questions/712407/correct-algorithm-for-shannon-entropy-with-r
# ######################################################################################################
GET_ENTROPY = function( SYMBOL_SOURCE, LOG=log2 ) { 
    MYFREQS = table( SYMBOL_SOURCE )/length( SYMBOL_SOURCE )
    MYVEC   = as.data.frame(MYFREQS)[,2]            # vectorize form of the named fields (just values)
    H       = sum(-MYVEC * LOG(MYVEC))               # H in bit
    return ( H )
}
# ######################################################################################################


# ######################################################################################################
# MUTUAL ENTROPY
# ######################################################################################################
GET_ENTROPY_XY_WRT = function( X=matrix(), Y=matrix(), x=0, y=0, LOG=log2, debug=FALSE ) {
    Frx = GET_MARGINALS_WRT( X=X, Y=Y, x=x, y=y, debug=debug )
    N = ncol(Frx)-1
    M = nrow(Frx)-1
    Hx = MATRIX(M,N)
    for( i in 1:M )
        for( j in 1:N )
        { Hx[i,j] = -(Frx[i,j]/Frx[i,'Sum']) * LOG(Frx[i,j]/Frx[i,'Sum']); if ( debug ) H_DEBUG( i, j, Frx ) }
    H = 0; for( i in 1:M ) H = H +   Frx[i,'Sum']/Frx['Sum','Sum']   * sum(Hx[i,1:N],na.rm=TRUE)
    return ( H )
}
# ######################################################################################################


# ######################################################################################################
# Maximal information gain ==> min. entropy choice
# TODO: with lookahead
# ######################################################################################################
WHICH_NEXT = function( HVALS=c(), debug=FALSE ) {
    print ( HVALS )
    WHICH = which( HVALS == min( HVALS, na.rm=TRUE ))
    if ( length(WHICH) > 0 ) WHICH=WHICH[1]
    if ( debug ) {
        print( sprintf( 'PICKED: %20s %.4f', rownames(HVALS)[WHICH], HVALS[WHICH] ) )
        cat( HEADER )
    }
    return ( WHICH )
}
# ######################################################################################################


# ######################################################################################################
GET_ENTROPY_VECTOR_FOR = function( X, Y ) {
    NC = ncol(X)
    HVALS = VECTOR(NC)
    rownames(HVALS) = colnames(X)
    for ( i in 1:NC )
        HVALS[i] = GET_ENTROPY_XY_WRT( X[,i], Y )

    MIN = which( HVALS == min(HVALS) )
    if ( length(MIN) == 0 ) {
        print( "WARNING: HVALS has missing values, " )
        print( HVALS )
        print ( MIN )
    }
    ATTRIBUTE_NAMES = colnames(X)[MIN]
    print( sprintf( "%20s %4s %8.4f", colnames(X), 1:NC, HVALS ) )
    cat(HEADER)

    Frx = GET_MARGINALS_WRT( X=X[,MIN], Y=Y )
    cat(HEADER)

    retvals = list( 'ENTROPIES'=HVALS, 'ATTR_NAMES'=ATTRIBUTE_NAMES, 'IG_ATTRIBUTE'=MIN, 'FREQUENCIES'=Frx )

    return ( retvals )
}
# ######################################################################################################


# ######################################################################################################
APPLY_CONDITIONAL_FUNCTION = function( X, Y, WRT_COLUMN=1, BEING_EQUAL_TO="", APPLY_FUNCTION=GET_ENTROPY_VECTOR_FOR, debug=FALSE ) {
    LN = which( str_detect(levels(X[,WRT_COLUMN]), BEING_EQUAL_TO ))
    if ( length(LN) == 0 ) print( LN )

    FACTOR_NAME = levels(X[,WRT_COLUMN])[LN] 
    if ( length(FACTOR_NAME) == 0 ) print( FACTOR_NAME )

    CONDITIONED_SELECTION = X[,WRT_COLUMN]==FACTOR_NAME
    if ( length(CONDITIONED_SELECTION) == 0 ) print( CONDITIONED_SELECTION )

    if ( debug ) print( CONDITIONED_SELECTION  )
    CONDITIONED_X = X[ CONDITIONED_SELECTION, ]
    CONDITIONED_Y = Y[ CONDITIONED_SELECTION, ]
    if ( debug ) print( cbind( CONDITIONED_X, CONDITIONED_Y ) )
    cat(HEADER)
    FVALS = APPLY_FUNCTION( CONDITIONED_X, CONDITIONED_Y )
    return ( FVALS )
}
# ######################################################################################################


# ######################################################################################################
FACTORIZE_X = function( X, nc=4 ) {
}
# ######################################################################################################


# ######################################################################################################
FACTORIZE_X_WRT = function( X, Y, nc=4 ) {
}
# ######################################################################################################


# ######################################################################################################
GET_ATTRIBUTE_NAME = function( X, COLNUM ) { return ( colnames(X)[COLNUM] ) }
# ######################################################################################################


# ######################################################################################################
GET_ATTRIBUTE_LEVELS = function( X, COLNUM ) { return ( levels(X[,COLUMN]) ) }
# ######################################################################################################


# ######################################################################################################
GET_FACTOR_PURITY = function( Frx, LOG=log2, debug=FALSE ) { 
    N = ncol(Frx)-1
    M = nrow(Frx)-1
    Hx = MATRIX(M,N)
    for( i in 1:M )
        for( j in 1:N )
        { Hx[i,j] = -(Frx[i,j]/Frx[i,'Sum']) * LOG(Frx[i,j]/Frx[i,'Sum']); if ( debug ) H_DEBUG( i, j, Frx ) }
    P = sapply( 1:M, function(x)  Frx[x,'Sum'] / Frx['Sum','Sum'] * sum( Hx[x, 1:N], na.rm=TRUE ) )
    names(P) = rownames(Frx)[1:M]
    if ( debug ) {
        print( P )
        cat( HEADER )
    }
    return ( P )
}
# ######################################################################################################


# ######################################################################################################
#            no  yes Sum
#  overcast   0    4   4
#  rainy      2    3   5
#  sunny      3    2   5
#  Sum        5    9  14
# ######################################################################################################
GET_ATTRIBUTE_EXPRESSION = function( ATTR_NAMES, ATTR_VALS, ATTR_OPS=rep("==", length(ATTR_NAMES)) ) {
    A = sprintf( "X$'%s'%s'%s'", ATTR_NAMES, ATTR_OPS, ATTR_VALS )
    A = CONCAT( A )
    if ( length( ATTR_NAMES ) > 1 ) {
        A = sprintf( "(X$'%s'%s'%s') & ", ATTR_NAMES, ATTR_OPS, ATTR_VALS )
        A = CONCAT( A )
        A = substr(A,1,nchar(A)-2)
    }
    return ( A )
}
# ######################################################################################################


# ######################################################################################################
WRITE_RULE_FOR = function( ATTRIBUTE_EXPRESSION, TARGET_FACTOR_LEVEL, Frx, APPROX=FALSE ) {
    M = nrow(Frx)-1
    N = ncol(Frx)-1

    YLABEL = which(    Frx[TARGET_FACTOR_LEVEL,1:N] == 
                   max(Frx[TARGET_FACTOR_LEVEL,1:N]))

    FREQUENCY = Frx[TARGET_FACTOR_LEVEL, YLABEL]

    DISCLAIMER = ""
    if ( APPROX ) DISCLAIMER = " APPROXIMATELY, "

    LHS = sprintf( "%s WHEN[ %s ], ", DISCLAIMER, ATTRIBUTE_EXPRESSION )
    RHS = sprintf( "THEN [%s] ==> %6s [%.2f%%] (%4s SAMPLE CASES)",  
                  GET_YCLASSNAME(), names(YLABEL), FREQUENCY/Frx[TARGET_FACTOR_LEVEL,'Sum'] * 100.0, FREQUENCY )

    RULE = sprintf( "%-80s  %20s", LHS, RHS )

    SHORT_RULE = sprintf( "%s %s", LHS, names(YLABEL) )

    PARSED_RULES[[length(PARSED_RULES)+1]] <<- list( EXPRESSION=ATTRIBUTE_EXPRESSION, N=length(PARSED_RULES)+1, FREQUENCY=FREQUENCY, YNAME=GET_YCLASSNAME(), YVAL=names(YLABEL), LHS=LHS, RHS=RHS, RULE=RULE )  

    RULES[[ATTRIBUTE_EXPRESSION]] <<- SHORT_RULE

    cat(HEADER)
    print( RULE )
    cat(HEADER)
    return ( RULE )
}
# ######################################################################################################


# ######################################################################################################
GET_YCLASSNAME = function() {
    return ( YCLASSNAME )
}
# ######################################################################################################


# ######################################################################################################
IS_LEAF_NODE = function( ENTROPY, WRITE_THRESHOLD=0.1 ) {
    val = FALSE
    if ( ENTROPY < 1*WRITE_THRESHOLD ) 
        val = TRUE
    return ( val )
}
# ######################################################################################################


# ######################################################################################################
TOLERATE_AS_LEAF_NODE = function( ENTROPY, SUBTREE_SIZE, WRITE_THRESHOLD=0.1, MIN_SUBTREE_SIZE=5 ) {
    val = FALSE
    if ( (ENTROPY < 2*WRITE_THRESHOLD)  && ( SUBTREE_SIZE <= MIN_SUBTREE_SIZE ) ) 
        val = TRUE
    return ( val )
}
# ######################################################################################################


# ######################################################################################################
DO_RULE_WRITING = function( ENTROPY, ATTR_EXPRESSION, FACTOR_LEVEL, SUBTREE_SIZE, Frx, M, N ) {
    RULE_HAS_COMPLETE_COVERAGE = FALSE
        if ( IS_LEAF_NODE( ENTROPY, WRITE_THRESHOLD=0.1 ) ) {
            RULE_RETVALS = WRITE_RULE_FOR( ATTR_EXPRESSION, FACTOR_LEVEL, Frx )
            RULE_HAS_COMPLETE_COVERAGE = TRUE
        }
        else if ( TOLERATE_AS_LEAF_NODE( ENTROPY, SUBTREE_SIZE ) ) {
            RULE_RETVALS = WRITE_RULE_FOR( ATTR_EXPRESSION, FACTOR_LEVEL, Frx, APPROX=TRUE )
            RULE_HAS_COMPLETE_COVERAGE = TRUE
        }
    return ( RULE_HAS_COMPLETE_COVERAGE )
}
# ######################################################################################################


# ######################################################################################################
WRITE_RULE = function( WHICH_COLNUM=1, x=0, y=0, WRITE_THRESHOLD = RULE_ERROR_RATE, MINNODESIZE=4, debug=FALSE ) {

    if ( nrow(X) == 0 ) {
        print( "DONE. NO MORE ROWS TO SCAN" )
        return()
    }

    Frx = GET_MARGINALS_WRT( X=X, Y=Y, x=WHICH_COLNUM, y=y, debug=debug )

    FACTOR_ENTROPIES = GET_FACTOR_PURITY( Frx )

    ATTRIBUTE_NAME = colnames(X)[WHICH_COLNUM]

    M = nrow(Frx)-1
    N = ncol(Frx)-1
    for (i in 1:M) {
        NEWLINE(1)
        ATTRIBUTE_NAMES  = c()
        ATTRIBUTE_VALUES = c()

        ENTROPY       = FACTOR_ENTROPIES[i]
        FACTOR_LEVEL  = names(FACTOR_ENTROPIES)[i]
        SUBTREE_SIZE  = Frx[FACTOR_LEVEL, 'Sum']

        if ( debug ) print ( paste( ATTRIBUTE_NAME, ENTROPY, FACTOR_LEVEL, SUBTREE_SIZE )  )

        ATTRIBUTE_NAMES    = append( ATTRIBUTE_NAMES,  ATTRIBUTE_NAME )
        ATTRIBUTE_VALUES   = append( ATTRIBUTE_VALUES, FACTOR_LEVEL )
        ATTR_EXPRESSION    = GET_ATTRIBUTE_EXPRESSION( ATTRIBUTE_NAMES, ATTRIBUTE_VALUES )

        RULE_HAS_COMPLETE_COVERAGE = DO_RULE_WRITING( ENTROPY, ATTR_EXPRESSION, FACTOR_LEVEL, SUBTREE_SIZE, Frx )
        if ( RULE_HAS_COMPLETE_COVERAGE ) {
            RULE_ACTIVATED_ROWS = GET_RULE_SCOPE(ATTR_EXPRESSION)
            MAPPING[RULE_ACTIVATED_ROWS,ncol(MAPPING)] <<- RULES[[ATTR_EXPRESSION]]
            WHICH_ROWS_TO_KEEP  = WHICH_ROWS_TO_KEEP( RULE_ACTIVATED_ROWS=RULE_ACTIVATED_ROWS )
            DO_UPDATE(WHICH_ROWS_TO_KEEP)
        }

        if ( !RULE_HAS_COMPLETE_COVERAGE ) {
            SO_RETVALS = APPLY_CONDITIONAL_FUNCTION( X, Y, WRT_COLUMN=WHICH_COLNUM, 
                                                             BEING_EQUAL_TO=FACTOR_LEVEL, 
                                                             APPLY_FUNCTION=GET_ENTROPY_VECTOR_FOR )

            SO_ENTROPIES    = SO_RETVALS$ENTROPIES                # entropy of all the attributes
            SO_ATTR_NAMES   = SO_RETVALS$ATTR_NAMES               # column names
            SO_ATTR_VALUE   = WHICH_NEXT( SO_ENTROPIES )          # column number for attribute with optimal information gain
            SO_FREQUENCIES  = SO_RETVALS$FREQUENCIES              # frequency table for this attribute wrt y conditioned to above
            SO_IG_ATTRIBUTE = SO_RETVALS$IG_ATTRIBUTE             # optimal attribute

            if (length( SO_IG_ATTRIBUTE) >0 )
                SO_IG_ATTRIBUTE = SO_RETVALS$IG_ATTRIBUTE[1]      # optimal attribute

            SO_FACTOR_ENTROPIES = GET_FACTOR_PURITY( SO_FREQUENCIES )

            for ( i in 1:length(SO_FACTOR_ENTROPIES)) {
                if ( RULE_HAS_COMPLETE_COVERAGE ) {
                    # THIS STUFF GOES ABOVE UNDER IF !COVERAGE_COMPLETED
                    # TODO: FIX ATTRIBUTE EXPRESSION HERE TO EXPAND MATCHING RECURSION VIA EXPLICIT ITERATION
                    # TODO: THIS NEEDS TO BE DONE ABOVE ( WHICH ONLY ITERATES ON 2nd order and all of its factors
                    # TODO: THE ABOVE NEEDS TO BECOME A LOOP TO EXPAND TO 3RD ORDER AND SO FORTH
                    # TODO: CONSIDER: BY RANDOMLY SELECTING WHICH ATTRIBUTES TO EXAMINE AS OPPOSED TO ALL --> RANDOM SUBTREE (TO REDUCE TIME, 
                    NEWLINE(2)
                } else {
                    NEWLINE(1)
                }
                ENTROPY       = SO_FACTOR_ENTROPIES[i]
                FACTOR_LEVEL  = names(SO_FACTOR_ENTROPIES)[i]
                SUBTREE_SIZE  = SO_FREQUENCIES[FACTOR_LEVEL, 'Sum']
                ATTRIBUTE_NAMES  = PUSH( ATTRIBUTE_NAMES,  colnames(X)[SO_IG_ATTRIBUTE] )
                ATTRIBUTE_VALUES = PUSH( ATTRIBUTE_VALUES, FACTOR_LEVEL )
                    ATTR_EXPRESSION  = GET_ATTRIBUTE_EXPRESSION( ATTRIBUTE_NAMES, ATTRIBUTE_VALUES )
                    RULE_HAS_COMPLETE_COVERAGE = DO_RULE_WRITING( ENTROPY, ATTR_EXPRESSION, FACTOR_LEVEL, SUBTREE_SIZE, SO_FREQUENCIES )
                    if ( RULE_HAS_COMPLETE_COVERAGE ) { 
                        RULE_ACTIVATED_ROWS = GET_RULE_SCOPE(ATTR_EXPRESSION)
                        MAPPING[RULE_ACTIVATED_ROWS,ncol(MAPPING)] <<- RULES[[ATTR_EXPRESSION]]
                        WHICH_ROWS_TO_KEEP  = WHICH_ROWS_TO_KEEP( RULE_ACTIVATED_ROWS=RULE_ACTIVATED_ROWS )
                        DO_UPDATE(WHICH_ROWS_TO_KEEP)
                    }
                ATTRIBUTE_NAMES  = POP( ATTRIBUTE_NAMES  )$QUEUE
                ATTRIBUTE_VALUES = POP( ATTRIBUTE_VALUES )$QUEUE
            }
        }
    }
    NEWLINE(1)
}
# ######################################################################################################


# ######################################################################################################
GET_RULE_SCOPE = function( ATTR_EXPRESSION, debug=FALSE ) {
    RULE_ACTIVATED_ROWS = c() 
    if ( nrow(X) == 0 ) return(RULE_ACTIVATED_ROWS)

    print( ATTR_EXPRESSION )
    RULE_ACTIVATED_ROWS = eval( parse( text=sprintf("rownames(X[(%s),])",ATTR_EXPRESSION) ) )
    if ( length(RULE_ACTIVATED_ROWS) == 0 ) return(RULE_ACTIVATED_ROWS)

    cat(HEADER)
        print( ATTR_EXPRESSION )
        print( sprintf( "X, Y UPDATED [nrow(X)=%s] DROPPING %s ROWS: %s", nrow(X), length(RULE_ACTIVATED_ROWS), CONCAT(RULE_ACTIVATED_ROWS)))
    cat(HEADER)

    RULE_ACTIVATED_ROWS = sapply( RULE_ACTIVATED_ROWS, as.character)

    print( summary( MAPPING[RULE_ACTIVATED_ROWS,] ) )

    return( RULE_ACTIVATED_ROWS )
}
# ######################################################################################################


# ######################################################################################################
WHICH_ROWS_TO_KEEP = function( RULE_ACTIVATED_ROWS=c(), ATTR_EXPRESSION="", debug=FALSE ) {
    CURRENT_ROWNAMES    = sapply( rownames(X),         as.character)

    if ( length(RULE_ACTIVATED_ROWS) == 0 )
        RULE_ACTIVATED_ROWS = GET_RULE_SCOPE( ATTR_EXPRESSION )

    WHICH_ROWS1 = setdiff( CURRENT_ROWNAMES, RULE_ACTIVATED_ROWS )

    WHICH_ROWS2 = CURRENT_ROWNAMES[WHICH_ROWS1]
    if ( length(setdiff( WHICH_ROWS1, WHICH_ROWS2 ))!= 0 ) {
        cat(HEADER)
        print( paste( "S0", CONCAT(WHICH_ROWS1) ) )
        cat(HEADER)
        print( paste( "S1", CONCAT(WHICH_ROWS2) ) )
        cat(HEADER)
    }

    return( WHICH_ROWS2 )
}
# ######################################################################################################

    
# ######################################################################################################
DO_UPDATE = function( WHICH_ROWS_TO_KEEP, debug=FALSE ) {
    XX = X[WHICH_ROWS_TO_KEEP,]
    YY = Y[WHICH_ROWS_TO_KEEP,]

    X <<- XX
    Y <<- YY

    n = nrow(XX)
    if( n == 0 ) {
        print( "DONE. NO MORE ROWS" )
        return(TRUE)
    } else {
        W = complete.cases(XX)
        XX = XX[W,]
        YY = YY[W,]
        X <<- XX
        Y <<- YY
    }

    print( paste( 'CURRENTLY, MISSING RULE COVERAGE FOR', GET_SIZE(X), 'SAMPLES OUT OF', GET_SIZE(MAPPING) ) )

    if ( debug ) {
        cat( HEADER )
        print( summary( cbind(X,Y)) )
        cat( HEADER )
        print( cbind(XX,YY) )
        cat(HEADER)
        NEWLINE(1)
    }
    return(FALSE)
}
# ######################################################################################################


# ######################################################################################################
TERMINATE = function( Q ) {
    END = FALSE
    if ( nrow(X) == 0 )       END = TRUE
    if ( all(MAPPING != 0 ))  END = TRUE
    if ( length(Q) == 0 )     END = TRUE
    return ( END )
}
# ######################################################################################################


# ######################################################################################################
# ACTUALLY IS NOT A FORMAL DECISION TREE NOT ID3, IN DEVELOPMENT
# ######################################################################################################
DO_TRIVIAL_ID3 = function( X, Y, WRITE_THRESHOLD=RULE_ERROR_RATE, debug=FALSE ) {
    cat(HEADER)
    print( summary( cbind(X, Y ) ) )
    cat(HEADER)

    RETVALS   = GET_ENTROPY_VECTOR_FOR( X, Y )
    ENTROPIES = RETVALS$ENTROPIES
    RETVALS   = WHICH_NEXT( ENTROPIES )
    Frx       = GET_MARGINALS_WRT( X=X, Y=Y )

    REMAINING_QUEUE = ENTROPIES[,]
    
    ITER = 0
    while( !TERMINATE(REMAINING_QUEUE) ) {
        ITER = ITER + 1

        ATTR_COLNUM = WHICH_NEXT( REMAINING_QUEUE )

        NEWLINE(5)
        print( sprintf( "ITERATION %3s, HANDLING ATTRIBUTE: %s", ITER, names(REMAINING_QUEUE)[ATTR_COLNUM] ) )
        print( sprintf( "AT ITERATION %3s,    ROWS REMAINING = %s SUMMARIZED AS FOLLOWS:", ITER, GET_SIZE(MAPPING[MAPPING$MAPPING==0,]) ) )
        print( summary(MAPPING[MAPPING$MAPPING==0,]) )
        cat(HEADER)

        WRITE_RULE( WHICH_COLNUM=ATTR_COLNUM, WRITE_THRESHOLD = RULE_ERROR_RATE )

        print( sprintf( "AFTER ITERATION %3s, ROWS REMAINING = %s SUMMARIZED AS FOLLOWS:", ITER, GET_SIZE(MAPPING[MAPPING$MAPPING==0,]) ) )
        print( summary(MAPPING[MAPPING$MAPPING==0,]) )
        cat(HEADER)

        REMAINING_QUEUE = REMAINING_QUEUE[-ATTR_COLNUM]

        if ( debug ) { print( REMAINING_QUEUE ); print( MAPPING ) }
    }

    ITEMIZE_MAPPINGS()

    SUMMARIZE_GENERATED_RULES()
}
# ######################################################################################################


# ######################################################################################################
ITEMIZE_MAPPINGS = function( ) {
    cat( HEADER )
    cat( HEADER )
    print( MAPPING )
    cat( HEADER )
    cat( HEADER )
    NEWLINE(5)
}
# ######################################################################################################


# ######################################################################################################
SUMMARIZE_GENERATED_RULES = function() {
    cat( HEADER )
    print( 'TRAINING MODEL GENERATED' )
    cat( HEADER )
    for ( RULE in PARSED_RULES ) {
        print( RULE$LHS)
        print( RULE$RHS)
        cat(HEADER)
    }
    cat( HEADER )
    cat( HEADER )
    NEWLINE(5)
}


# ######################################################################################################
PREDICT_USING_RULES = function( MAPPING ) {
    print( 'PREDICTING' )
    for ( RULE in PARSED_RULES ) {
        ATTR_EXPRESSION = RULE$EXPRESSION
        RULE_ACTIVATED_ROWS = GET_RULE_SCOPE( ATTR_EXPRESSION )
        RULE_ACTIVATED_ROWS = names(RULE_ACTIVATED_ROWS)
        MAPPING[RULE_ACTIVATED_ROWS,ncol(MAPPING)] = RULES[[ATTR_EXPRESSION]]
        X = X[which(!(rownames(X) %in% RULE_ACTIVATED_ROWS)),]
    }
    cat(HEADER)
    print( paste( 'MISSING RULE COVERAGE FOR', GET_SIZE(X), 'SAMPLES OUT OF', GET_SIZE(MAPPING) ) )
    print( summary( X ) )
    cat(HEADER)
    return( MAPPING )
}
# ######################################################################################################


# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################






















# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################


# ######################################################################################################
TEST_ENABLED = TRUE
# ######################################################################################################


# ######################################################################################################
if ( TEST_ENABLED ) {
    sink('output.marginals.out', split=TRUE)
    opts = options( width=160 )


    # ##################################################################################################
    golf        = read.csv( 'golf.txt', sep="\t", header=TRUE, stringsAsFactors=TRUE )
    YCLASSNAME  = "PlayGolf"

    RULES             = list()
    PARSED_RULES      = list()
    RULE_ERROR_RATE   = 0.05

    X           = golf[,c(1,3,5,6)]
    Y           = cbind( golf[7], as.factor(1:nrow(X)))
    colnames(Y) = c(colnames(Y)[1], "ORDERING" )

    MAPPING           = cbind( X, Y, VECTOR( nrow(X), initval="" ) )
    colnames(MAPPING) = c(colnames(X), colnames(Y), "MAPPING")

    DO_TRIVIAL_ID3( X, Y, WRITE_THRESHOLD=RULE_ERROR_RATE )
    NEWLINE(20)
    # ##################################################################################################


    # ##################################################################################################
    YCLASSNAME     = "CarEvaluation"
    ORIGINAL_XY    = READ_C45_DATA( 'car' )
    M              = nrow( ORIGINAL_XY )
    ORDERING       = sample(  rownames(ORIGINAL_XY), M )

    TEST_SIZE      = as.integer(M * 0.3)
    TRAIN_SIZE     = M - TEST_SIZE

    TRAIN_ORDERING = ORDERING[1:TRAIN_SIZE]

    TEST_ORDERING  = ORDERING[ (TRAIN_SIZE+1):M]

    RULES          = list()
    PARSED_RULES   = list()
    RULE_ERROR_RATE= 0.05

    # ##################################################################################################
    # TRAIN
    # ##################################################################################################
    XY         = ORIGINAL_XY[TRAIN_ORDERING,]

        X              = SLICE_DATAFRAME(XY, c(1:6)) 
        Y              = SLICE_DATAFRAME(XY, 7)
        Y              = cbind( Y, TRAIN_ORDERING)
        colnames(Y)    = c(colnames(Y)[1], "ORDERING" )

        MAPPING        = cbind( X, Y, as.matrix(rep(0.0, nrow(X) )))
        colnames(MAPPING) = c(colnames(X), colnames(Y), "MAPPING")

        DO_TRIVIAL_ID3( X, Y, WRITE_THRESHOLD=RULE_ERROR_RATE )

    # ##################################################################################################
    # TEST
    # ##################################################################################################
    PREDICT = TRUE
    if ( PREDICT ) {
        XY         = ORIGINAL_XY[TEST_ORDERING,]
        X              = SLICE_DATAFRAME(XY, c(1:6)) 
        Y              = SLICE_DATAFRAME(XY, 7)
        Y              = cbind( Y, TEST_ORDERING)
        colnames(Y)    = c(colnames(Y)[1], "ORDERING" )

        MAPPING        = cbind( X, Y, as.matrix(rep(0.0, nrow(X) )))
        colnames(MAPPING) = c(colnames(X), colnames(Y), "MAPPING")

        PRED_MAPPING   = PREDICT_USING_RULES( MAPPING )
    }

    SUMMARIZE_GENERATED_RULES()

sink()
options( opts )
}
# ######################################################################################################

