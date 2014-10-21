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
    WHICH = which( HVALS == min( HVALS, na.rm=TRUE ))
    if ( length(WHICH) > 0 ) WHICH=WHICH[1]
    if ( debug ) {
        print ( HVALS )
        print( sprintf( 'PICKED: %20s %.4f', rownames(HVALS)[WHICH], HVALS[WHICH] ) )
        cat( HEADER )
    }
    return ( WHICH )
}
# ######################################################################################################


# ######################################################################################################
GET_ENTROPY_VECTOR_FOR = function( X, Y, debug=FALSE ) {
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

    Frx = GET_MARGINALS_WRT( X=X[,MIN], Y=Y )

    retvals = list( 'ENTROPIES'=HVALS, 'ATTR_NAMES'=ATTRIBUTE_NAMES, 'IG_ATTRIBUTE'=MIN, 'FREQUENCIES'=Frx )

    return ( retvals )
}
# ######################################################################################################


# ######################################################################################################
APPLY_CONDITIONAL_FUNCTION = function( CX, CY, ATTR_EXPRESSION, WRT_COLUMN=1, BEING_EQUAL_TO="", APPLY_FUNCTION=GET_ENTROPY_VECTOR_FOR, debug=FALSE ) {
    BEING_EQUAL_TO = paste("^",BEING_EQUAL_TO,sep="")
    LN = grep( BEING_EQUAL_TO, levels(CX[,WRT_COLUMN]) ) # LN = which( str_detect(levels(CX[,WRT_COLUMN]), BEING_EQUAL_TO ))
    FACTOR_NAME = levels(CX[,WRT_COLUMN])[LN] 
    RULE_ACTIVATED_ROWS = GET_RULE_SCOPE(ATTR_EXPRESSION, SPECULATIVE=TRUE)

    CONDITIONED_SELECTION = CX[,WRT_COLUMN]==FACTOR_NAME
    CONDITIONED_X = CX[ CONDITIONED_SELECTION, ]
    CONDITIONED_Y = CY[ CONDITIONED_SELECTION, ]

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
WRITE_SPECIFICATION_FOR_RULE_FOR = function( ATTRIBUTE_EXPRESSION, TARGET_FACTOR_LEVEL, Frx, APPROX=FALSE ) {
    M = nrow(Frx)-1
    N = ncol(Frx)-1

    YLABEL = which(    Frx[TARGET_FACTOR_LEVEL,1:N] == max(Frx[TARGET_FACTOR_LEVEL,1:N]))

    print( Frx )

    FREQUENCY = Frx[TARGET_FACTOR_LEVEL, YLABEL]

    DISCLAIMER = ""
    if ( APPROX ) DISCLAIMER = " APPROXIMATELY, "

    ACCURACY = FREQUENCY/Frx[TARGET_FACTOR_LEVEL,'Sum'] * 100.0

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
IS_LEAF_NODE = function( ENTROPY, ACCURACY, WRITE_THRESHOLD=RULE_ERROR_RATE ) {
    val = FALSE
    if (ACCURACY/100 >= (1 - WRITE_THRESHOLD ))
        if ( ENTROPY <= 1*WRITE_THRESHOLD ) 
            val = TRUE
    if ( val ) {
        print( "TERMINAL NODE" )
        cat( HEADER )
    }
    return ( val )
}
# ######################################################################################################


# ######################################################################################################
TOLERATE_AS_LEAF_NODE = function( ENTROPY, ACCURACY, SUBTREE_SIZE, WRITE_THRESHOLD=RULE_ERROR_RATE, MIN_SUBTREE_SIZE=16 ) {
    val = FALSE
    if (ACCURACY/100 >= (1 - WRITE_THRESHOLD ))
        if ( (ENTROPY <= 1.25*WRITE_THRESHOLD)  )
            if ( SUBTREE_SIZE > MIN_SUBTREE_SIZE )
                val = TRUE
    if ( val ) print( "TOLERATE_AS_LEAF_NODE" )
    return ( val )
}
# ######################################################################################################


# ######################################################################################################
GET_FACTOR_ACCURACY = function( FACTOR_LEVEL, Frx, debug=FALSE ) {
    N = ncol(Frx)-1
    YLABEL = which(    Frx[FACTOR_LEVEL,1:N] == max(Frx[FACTOR_LEVEL,1:N]))
    FREQUENCY = Frx[FACTOR_LEVEL, YLABEL]
    ACCURACY = FREQUENCY/Frx[FACTOR_LEVEL,'Sum'] * 100.0
    if (length(ACCURACY) > 1) {
        if ( debug ) {
            print( ACCURACY )
            print( Frx )
            print( FACTOR_LEVEL )
        }
        ACCURACY = ACCURACY[1]
    }
    return( ACCURACY )
}
# ######################################################################################################


# ######################################################################################################
DO_RULE_WRITING = function( ENTROPY, ATTR_EXPRESSION, FACTOR_LEVEL, SUBTREE_SIZE, Frx, M, N, ORDER ) {
    ACCURACY = GET_FACTOR_ACCURACY ( FACTOR_LEVEL, Frx ) 

    print( sprintf( "ENTROPY=%.3f, WRT=%32s, FACTOR=%6s TREESIZE=%3s ACCURACY=%.1f", 
                     ENTROPY, ATTR_EXPRESSION, FACTOR_LEVEL, SUBTREE_SIZE, ACCURACY ) )

    RULE_HAS_COMPLETE_COVERAGE = FALSE
        if ( IS_LEAF_NODE( ENTROPY, ACCURACY, WRITE_THRESHOLD=RULE_ERROR_RATE ) ) {
            RULE_RETVALS = WRITE_SPECIFICATION_FOR_RULE_FOR( ATTR_EXPRESSION, FACTOR_LEVEL, Frx )
            RULE_HAS_COMPLETE_COVERAGE = TRUE
        }
        else if ( TOLERATE_AS_LEAF_NODE( ENTROPY, ACCURACY, SUBTREE_SIZE ) ) {
            RULE_RETVALS = WRITE_SPECIFICATION_FOR_RULE_FOR( ATTR_EXPRESSION, FACTOR_LEVEL, Frx, APPROX=TRUE )
            RULE_HAS_COMPLETE_COVERAGE = TRUE
        }

    print( paste( "DOES AN ORDER", ORDER, "RULE_HAS_COMPLETE_COVERAGE?", ATTR_EXPRESSION, RULE_HAS_COMPLETE_COVERAGE  ) )  

    return ( RULE_HAS_COMPLETE_COVERAGE )
}
# ######################################################################################################


# ######################################################################################################
WRITE_RULE = function( WHICH_COLNUM=1, x=0, y=0, WRITE_THRESHOLD = RULE_ERROR_RATE, MINNODESIZE=4, debug=FALSE ) {

    if ( nrow(X) == 0 ) {
        print( "DONE. NO MORE ROWS TO SCAN" )
        return()
    }
    FREQS_FACTORS2CLASS = GET_MARGINALS_WRT( X=X, Y=Y, x=WHICH_COLNUM, y=y, debug=debug )
    FACTOR_ENTROPIES    = GET_FACTOR_PURITY( FREQS_FACTORS2CLASS )
    ATTRIBUTE_NAME      = colnames(X)[WHICH_COLNUM]

    M = nrow(FREQS_FACTORS2CLASS)-1
    N = ncol(FREQS_FACTORS2CLASS)-1
    for (i in 1:M) {
        NEWLINE(1)
        ATTRIBUTE_NAMES  = c()
        ATTRIBUTE_VALUES = c()

        ENTROPY       = FACTOR_ENTROPIES[i]
        FACTOR_LEVEL  = names(FACTOR_ENTROPIES)[i]
        SUBTREE_SIZE  = FREQS_FACTORS2CLASS[FACTOR_LEVEL, 'Sum']
        if ( debug ) print ( paste( ATTRIBUTE_NAME, ENTROPY, FACTOR_LEVEL, SUBTREE_SIZE )  )

        ATTRIBUTE_NAMES    = PUSH( ATTRIBUTE_NAMES,  ATTRIBUTE_NAME )
        ATTRIBUTE_VALUES   = PUSH( ATTRIBUTE_VALUES, FACTOR_LEVEL )
        ATTR_EXPRESSION    = GET_ATTRIBUTE_EXPRESSION( ATTRIBUTE_NAMES, ATTRIBUTE_VALUES )

        print( paste( "DOING FIRST ORDER EXPANSION, ADDRESSING NOW", CONCAT(ATTRIBUTE_NAMES), CONCAT(ATTRIBUTE_VALUES) ) )

        RULE_HAS_COMPLETE_COVERAGE = DO_RULE_WRITING( ENTROPY, ATTR_EXPRESSION, FACTOR_LEVEL, SUBTREE_SIZE, FREQS_FACTORS2CLASS, ORDER=1 )

        if ( RULE_HAS_COMPLETE_COVERAGE ) {
            WHICH_ROWS_TO_KEEP = UPDATE_MAPPING_WITH_RULE( ATTR_EXPRESSION )
            DO_UPDATE(WHICH_ROWS_TO_KEEP, ORDER=1)
        } else {
            DO_SECOND_ORDER_EXPANSION ( X, Y, WHICH_COLNUM, FACTOR_LEVEL, ATTRIBUTE_NAMES, ATTRIBUTE_VALUES )
        }
    }
    NEWLINE(1)
}
# ######################################################################################################


# ######################################################################################################
DO_SECOND_ORDER_EXPANSION = function( X, Y, WHICH_COLNUM, FACTOR_LEVEL, ATTRIBUTE_NAMES, ATTRIBUTE_VALUES, debug=FALSE ) {
    print( paste( "DOING SECOND ORDER EXPANSION, ADDRESSING NOW", FACTOR_LEVEL, CONCAT(ATTRIBUTE_NAMES), CONCAT(ATTRIBUTE_VALUES) ) )

    ATTR_EXPRESSION = GET_ATTRIBUTE_EXPRESSION( ATTRIBUTE_NAMES, ATTRIBUTE_VALUES )
    SO_RETVALS      = APPLY_CONDITIONAL_FUNCTION( X, Y, 
                                                    ATTR_EXPRESSION, 
                                                    WRT_COLUMN=WHICH_COLNUM, 
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
        ENTROPY       = SO_FACTOR_ENTROPIES[i]
        FACTOR_LEVEL  = names(SO_FACTOR_ENTROPIES)[i]
        SUBTREE_SIZE  = SO_FREQUENCIES[FACTOR_LEVEL, 'Sum']
        ATTRIBUTE_NAMES  = PUSH( ATTRIBUTE_NAMES,  colnames(X)[SO_IG_ATTRIBUTE] )
        ATTRIBUTE_VALUES = PUSH( ATTRIBUTE_VALUES, FACTOR_LEVEL )

        ATTR_EXPRESSION  = GET_ATTRIBUTE_EXPRESSION( ATTRIBUTE_NAMES, ATTRIBUTE_VALUES )
        RULE_HAS_COMPLETE_COVERAGE = DO_RULE_WRITING( ENTROPY, ATTR_EXPRESSION, FACTOR_LEVEL, SUBTREE_SIZE, SO_FREQUENCIES, ORDER=2 )

        if ( debug ) print( paste( "DOES RULE HAS COMPLETE COVERAGE?", RULE_HAS_COMPLETE_COVERAGE,  ATTR_EXPRESSION ) )

        if ( RULE_HAS_COMPLETE_COVERAGE ) { 
            WHICH_ROWS_TO_KEEP = UPDATE_MAPPING_WITH_RULE( ATTR_EXPRESSION )
            DO_UPDATE(WHICH_ROWS_TO_KEEP, ORDER=2)
        } else  {
            # TODO: THE ABOVE NEEDS TO BECOME A LOOP TO EXPAND TO 3RD ORDER AND SO FORTH
            # TODO: CONSIDER: BY RANDOMLY SELECTING WHICH ATTRIBUTES TO EXAMINE AS OPPOSED TO ALL --> RANDOM SUBTREE (TO REDUCE TIME, 
            SETUP_RECURSION( X, Y, ATTR_EXPRESSION, ATTRIBUTE_NAMES, ATTRIBUTE_VALUES )
        }
        ATTRIBUTE_NAMES  = POP( ATTRIBUTE_NAMES  )$QUEUE
        ATTRIBUTE_VALUES = POP( ATTRIBUTE_VALUES )$QUEUE
    }
}
# ######################################################################################################


# ######################################################################################################
SETUP_RECURSION = function( CX, XY, ATTR_EXPRESSION, ATTRIBUTE_NAMES, ATTRIBUTE_VALUES, debug=FALSE ) {
    TARGET_LOOKAHEAD_COLUMNS = which(  colnames(X) %in% setdiff( colnames(X), ATTRIBUTE_NAMES ))
    if( length(TARGET_LOOKAHEAD_COLUMNS)!=0 ) {
        IN_USE_ATTRIBUTES = which(colnames(X) %in% ATTRIBUTE_NAMES )
        RULE_ACTIVATED_ROWS = GET_RULE_SCOPE(ATTR_EXPRESSION, SPECULATIVE=TRUE)

        OPERATE_ON_X = X[RULE_ACTIVATED_ROWS,-IN_USE_ATTRIBUTES]
        OPERATE_ON_Y = Y[RULE_ACTIVATED_ROWS,] 

        if ( debug ) print( cbind( OPERATE_ON_X, OPERATE_ON_Y ) )

        RULE_HAS_COMPLETE_COVERAGE = DO_THIRD_ORDER_EXPANSION( OPERATE_ON_X, 
                                                               OPERATE_ON_Y, 
                                                               ATTRIBUTE_NAMES, 
                                                               ATTRIBUTE_VALUES )
    }
    return ( RULE_HAS_COMPLETE_COVERAGE )
}
# ######################################################################################################


# ######################################################################################################
DO_THIRD_ORDER_EXPANSION = function( CX, CY, ATTRIBUTE_NAMES, ATTRIBUTE_VALUES, debug=FALSE ) {
    TO_RETVALS       = GET_ENTROPY_VECTOR_FOR( CX, CY )
    if ( debug ) { 
        cat( HEADER); print( TO_RETVALS$FREQUENCIES ) ; cat( HEADER) 
    }

    TO_ENTROPIES    = TO_RETVALS$ENTROPIES                # entropy of all the attributes
    TO_ATTR_NAMES   = TO_RETVALS$ATTR_NAMES               # column names
    TO_ATTR_VALUE   = WHICH_NEXT( TO_ENTROPIES )          # column number for attribute with optimal information gain
    TO_FREQUENCIES  = TO_RETVALS$FREQUENCIES              # frequency table for this attribute wrt y conditioned to above
    TO_IG_ATTRIBUTE = TO_RETVALS$IG_ATTRIBUTE             # optimal attribute
    if (length( TO_IG_ATTRIBUTE) >0 )
        TO_IG_ATTRIBUTE = TO_RETVALS$IG_ATTRIBUTE[1]      # optimal attribute

    TO_FACTOR_ENTROPIES = GET_FACTOR_PURITY( TO_FREQUENCIES )

    for ( i in 1:length(TO_FACTOR_ENTROPIES)) {
        ENTROPY       = TO_FACTOR_ENTROPIES[i]
        FACTOR_LEVEL  = names(TO_FACTOR_ENTROPIES)[i]
        SUBTREE_SIZE  = TO_FREQUENCIES[FACTOR_LEVEL, 'Sum']
        print( paste( "EVALUATING THIRD ORDER EXPANSION, ADDRESSING NOW", colnames(CX)[TO_IG_ATTRIBUTE], FACTOR_LEVEL ) )

        ATTRIBUTE_NAMES  = PUSH( ATTRIBUTE_NAMES,  colnames(CX)[TO_IG_ATTRIBUTE] )
        ATTRIBUTE_VALUES = PUSH( ATTRIBUTE_VALUES, FACTOR_LEVEL )

            ATTR_EXPRESSION  = GET_ATTRIBUTE_EXPRESSION( ATTRIBUTE_NAMES, ATTRIBUTE_VALUES )

            RULE_HAS_COMPLETE_COVERAGE = DO_RULE_WRITING( ENTROPY, ATTR_EXPRESSION, FACTOR_LEVEL, SUBTREE_SIZE, TO_FREQUENCIES, ORDER=3 )
            if ( debug ) print( paste( "DOES RULE HAS COMPLETE COVERAGE?", RULE_HAS_COMPLETE_COVERAGE,  ATTR_EXPRESSION ) )
    
            if ( RULE_HAS_COMPLETE_COVERAGE ) { 
                WHICH_ROWS_TO_KEEP = UPDATE_MAPPING_WITH_RULE( ATTR_EXPRESSION, debug=FALSE )
                DO_UPDATE(WHICH_ROWS_TO_KEEP, ORDER=3)
            } else {
                print( 'AT THIS TIME, SEARCH LEVEL EXPLORATION SEEKS TO STOP AT 3RD ORDER' )
                cat(HEADER)
            }
        ATTRIBUTE_NAMES  = POP( ATTRIBUTE_NAMES  )$QUEUE
        ATTRIBUTE_VALUES = POP( ATTRIBUTE_VALUES )$QUEUE
    }

    return ( RULE_HAS_COMPLETE_COVERAGE )
}
# ######################################################################################################


# ######################################################################################################
UPDATE_MAPPING_WITH_RULE = function( ATTR_EXPRESSION, debug=FALSE ) {
    RULE_ACTIVATED_ROWS = GET_RULE_SCOPE(ATTR_EXPRESSION)

    ALREADY_ALLOCATED   = which( MAPPING[RULE_ACTIVATED_ROWS,ncol(MAPPING)] != MAPPING_INIT_VAL )
    if ( length(ALREADY_ALLOCATED)!= 0 )
        print( paste( length(ALREADY_ALLOCATED), "ALREADY ALLOCATED TO RULE WITH HIGHER PRECEDENCE", CONCAT( ALREADY_ALLOCATED) ) )

    if ( length(RULES[[ATTR_EXPRESSION]] ) > 1 ) 
        print( RULES[[ATTR_EXPRESSION]] )

    # in case of ties, multiple rules are possible to bubble up
    MAPPING[RULE_ACTIVATED_ROWS,ncol(MAPPING)] <<- RULES[[ATTR_EXPRESSION]][1]

    if ( debug ) 
        print( MAPPING[RULE_ACTIVATED_ROWS,] )
 
    WHICH_ROWS_TO_KEEP  = WHICH_ROWS_TO_KEEP( RULE_ACTIVATED_ROWS=RULE_ACTIVATED_ROWS )
 
    return( WHICH_ROWS_TO_KEEP )
}
# ######################################################################################################


# ######################################################################################################
GET_RULE_SCOPE = function( ATTR_EXPRESSION, SPECULATIVE=FALSE, debug=FALSE ) {
    RULE_ACTIVATED_ROWS = c() 
    if ( nrow(X) == 0 ) return(RULE_ACTIVATED_ROWS)

    RULE_ACTIVATED_ROWS = eval( parse( text=sprintf("rownames(X[(%s),])",ATTR_EXPRESSION) ) )
    if ( length(RULE_ACTIVATED_ROWS) == 0 ) return(RULE_ACTIVATED_ROWS)

    cat(HEADER)
        print( ATTR_EXPRESSION )
        if ( SPECULATIVE )
            print( sprintf( "X, Y ANALYZING [nrow(X)=%s] WRT SCOPING OF %s ROWS: %s", nrow(X), length(RULE_ACTIVATED_ROWS), CONCAT(RULE_ACTIVATED_ROWS)))
        else
            print( sprintf( "X, Y UPDATING  [nrow(X)=%s] BY DROPPING %s ROWS: %s", nrow(X), length(RULE_ACTIVATED_ROWS), CONCAT(RULE_ACTIVATED_ROWS)))
    cat(HEADER)

    RULE_ACTIVATED_ROWS = sapply( RULE_ACTIVATED_ROWS, as.character)

    cat(HEADER)
    print( summary( MAPPING[RULE_ACTIVATED_ROWS,] ) )
    cat(HEADER)

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
DO_UPDATE = function( WHICH_ROWS_TO_KEEP, ORDER=0, debug=FALSE ) {
    OLD_N = GET_SIZE(X)

    XX = X[WHICH_ROWS_TO_KEEP,]
    YY = Y[WHICH_ROWS_TO_KEEP,]

    X <<- XX
    Y <<- YY

    N = nrow(XX)
    if( N == 0 ) {
        print( "DONE. NO MORE ROWS" )
        return(TRUE)
    } else {
        W = complete.cases(XX)
        XX = XX[W,]
        YY = YY[W,]
        X <<- XX
        Y <<- YY
    }

    UPDATE_STATUS( OLD_N, ORDER )

    if ( debug ) {
        cat( HEADER )
        print( summary( cbind(X,Y)) )
        cat( HEADER )
        print( cbind(XX,YY) )
        cat(HEADER)
        NEWLINE(1)
    }

    return (FALSE)
}
# ######################################################################################################


# ######################################################################################################
UPDATE_STATUS = function( N, ORDER ) {
    cat(HEADER)
    print( paste( 'STATUS: CURRENTLY MISSING COVERAGE FOR', GET_SIZE(X), 'SAMPLES OUT OF', GET_SIZE(MAPPING) ) )
    STATUS[[ GET_SIZE(STATUS)+1 ]] <<- c(GET_SIZE(STATUS)+1, GET_SIZE(X), N-GET_SIZE(X), ORDER )
    cat(HEADER)
    cat(HEADER)
    NEWLINE(1)
}
# ######################################################################################################


# ######################################################################################################
TERMINATE = function( Q ) {
    END = FALSE
    if ( nrow(X) == 0 )                      END = TRUE
    if ( length(Q) == 0 )                    END = TRUE
    if ( all(MAPPING != MAPPING_INIT_VAL ))  END = TRUE
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
        print( sprintf( "AT ITERATION %3s,    ROWS REMAINING = %s", ITER, GET_SIZE(MAPPING[MAPPING$MAPPING==0,]) ) )
        if ( debug ) {
            print( summary(MAPPING[MAPPING$MAPPING==0,]) )
            cat(HEADER)
            cat(HEADER)
        }

        WRITE_RULE( WHICH_COLNUM=ATTR_COLNUM, WRITE_THRESHOLD = RULE_ERROR_RATE )

        if ( debug ) {
            print( sprintf( "AFTER ITERATION %3s, ROWS REMAINING = %s SUMMARIZED AS FOLLOWS:", ITER, GET_SIZE(MAPPING[MAPPING$MAPPING==0,]) ) )
            print( summary(MAPPING[MAPPING$MAPPING==0,]) )
            cat(HEADER)
            cat(HEADER)
        }

        REMAINING_QUEUE = REMAINING_QUEUE[-ATTR_COLNUM]

        if ( debug ) { print( REMAINING_QUEUE ); print( MAPPING ) }
    }

}
# ######################################################################################################


# ######################################################################################################
ITEMIZE_MAPPINGS = function( ) {
    cat( HEADER )
    cat( HEADER )
    print( MAPPING[,(ncol(MAPPING)-1):ncol(MAPPING)] )
    cat( HEADER )
    cat( HEADER )
    NEWLINE(5)
}
# ######################################################################################################


# ######################################################################################################
SUMMARIZE_GENERATED_RULES = function() {
    NEWLINE(5)
    cat( HEADER )
    cat( HEADER )
    print( 'RULE-BASED TRAINING MODEL GENERATED' )
    cat( HEADER )
    i = 0
    for ( RULE in PARSED_RULES ) {
        i = i + 1
        print( paste( i, RULE$LHS) )
        print( RULE$RHS)
        cat(HEADER)
    }
    STATUS = t(sapply( STATUS, function(x) c(x[1], x[2], x[3], x[4]) ))
    print( "PDF file plot_marginals_decision_tree_performance.pdf was generated depicting performance of the rule search across iterations" )

    pdf( 'plot_marginals_decision_tree_performance.pdf' )
    par(mfrow=c(3,1))
    delta = 0
    plot(STATUS[,1], STATUS[,2], main="REMAINING SAMPLES MISSING RULE COVERAGE", t="l", cex=1, col="black", pch="o", 
         xlab="I-TH ITERATION", ylab="NUM SAMPLES LEFT" )
    text(STATUS[,1]+0.5, STATUS[,2]+delta, STATUS[,4], cex=1.0, col=STATUS[,4] )

    delta3 = 0 
    plot(STATUS[,1], STATUS[,3], main="INCREASE COVERAGE OF THE I-TH RULE", t="l", cex=1, col="black", pch="o", 
         xlab="I-TH ITERATION", ylab="SCOPE OF THE RULE (# SAMPLES" )
    text(STATUS[,1]+0.5, STATUS[,3]+delta3, STATUS[,4], cex=1.0, col=STATUS[,4] )

    delta4 = 0 
    plot(log(STATUS[,3]), STATUS[,4], main="ORDER OF THE I-TH RULE VS INCREASED COVERAGE", t="p", cex=0.3, col="black", pch="o",
         xlab="log(INCREASE)", ylab="ORDER OF THE RULE", ylim=c(0,max(STATUS[,4])+1 ) )
    text(log(STATUS[,3])+delta3, STATUS[,4]+delta4, STATUS[,1], cex=1.0, col=STATUS[,4] )
    dev.off()

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

        print( RULES[[ATTR_EXPRESSION]] )

        # MAPPING[RULE_ACTIVATED_ROWS,ncol(MAPPING)] = RULES[[ATTR_EXPRESSION]]
        MAPPING[RULE_ACTIVATED_ROWS,ncol(MAPPING)] = RULES[[ATTR_EXPRESSION]][1]

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
MAPPING_INIT_VAL = 0.0
# ######################################################################################################

if ( TEST_ENABLED ) {
    # for(i in 1:10) sink()
    sink('output.marginals.out', split=TRUE)
    opts = options( width=160 )


    # ##################################################################################################
    GOLF        = read.csv( 'golf.txt', sep="\t", header=TRUE, stringsAsFactors=TRUE )
    YCLASSNAME  = "PlayGolf"

    # ##################################################################################################
    STATUS            = list()
    RULES             = list()
    PARSED_RULES      = list()

    RULE_ERROR_RATE   = 0.10
    # ##################################################################################################

    # ##################################################################################################
    X                 = GOLF[,c(1,3,5,6)]
    Y                 = cbind( GOLF[7], as.factor(1:nrow(X)))
    MAPPING           = cbind( X, Y, as.matrix(rep(MAPPING_INIT_VAL, nrow(X) )))
    colnames(MAPPING) = c(colnames(X), colnames(Y), "MAPPING")
    colnames(Y) = c(colnames(Y)[1], "ORDERING" )
    # ##################################################################################################

    DO_TRIVIAL_ID3( X, Y, WRITE_THRESHOLD=RULE_ERROR_RATE )
    ITEMIZE_MAPPINGS()
    SUMMARIZE_GENERATED_RULES()

    # ##################################################################################################


    NEWLINE(20)


    # ##################################################################################################
    YCLASSNAME     = "CarEvaluation"
    ORIGINAL_XY    = READ_C45_DATA( 'car' )
    M              = nrow( ORIGINAL_XY )
    ORDERING       = sample(  rownames(ORIGINAL_XY), M )

    # ##################################################################################################
    TEST_SIZE      = as.integer(M * 0.3)
    TRAIN_SIZE     = M - TEST_SIZE
    TRAIN_ORDERING = ORDERING[1:TRAIN_SIZE]
    TEST_ORDERING  = ORDERING[ (TRAIN_SIZE+1):M]
    # ##################################################################################################

    # ##################################################################################################
    STATUS         = list()
    RULES          = list()
    PARSED_RULES   = list()

    RULE_ERROR_RATE= 0.05
    # ##################################################################################################

    # ##################################################################################################
    # TRAIN
    # ##################################################################################################
    XY         = ORIGINAL_XY[TRAIN_ORDERING,]

        X              = SLICE_DATAFRAME(XY, c(1:6)) 
        Y              = SLICE_DATAFRAME(XY, 7)
        Y              = cbind( Y, TRAIN_ORDERING)
        colnames(Y)    = c(colnames(Y)[1], "ORDERING" )
        MAPPING        = cbind( X, Y, as.matrix(rep(MAPPING_INIT_VAL, nrow(X) )))
        colnames(MAPPING) = c(colnames(X), colnames(Y), "MAPPING")

        DO_TRIVIAL_ID3( X, Y, WRITE_THRESHOLD=RULE_ERROR_RATE )
        TRAIN_MAPPING  = MAPPING
        # ITEMIZE_MAPPINGS()
        # SUMMARIZE_GENERATED_RULES()

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

        TEST_MAPPING   = cbind( X, Y, as.matrix(rep(MAPPING_INIT_VAL, nrow(X) )))
        colnames(TEST_MAPPING) = c(colnames(X), colnames(Y), "MAPPING")

        TEST_MAPPING   = PREDICT_USING_RULES( TEST_MAPPING )
    }

    SUMMARIZE_GENERATED_RULES()

sink()
options( opts )
}
# ######################################################################################################

