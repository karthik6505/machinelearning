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
# http://blog.revolutionanalytics.com/2013/06/plotting-classification-and-regression-trees-with-plotrpart.html
# library(partykit)				# Convert rpart object to BinaryTree
# ######################################################################################################
library(rpart)				    # Popular decision tree algorithm
library(rattle)					# Fancy tree plot
library(rpart.plot)				# Enhanced tree plots
library(RColorBrewer)				# Color selection for fancy tree plot
library(party)					# Alternative decision tree algorithm
library(caret)					# Just a data source for this script # but probably one of the best R packages ever. 
# ######################################################################################################


# ######################################################################################################
source( 'utilities.R' )
source( 'marginals.R' )
# ######################################################################################################


# ######################################################################################################
FIND_COMMON_MINIMUM_INDEXES = function( H_LHS, H_RHS, EPSILON=0.0, debug=FALSE ) {
    LHS = which ( H_LHS <= ( min( H_LHS + EPSILON ) ) )
    RHS = which ( H_RHS <= ( min( H_RHS + EPSILON ) ) )
    COMMON   = intersect(LHS, RHS)
    if ( debug ) print( sprintf( "AT COMMON INDEX=%s, LHS=%.3f .vs. RHS=%.3f", COMMON, H_LHS[COMMON], H_RHS[COMMON]) )
    RETVALS = list( COMMON=COMMON, LHS=LHS, RHS=RHS, EPSILON=EPSILON )
    return( RETVALS )
}
# ######################################################################################################


# ######################################################################################################
FIND_MINIMUM_ENTROPY_CUT_VALUE = function( XX=c(), WRT_YY=c(), SILENT=FALSE, DO_PLOT=FALSE ) {
    N = GET_SIZE(XX)

    XX  = AS_DATAFRAME( XX )
    YY  = AS_DATAFRAME( WRT_YY )

    ORDERING  = order(XX[,1])
    XX  = XX[ORDERING,]
    YY  = YY[ORDERING,]

    H = sapply( 1:N, function(i)  c( GET_ENTROPY_XY_WRT( XX[1:i,    1],YY[1:i,    1] ),
                                     GET_ENTROPY_XY_WRT( XX[(i+1):N,1],YY[(i+1):N,1] )))
    H = t(H)
    rownames(H) = 1:N
    colnames(H) = c('LHS', 'RHS')

    RETVALS = FIND_COMMON_MINIMUM_INDEXES( H[,1], H[,2], EPSILON=0 )
    COMMON  = RETVALS$COMMON
    EPSILON = 0
    DELTA   = sd(H)/8 
    while( length(COMMON)== 0 ) {
        EPSILON = EPSILON + DELTA
        RETVALS = FIND_COMMON_MINIMUM_INDEXES( H[,1], H[,2], EPSILON=EPSILON )
        COMMON  = RETVALS$COMMON
    }
    LHS     = RETVALS$LHS
    RHS     = RETVALS$RHS
    ITSVALUE= XX[COMMON,1]

    if ( DO_PLOT ) {
        plot(   XX[,1] , H[,1], pch="o" )
        points( XX[,1] , H[,2], pch="+" )
        text( XX[LHS,1] , H[LHS,1], Y[LHS,1], col="black", cex=0.5 )
        text( XX[LHS,1] , H[LHS,2], Y[RHS,1], col="blue",  cex=0.5 )
        text( XX[COMMON,1] , H[COMMON,2], "COMMON", col="red",  cex=1.0 )
    }

    if ( !SILENT ) {
        XY = cbind( XX[,1], YY[,1] )
        colnames(XY) = c(colnames(XX)[1], colnames(YY)[1])
        print( summary( XX[XX[,1]<=ITSVALUE,1] ) )
        print( summary( XX[XX[,1]> ITSVALUE,1] ) )
        print( sprintf( "PRIOR PIVOT=%s => %s", ITSVALUE, 
                       substr(CONCAT( paste( XX[XX[,1]<=ITSVALUE,1], YY[XX[,1]<=ITSVALUE,1] )),1,120)))
        print( sprintf( "AFTER PIVOT=%s => %s", ITSVALUE, 
                       substr(CONCAT( paste( XX[XX[,1]>ITSVALUE,1],  YY[XX[,1]>ITSVALUE,1]  )),1,120)))
    }

    return ( ITSVALUE )
}
# ######################################################################################################


# ######################################################################################################
VERIFY_PIVOT = function( SELECTED_PIVOT, XX, debug=FALSE ) {
    A = table( cbind( XX, YY ))
    XVALS = sort( XX[,1] )
    ORDERING  = order(XX[,1])
    XX  = XX[ORDERING,]
    YY  = YY[ORDERING,]
    for( i in 1:(GET_SIZE(XX)-1) ) {
        PIVOT = XVALS[i]
        XXX1 = XX[XX<=PIVOT]
        # YYY1 = YY[XX<=PIVOT]
        YYY1 = A[,1][XX<=PIVOT]
        XXX2 = XX[XX> PIVOT]
        # YYY2 = YY[XX> PIVOT]
        YYY2 = A[,1][XX>PIVOT]
    
        METRIC = function( N1, N2, HHH1, HHH2 ) { HHH1/N1 + HHH2/N2 }
        HHH1 = GET_ENTROPY_XY_WRT( XXX1, YYY1 )
        HHH2 = GET_ENTROPY_XY_WRT( XXX2, YYY2 )
        N1   = GET_SIZE(XXX1)
        N2   = GET_SIZE(XXX2)
        V  = METRIC(N1, N2, HHH1, HHH2 )
        print( sprintf( "PIVOT=%s  TOTAL=%.3f   LHS=%.4f (%4s items)   RHS=%.4f (%4s items)", PIVOT, V, HHH1, N1, HHH2, N2 ) )
        if ( debug ) {
            print( paste( CONCAT(paste(XXX1, YYY1))))
            print( paste( CONCAT(paste(XXX2, YYY2))))
            cat( HEADER )
        }
    }
}
# ######################################################################################################


# ######################################################################################################
READ_C45_DATA = function( filename_stem ) {
    t = read.csv( paste(filename_stem,"data",sep="."), header=FALSE, stringsAsFactors=TRUE )
    h = read.csv2(paste(filename_stem,'c45-names', sep="."), sep=":")
    h = rownames(h)[c(-2,-3)]
    h = gsub("\\| ","", h )
    h = gsub(" ","_", h )
    h = c(h[2:length(h)], h[1])
    colnames(t) = h
    print( summary(t))
    return ( t )
}
# ######################################################################################################


# ######################################################################################################
SCORE = function( Y, YP, i, YLEVELS_SCORING=list( 'unacc'=0, 'acc' =1, 'good'=2, 'vgood'=3 ), silent=FALSE, debug=FALSE ) {
    y  = as.character(Y[i])
    yp = as.character(YP[i])
    s_y  = YLEVELS_SCORING[[y]]
    s_yp = YLEVELS_SCORING[[yp]]
    MSE = (s_yp - s_y )^2
    if ( debug ) 
        print( sprintf( "%5s [%12s vs. %12s]  ==>  [%3s]    IMSE=%.4g", i, y, yp, s_yp - s_y, MSE/i  ) )
    else if ( !silent & abs(s_yp - s_y) != 0 )
            print( sprintf( "%5s [%12s vs. %12s]  ==>  [%3s]    IMSE=%.4g", i, y, yp, s_yp - s_y, MSE/i  ) )
    return ( MSE )
}
# ######################################################################################################


# ######################################################################################################
# http://blog.revolutionanalytics.com/2013/06/plotting-classification-and-regression-trees-with-plotrpart.html
# ######################################################################################################
DO_DECISION_TREE = function( XY, Y, FORMULA="", DO_PRUNING=TRUE, Q=0.1, ... ) {
    if (nchar(FORMULA)==0) FORMULA = sprintf( "%s ~ .", colnames(Y) )
    DETAILED_MODEL   = rpart( as.formula(FORMULA), data=XY, ... )
    MODEL            = DETAILED_MODEL
    if ( DO_PRUNING ) {
        MODEL_CV         = printcp( DETAILED_MODEL )
        R_ERROR_THRESHOLD= quantile( MODEL_CV[,'rel error'], 0.10)
        Q = 0.10
        while( length(R_ERROR_THRESHOLD)== 0 ) {
            Q = Q + 0.01
            R_ERROR_THRESHOLD= quantile( MODEL_CV[,'rel error'], Q)
        }
        CP_CHOICES       = MODEL_CV[,'rel error'] < R_ERROR_THRESHOLD
        CP_PIVOT         = MODEL_CV[CP_CHOICES,'CP'][1]
        MODEL            = prune( DETAILED_MODEL, cp=CP_PIVOT )
    }
    NEWLINE(3)
    print( summary( MODEL ) )
    cat(HEADER)
    NEWLINE(3)
    printcp( MODEL )
    cat(HEADER)
    return ( MODEL )
}
# ######################################################################################################


# ######################################################################################################
# http://blog.revolutionanalytics.com/2013/06/plotting-classification-and-regression-trees-with-plotrpart.html
# ######################################################################################################
PLOT_TREE = function( MODEL, INCREMENTAL_MSE, CONFUSION_MATRIX ) {
    def.par <- par(no.readonly = TRUE) # save default, for resetting...

    nf <- layout(matrix(c(1,1,1,1,1,1,2,3,4), 3, 3, byrow = TRUE), respect = TRUE)
    prp(MODEL, type=4, nn=TRUE, cex=0.7, extra=2, under=TRUE, branch=1, main="RECURSIVE PARTITIONING TREE")#, varlen=6)# Shorten variable names
    plot( INCREMENTAL_MSE, t='l', main="INCREMENTAL MSE" )

    ACM = addmargins(CM)
    ACM = apply( ACM, 2, function(x) x/x[length(x)] )
    ACM = ACM[1:nrow(CM),1:ncol(CM)]

    colors = c("blue", "green", "brown", "red", "gray", "white", 20:40 )
    plot( CONFUSION_MATRIX, main="CONFUSION MATRIX (CM)", col=colors[1:max(ncol(CM),nrow(CM))] )
    plot( as.table(ACM), main="CLASS FREQUENCIES", col=colors[1:max(ncol(CM),nrow(CM))] )

    par( def.par )
    return ( ACM )
}
# ######################################################################################################


# ######################################################################################################
PREDICT_TREE = function( XY, YTRUE, as_probas=FALSE ) {
    YP_PRED   = predict(MODEL, XY)
    if ( !as_probas ) {
        YP_MATRIX = matrix(predict(MODEL), nrow(X))
        YP        = apply( YP_MATRIX, 1, function( x ) colnames(YP_PRED)[which(x == max(x))] )
        return ( YP )
    } else {
        return ( YP_PRED )
    }
}
# ######################################################################################################


# ######################################################################################################
GET_TREE_IMSE = function( Y, YP, total=FALSE ) {
    SSE  = sapply( 1:GET_SIZE(Y),   function(i) SCORE(Y, YP, i ) ) 
    CSSE = cumsum( SSE )
    IMSE = sapply( 1:GET_SIZE(SSE), function( i ) CSSE[i]/i )
    cat( HEADER )
    if ( total ) return ( sum(IMSE) )
    return ( IMSE )
}
# ######################################################################################################


# ######################################################################################################
GET_CONFUSION_MATRIX = function( Y, YP ) {
    CM = table( Y, YP )
    print ( CM )
    cat( HEADER )
    return ( CM )
}
# ######################################################################################################


# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
    opts = options( width=132 )
    golf = read.csv( 'golf.txt', sep="\t", header=TRUE, stringsAsFactors=TRUE )
    XX   = SLICE_DATAFRAME( golf, 4 )
    YY   = SLICE_DATAFRAME( golf, 7 )
    # ######################################################################################################
    # PIVOT = FIND_MINIMUM_ENTROPY_CUT_VALUE( XX, YY, SILENT=FALSE, DO_PLOT=TRUE )
    # VERIFY_PIVOT( PIVOT, XX, debug=FALSE )
    # A = table( cbind( XX, YY ))
    # for ( i in 1:10 )  print( paste( i, rownames(A)[i], CONCAT(XX[XX<as.numeric(rownames(A)[i])]), "||", CONCAT(A[1:i,1]), "||", CONCAT(A[1:i,2]), "||", GET_ENTROPY_XY_WRT( A[1:i,1], A[1:i,2], LOG=log10 )))
    # ######################################################################################################


    # ######################################################################################################
    XY = READ_C45_DATA( 'car' )
    ORDERING = sample( rownames(XY), nrow(XY) )
    XY = XY[ORDERING,]
    X  = SLICE_DATAFRAME(XY, c(1:6)) 
    Y  = SLICE_DATAFRAME(XY, 7)

    FORMULA = sprintf( "%s ~ .", colnames(Y) )

    MODEL = DO_DECISION_TREE( XY, Y, FORMULA=FORMULA, DO_PRUNING=TRUE, minbucket=20 )
    YP    = PREDICT_TREE( XY, Y[,1], as_probas=FALSE )
    IMSE  = GET_TREE_IMSE( Y[,1], YP, total=FALSE )
    CM    = GET_CONFUSION_MATRIX( Y[,1], YP )
    PLOT_TREE( MODEL, IMSE, CM )
    # ######################################################################################################



# require(party)
#  
# (ct = ctree(FORMULA, data = XY ))
# plot(ct, main="Conditional Inference Tree")
#  
# #Table of prediction errors
# table(predict(ct), Y[,1])
#  
# # Estimated class probabilities
# tr.pred = predict(ct, newdata=XY, type="prob" )
# 
options( opts )
# ######################################################################################################




