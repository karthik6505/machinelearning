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
DO_FEATURE_SET_SUBSELECTION = function( XY, dfname="R:N", cmax=0.9, topn=10, nmax=nrow(X), from_quantile=0.5, refine=TRUE, debug=FALSE, key="T" ) {
    new_colnames = c(1:(ncol(XY)-1)) + 1000
    old_colnames = paste( key, new_colnames, sep="" )
    old_colnames = append( old_colnames, "Y" )
    new_colnames = append( new_colnames, "Y" )
    if ( debug ) { cat( HEADER ) ; print( paste( "MAPS", old_colnames, new_colnames ) ) ; cat( HEADER ) }
    colnames(XY) = old_colnames
	F10 = DO_GENERAL_SUBSET_SELECTION( XY, dfname=dfname,
	                                           using_approach=linear.correlation, 
                                               approach_ppname="linear.correlation", 
	                                           rtypes="SUBSAMPLE|COMPLETECASES|CORRELATION",
	                                           target_var="Y", 
                                               top=topn, 
                                               nmax=nmax, 
                                               cmax=cmax, 
                                               percentile_threshold=from_quantile, 
                                               refine=refine )
    #colnames(XY) = old_colnames
	WHICH_VARS = F10[[3]]           
	if ( debug ) {
        cat( HEADER )
        print( F10 )
        str( F10 )
        cat( HEADER )
        print(paste("any(WHICH_VARS==Y)", any(WHICH_VARS=="Y")))
        print( setdiff( old_colnames, WHICH_VARS) )
        print( WHICH_VARS )
        cat( HEADER )
    } 

    # SELECTED = c(); for ( i in 1:length(WHICH_VARS) ) { SELECTED= append( SELECTED, which( colnames(XY)==WHICH_VARS[i] ))}
    Xx = as.data.frame(as.matrix(XY[, WHICH_VARS]))
    colnames(Xx) = WHICH_VARS 
	# Xx = Xx[,-ncol(Xx)]                  
	# Zx = DO_PCA( scale( Xx ) )$Z    
    if ( debug ) 
        MDF_STATS( Xx )

    return ( Xx)
}
# ######################################################################################################


# ######################################################################################################
GENERATE_INTERACTION_TERMS = function( X, interaction="mult", op=as.numeric ) {
    START = TRUE
    for ( i in 1:(ncol(X)-1) ) {
        for ( j in (i+1):ncol(X) ) {
            if ( interaction == "mult" ) ITERM = op(X[,i]) * op(X[,j])
            if ( interaction == "div" )  ITERM = op(X[,i]) / op(X[,j])
            if ( interaction == "add" )  ITERM = op(X[,i]) + op(X[,j])
            if ( interaction == "sub" )  ITERM = op(X[,i]) - op(X[,j])
            if ( START ) {
                INTERACTION_TERMS = data.frame( 'ITERM'=ITERM )
                START = FALSE
            } else {
                INTERACTION_TERMS = cbind( INTERACTION_TERMS, ITERM ) 
            }
        }
    }
    return ( INTERACTION_TERMS )
}
# ######################################################################################################


# ######################################################################################################
PRINT_DESIGN_MATRIX_DETAILS = function( XX, FINAL_MAPPING, debug=FALSE ) {
    cat(HEADER)
    cat(HEADER)
        SELECTED = c()
        for ( i in 1:length(colnames(XX)) ) { SELECTED= append( SELECTED, which( str_detect( FINAL_MAPPING, colnames(XX)[i]) == TRUE ))}
        if ( debug ) { print ( SELECTED ) ; print( FINAL_MAPPING[SELECTED] ) }

        X = XX
        if ( debug ) str(X)
        cat( HEADER )
        print( "FINAL_MAPPING[FEATURE SELECTED VARIABLES]" )
        print( FINAL_MAPPING[SELECTED] )
        cat( HEADER )
        print( "CORRELATION BETWEEN SELECTED FEATURES" )
        print( ifelse(cor(X) > 0.8, 111, 0 ) )
        cat( HEADER )

        HASH_SELECTED = list()
        for ( i in 1:length(colnames(XX)) ) { 
            HASH_SELECTED[[i]] = c( colnames(XX)[i], FINAL_MAPPING[SELECTED[i]]) 
            if ( debug ) print( HASH_SELECTED[[i]] )
        }
    cat(HEADER)
    cat(HEADER)
    return ( HASH_SELECTED )
}
# ######################################################################################################


# ######################################################################################################
    # ######################################################################################################
	# > summary(X)
	#        zn             indus            nox               rm             age              dis        
	#  Min.   :  0.00   Min.   : 0.46   Min.   :0.3850   Min.   :3.561   Min.   :  2.90   Min.   : 1.130  
	#  1st Qu.:  0.00   1st Qu.: 5.19   1st Qu.:0.4490   1st Qu.:5.886   1st Qu.: 45.02   1st Qu.: 2.100  
	#  Median :  0.00   Median : 9.69   Median :0.5380   Median :6.208   Median : 77.50   Median : 3.207  
	#  Mean   : 11.36   Mean   :11.14   Mean   :0.5547   Mean   :6.285   Mean   : 68.57   Mean   : 3.795  
	#  3rd Qu.: 12.50   3rd Qu.:18.10   3rd Qu.:0.6240   3rd Qu.:6.623   3rd Qu.: 94.08   3rd Qu.: 5.188  
	#  Max.   :100.00   Max.   :27.74   Max.   :0.8710   Max.   :8.780   Max.   :100.00   Max.   :12.127  
	#       rad              tax           ptratio            b         
	#  Min.   : 1.000   Min.   :187.0   Min.   :12.60   Min.   :  0.32  
	#  1st Qu.: 4.000   1st Qu.:279.0   1st Qu.:17.40   1st Qu.:375.38  
	#  Median : 5.000   Median :330.0   Median :19.05   Median :391.44  
	#  Mean   : 9.549   Mean   :408.2   Mean   :18.46   Mean   :356.67  
	#  3rd Qu.:24.000   3rd Qu.:666.0   3rd Qu.:20.20   3rd Qu.:396.23  
	#  Max.   :24.000   Max.   :711.0   Max.   :22.00   Max.   :396.90  
	# > 
    # ######################################################################################################
    # Boston {MASS}	R Documentation     Housing Values in Suburbs of Boston     Description
	# The Boston data frame has 506 rows and 14 columns.  [Package MASS version 7.2-29 Index]
    # ######################################################################################################
	# crim    per capita crime rate by town 
	# zn      proportion of residential land zoned for lots over 25,000 sq.ft. 
	# indus   proportion of non-retail business acres per town 
	# chas    Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
	# nox     nitrogen oxides concentration (parts per 10 million) 
	# rm      average number of rooms per dwelling 
	# age     proportion of owner-occupied units built prior to 1940 
	# dis     weighted mean of distances to five Boston employment centres 
	# rad     index of accessibility to radial highways 
	# tax     full-value property-tax rate per $10,000 
	# ptratio pupil-teacher ratio by town 
	# black   1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
	# lstat   lower status of the population (percent) 
	# medv    median value of owner-occupied homes in $1000 
    # ######################################################################################################
	# Source
	# Harrison, D. and Rubinfeld, D.L. (1978) Hedonic prices and the demand for clean air. J. Environ. Economics and Management 5, 81–102.
	# Belsley D.A., Kuh, E. and Welsch, R.E. (1980) Regression Diagnostics. Identifying Influential Data and Sources of Collinearity. New York: Wiley.
    # ######################################################################################################
# ######################################################################################################
BUILD_BOSTONHOUSING_DATASET = function( DO_FSELECTION=TRUE, DO_FEATURE_EXPLORATION=TRUE, pix=14, do_log=TRUE ) {
    require('mlbench')
    data(BostonHousing)

    Y = log(BostonHousing[pix])
    X = BostonHousing[-c(4,pix)]
    if ( do_log ) Y = log(BostonHousing[pix] + 1)
    if ( do_log ) X = log(BostonHousing[-c(4,pix)]+ 1)

    FEATURE_SELECTION = DO_FSELECTION

    if ( DO_FEATURE_EXPLORATION ) {
        INTERACTION_TERMS   = GENERATE_INTERACTION_TERMS( X, interaction="mult", op=sqrt )
        HIGHER_DEGREE_TERMS = cbind(  X^(3/2), X^(1/3), sqrt(X+1) )
        INTERESTING_TERMS   = with(X, cbind( rm^2,dis^2,
                                                (log(age)*rm)/(lstat+1),
                                                1/(rad*dis),
                                                1/sqrt(lstat),
                                                dis^2*nox,
                                                1/sqrt(crim),
                                                log(tax^2/rm),
                                                age/(ptratio+1)))
        COMPLETE_TERMS_DF   = cbind(  X, INTERACTION_TERMS, HIGHER_DEGREE_TERMS, INTERESTING_TERMS)
        colnames(COMPLETE_TERMS_DF) = paste("X",1:ncol(COMPLETE_TERMS_DF), sep="")
    } else {
        COMPLETE_TERMS_DF = X
    }

    FULL_DF    = EXTEND_DF( COMPLETE_TERMS_DF, Y, colname="Y" )
    WHICH_ROWS = complete.cases(FULL_DF)
    YCOL       = ncol(FULL_DF)
    Y          = FULL_DF[WHICH_ROWS,         YCOL]
    FULL_DF    = FULL_DF[WHICH_ROWS,        -YCOL]
    ORIG_X     = FULL_DF[WHICH_ROWS,c(1:ncol(FULL_DF))]
    XX         = ORIG_X

    FINAL_MAPPING = sprintf("FINAL_MAPPING: %16s-->%16s", colnames(XX), paste("S",1000+1:ncol(XX), sep=""))

        NR         = nrow(FULL_DF)
        if ( FEATURE_SELECTION ) {
            if ( DO_FEATURE_EXPLORATION ) {
                XY = EXTEND_DF( INTERACTION_TERMS,   Y, colname="Y" )
                    X1 = DO_FEATURE_SET_SUBSELECTION( XY, dfname="Interaction_Terms,  R:N",cmax=0.8, topn=6, nmax=NR, from_quantile=0.6, key="I" )
                XY = EXTEND_DF( HIGHER_DEGREE_TERMS, Y, colname="Y" )
                    X2 = DO_FEATURE_SET_SUBSELECTION( XY, dfname="Higher_Order_Terms, R:N",cmax=0.8, topn=6, nmax=NR, from_quantile=0.6, key="O" )
                XY = EXTEND_DF( INTERESTING_TERMS,   Y, colname="Y" )
                    X3 = DO_FEATURE_SET_SUBSELECTION( XY, dfname="Interesting_Terms,  R:N",cmax=0.8, topn=6, nmax=NR, from_quantile=0.6, key="P" )
                XY = EXTEND_DF( X,                   Y, colname="Y" )
                    X0 = DO_FEATURE_SET_SUBSELECTION( XY, dfname="Original    Terms,  R:N",cmax=0.7, topn=8, nmax=NR, from_quantile=0.6, key="X" )
                XX = cbind( X0, X1, X2, X3 )
            }

            FINAL_MAPPING = sprintf("FINAL_MAPPING: %16s-->%16s", colnames(XX), paste("S",1000+1:ncol(XX), sep=""))

            XY = EXTEND_DF( XX,                  Y, colname="Y" )
                 XX = DO_FEATURE_SET_SUBSELECTION( XY, dfname="Union Sel'd Subsets,R:N",cmax=0.8, topn=4, nmax=NR, from_quantile=0.6, key="S", 
                                                  debug=FALSE, refine=FALSE)

            MDF_STATS( XX )
        }

    SELECTED_FEATURES = PRINT_DESIGN_MATRIX_DETAILS( XX, FINAL_MAPPING )

    X = as.matrix(XX)
    Y = as.matrix(Y)

    retvals = list( 'X'=X, 'Y'=Y, 'MAPPING'=FINAL_MAPPING, 'SELECTED_FEATURES'=SELECTED_FEATURES, 'DO_FS'=FEATURE_SELECTION, 'DO_FE'=DO_FEATURE_EXPLORATION )
    return ( retvals )
}
# ######################################################################################################


# ######################################################################################################
BUILD_SYNTHETIC_DATASET = function( DO_FSELECTION=FALSE, DO_FEATURE_EXPLORATION=FALSE, do_log=TRUE ) {
    FEATURE_SELECTION = DO_FSELECTION
    NR = 1E4
    NF = 7
    NE = NR*NF
    X = matrix(rnorm(NE,0,10),NR)
        MODEL     = (1/t(as.matrix(seq(1,NF,1))))^2
        INTERCEPT = pi
        NOISE     = 5E-6 * matrix(rnorm(NR,0,10),NR)
        Y         = INTERCEPT + ( t( MODEL %*% t(X) ) + NOISE )

    if ( DO_FEATURE_EXPLORATION ) {
        INTERACTION_TERMS   = GENERATE_INTERACTION_TERMS( X, interaction="mult", op=abs )
        HIGHER_DEGREE_TERMS = cbind(  abs(X)^(1/2 ) )
        COMPLETE_TERMS_DF   = cbind(  X, INTERACTION_TERMS, HIGHER_DEGREE_TERMS )
        colnames(COMPLETE_TERMS_DF) = paste("X",1:ncol(COMPLETE_TERMS_DF), sep="")
    } else {
        COMPLETE_TERMS_DF = X
    }
    COMPLETE_TERMS_DF = X
    colnames(COMPLETE_TERMS_DF) = paste("X",1:ncol(COMPLETE_TERMS_DF), sep="")

    FULL_DF = EXTEND_DF( COMPLETE_TERMS_DF, Y, colname="Y" )
    WHICH_ROWS = complete.cases(FULL_DF)
    YCOL       = ncol(FULL_DF)
    Y          = FULL_DF[WHICH_ROWS,         YCOL]
    FULL_DF    = FULL_DF[WHICH_ROWS,        -YCOL]
    ORIG_X     = FULL_DF[WHICH_ROWS,c(1:ncol(FULL_DF))]
    XX         = ORIG_X

    FINAL_MAPPING = sprintf("FINAL_MAPPING: %4s-->%4s", colnames(XX), paste("S",1000+1:ncol(XX), sep=""))

        NR         = nrow(FULL_DF)
        if ( FEATURE_SELECTION ) {
            if ( DO_FEATURE_EXPLORATION ) {
                XY = EXTEND_DF( INTERACTION_TERMS,   Y, colname="Y" )
                    X1 = DO_FEATURE_SET_SUBSELECTION( XY, dfname="Interaction_Terms,  R:N",cmax=0.8, topn=6, nmax=NR, from_quantile=0.6, key="I" )
                XY = EXTEND_DF( HIGHER_DEGREE_TERMS, Y, colname="Y" )
                    X2 = DO_FEATURE_SET_SUBSELECTION( XY, dfname="Higher_Order_Terms, R:N",cmax=0.8, topn=6, nmax=NR, from_quantile=0.6, key="O" )
                XY = EXTEND_DF( X,                   Y, colname="Y" )
                    X0 = DO_FEATURE_SET_SUBSELECTION( XY, dfname="Original    Terms,  R:N",cmax=0.7, topn=8, nmax=NR, from_quantile=0.6, key="X" )
                XX = cbind( X0, X1, X2 )
            }

            FINAL_MAPPING = sprintf("FINAL_MAPPING: %16s-->%16s", colnames(XX), paste("S",1000+1:ncol(XX), sep=""))

            XY = EXTEND_DF( XX,                  Y, colname="Y" )
                 XX = DO_FEATURE_SET_SUBSELECTION( XY, dfname="Union Sel'd Subsets,R:N",cmax=0.7, topn=4, nmax=NR, from_quantile=0.6, key="S", 
                                                  debug=FALSE, refine=FALSE)

            MDF_STATS( XX )
        }

    SELECTED_FEATURES = PRINT_DESIGN_MATRIX_DETAILS( XX, FINAL_MAPPING )

    X = as.matrix(XX)
    Y = as.matrix(Y)

    retvals = list( 'X'=X, 'Y'=Y, 'MAPPING'=FINAL_MAPPING, 'SELECTED_FEATURES'=SELECTED_FEATURES, 'DO_FS'=FEATURE_SELECTION, 'DO_FE'=DO_FEATURE_EXPLORATION )
    return ( retvals )
}
# ######################################################################################################


