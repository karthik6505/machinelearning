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
require(stringr)
# ######################################################################################################


# ######################################################################################################
OLD_CA = commandArgs()
commandArgs <- function() list(TEST_ENABLED=FALSE, DO_DISTANCE_VALIDATION=FALSE, RUN_VALIDATE_SGD=FALSE )
source( 'utilities.R' )
source( 'fselect.R' )
source( 'datasets.R' )
source( 'distances.R' )
source( 'stochastic_gradient_descent.R' )
# ######################################################################################################


# ###################################################################################################
# GLOBALS ***** GLOBALS ***** GLOBALS ***** GLOBALS ***** GLOBALS ***** GLOBALS ***** GLOBALS *****
# ###################################################################################################
PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX = matrix(0)
MY_DISTANCE_MATRIX_RETVALS = list()
# ###################################################################################################


# ###################################################################################################
COMBINED_RECOMMENDER_SYSTEM = function() {
}
# ###################################################################################################


# ###################################################################################################
IDENTIFY_SIMILAR_USERS  = function( ) {
}
# ###################################################################################################


# ###################################################################################################
IDENTIFY_SIMILAR_MOVIES = function( ) {
}
# ###################################################################################################


# ###################################################################################################
GENERATE_FEATURE_RATING_RECOMMENDATIONS_FOR_USER = function () {
}
# ###################################################################################################


# ###################################################################################################
IDENTIFY_TOP_MOVIE_RECOMMENDATIONS_FOR_USER = function () {
}
# ###################################################################################################


# ######################################################################################################
EXTRACT_RATINGS_SUBMATRIX = function( OR, m=100, n=200, WO_ZERO_ENTRIES=TRUE, debug=FALSE ) {
    m = min(nrow(OR), m)
    n = min(ncol(OR), n)
    if ( m == 0 ) m = nrow(OR)
    if ( n == 0 ) n = ncol(OR)

    sampled_users  = sort(sample(1:nrow(OR),m))
    sampled_movies = sort(sample(1:ncol(OR),n))

    mdat = OR[sampled_users, sampled_movies]

    retvals = list( 'RATMAT'=mdat, 'WHICH_USERS'=sampled_users, 'WHICH_MOVIES'=sampled_movies )
    if (debug ) { 
        cat(HEADER)
        str(retvals)
        cat(HEADER)
    }
    return ( retvals )
}
# ######################################################################################################


# ###################################################################################################
GET_XY_FOR_GENRE_AS_X_AND_USER_RATING_AS_Y = function( ORIG_RATINGS, NUSERS=100, NMOVIES=200, debug=TRUE ) {
    RATINGS_RETVALS = EXTRACT_RATINGS_SUBMATRIX( ORIG_RATINGS, m=NUSERS, n=NMOVIES)
        M2U = t( RATINGS_RETVALS$RATMAT )
        Y = M2U
        WHICH_USERS  = RATINGS_RETVALS$WHICH_USERS
        WHICH_MOVIES = RATINGS_RETVALS$WHICH_MOVIES

        X  = MOVIES[WHICH_MOVIES, c(1,6:ncol(M))] 
        MU = matrix(apply(M2U,1,mean,na.rm=TRUE))
        SD = matrix(apply(M2U,1,sd,na.rm=TRUE))  
        SD = ifelse( is.na(SD), 0, SD )

        if ( TRUE ) {
            MU = MU / mean(MU,na.rm=TRUE)
            SD = SD / mean(SD,na.rm=TRUE)
        }

        # ###########################################################################
        # BOOTSTRAP the FIRST generation of the PREDICTIVE FEATURES (i.e., GENREs) as 
        # the AVERAGE Rating BY THOSE WHO RATED THE MOVIE plus STD.DEV on such ratings
        # ###########################################################################
        # NX = data.frame( "movie_id"=X[,1], "rating_mean"=MU, "rating_dev"=SD )
        NX = data.frame( "rating_mean"=MU, "rating_dev"=SD )
        for( i in 2:ncol(X)) {
            Xi = matrix(X[,i])
            Xi = Xi * ( MU + SD )
            NX = cbind( NX, Xi )
        }
        # colnames(NX) = c( "movie_id", "rating_mean", "rating_dev", colnames(MOVIES[c(6:ncol(M))]))
        colnames(NX) = c( "rating_mean", "rating_dev", colnames(MOVIES[c(6:ncol(M))]))
        rownames(NX) = rownames(M2U)

        # NX = NX[,3:ncol(NX)]

    retvals = list ( 'X'=NX, 'Y'=Y, 'WHICH_USERS'=WHICH_USERS, 'WHICH_MOVIES'=WHICH_MOVIES )
    if ( debug ) str(retvals)
    return ( retvals )
}
# ###################################################################################################


# ###################################################################################################
SUBSELECT_THOSE_REVIEWED_BY_USER = function( X, Y, debug=FALSE ) {
    XY = cbind(X,Y)
    idx = which( complete.cases(XY) )
    if ( debug ) {
        print ( idx )
        SUMMARY( XY[idx,] )
    }
    return ( idx )
}
# ###################################################################################################


# ###################################################################################################
GET_XY_FOR_USER_PREFERENCES_FOR_GENRE_MATRIX = function( X_GEN, Y_M2U, regparam=2E-1, w_intercept=FALSE, debug=FALSE ) {
    DROP=c()

    GENRE_COLS = c(1:ncol(X_GEN))
    N_GENRE = length(GENRE_COLS)
    N_USERS = ncol(Y_M2U)

    if ( w_intercept )
        PREFERENCES = matrix(rep(NA, N_USERS*(N_GENRE+1)), N_USERS)
    else
        PREFERENCES = matrix(rep(NA, N_USERS*(N_GENRE)), N_USERS)

    X_GEN = X_GEN[, GENRE_COLS ]

    if ( debug ) cat(HEADER)
    for ( USER in 1:ncol(Y_M2U)) {
        idx = SUBSELECT_THOSE_REVIEWED_BY_USER( X_GEN, Y_M2U[,USER] )
        if ( length(idx) <= DROP_TRIGGER ) { 
            DROP = append(DROP, USER)
            next 
        }

        XX = X_GEN[idx, ]
        YY = Y_M2U[idx, USER]

        NEQ =  DO_NORMAL_EQUATIONS( XX, YY, regularization_parameter=regparam, do_scale=FALSE, w_intercept=w_intercept, debug=debug )
        THETA_GENRE = GET_NEQ_THETA( NEQ )

        PREFERENCES[ USER, ] = THETA_GENRE
    }

    # if ( length(DROP)!= 0 ) print( DROP )

    PREFERENCES = as.data.frame( PREFERENCES )
    if ( w_intercept )
        colnames( PREFERENCES ) = c( "UINTERCEPT", colnames(X_GEN)[GENRE_COLS] )
    else
        colnames( PREFERENCES ) = colnames(X_GEN)[GENRE_COLS]

    rownames( PREFERENCES ) = rownames(t(Y_M2U))
    PREFERENCES = as.matrix( PREFERENCES )
    return ( PREFERENCES )
}
# ###################################################################################################


# ###################################################################################################
GET_XY_FOR_PREFERENCES_AS_X_AND_USER_RATING_AS_Y = function( X_GEN, Y_U2M, regparam=2E-1, w_intercept=FALSE, debug=FALSE ) {
    DROP=c()

    GENRE_COLS = c(1:ncol(X_GEN))
    N_GENRE = length(GENRE_COLS)

    N_USERS  = nrow(Y_U2M)
    N_MOVIES = ncol(Y_U2M)

    if ( w_intercept )
        MOVIEGENRE_FEATURES = matrix( rep(NA, N_MOVIES*(N_GENRE+1)), N_MOVIES )
    else
        MOVIEGENRE_FEATURES = matrix( rep(NA, N_MOVIES*(N_GENRE)), N_MOVIES )

    X_GEN = X_GEN[, GENRE_COLS ]

    if ( debug ) cat(HEADER)
    for ( MOVIE in 1:ncol(Y_U2M)) {
        idx = SUBSELECT_THOSE_REVIEWED_BY_USER( X_GEN, Y_U2M[,MOVIE] )
        if ( length(idx) <= DROP_TRIGGER ) { 
            DROP = append(DROP, MOVIE)
            next 
        }

        XX = X_GEN[idx, ]
        YY = Y_U2M[idx, MOVIE]

        NEQ =  DO_NORMAL_EQUATIONS( XX, YY, regularization_parameter=regparam, do_scale=FALSE, w_intercept=w_intercept, debug=debug )
        THETA_MOVIE = GET_NEQ_THETA( NEQ )

        MOVIEGENRE_FEATURES[ MOVIE, ] = THETA_MOVIE
    }

    # if ( length(DROP)!= 0 ) print( DROP )

    MOVIEGENRE_FEATURES = as.data.frame( MOVIEGENRE_FEATURES )
    if ( w_intercept )
        colnames( MOVIEGENRE_FEATURES ) = c( "MINTERCEPT", colnames(X_GEN) )
    else
        colnames( MOVIEGENRE_FEATURES ) = colnames(X_GEN)
    rownames( MOVIEGENRE_FEATURES ) = rownames(t(Y_U2M))
    MOVIEGENRE_FEATURES = as.matrix( MOVIEGENRE_FEATURES )
    return ( MOVIEGENRE_FEATURES )
}
# ###################################################################################################


# ###################################################################################################
DO_PLOTTING_ITERATION = function( L_PREFERENCES, L_MOVIEGEN_FEATURES, xval=4 ) {
    XYZ0 = L_PREFERENCES [complete.cases(L_PREFERENCES ),]
        Z = DO_PCA( XYZ0, silent=TRUE, debug=FALSE )$Z
        plot( Z[,1], Z[,2], pch=".", cex=0.4, xlim=c(-(xval+1),xval+1), ylim=c(-(xval+1),xval+1) )
        text( Z[,1], Z[,2], rownames(XYZ0), col='blue', cex=0.3 )

    XYZ1 = L_MOVIEGEN_FEATURES[complete.cases(L_MOVIEGEN_FEATURES),]
        Z = DO_PCA( XYZ1, silent=TRUE, debug=FALSE )$Z
        plot( Z[,1], Z[,2], pch=".", cex=0.4, xlim=c(-xval,xval), ylim=c(-xval,xval) )
        text( Z[,1], Z[,2], rownames(XYZ1), col='brown', cex=0.3 )

    retvals = list ( XYZ0, XYZ1 )

    return ( retvals )
}
# ###################################################################################################


# ###################################################################################################
DO_ANALYSIS_DIFFERENCES_BETWEEN_COEFFICIENTS = function( PREFERENCES2, PREFERENCES1, ptitle="PREFERENCES DIFF" )  {
    PREFERENCES_DIFF   = PREFERENCES2  -  PREFERENCES1
    PREFERENCES_DIFF   = PREFERENCES_DIFF[complete.cases(PREFERENCES_DIFF),]
    P_TS = apply( PREFERENCES_DIFF, 1, function(x) { t = t(x)%*%x ; ifelse(is.na(t),0,t)} )
    DO_HIST(P_TS, nbins=64, ptitle=ptitle)
    P_MSE = as.numeric( t(P_TS) %*% P_TS ) / length(P_TS) 
    return ( P_MSE )
}
# ###################################################################################################


# ###################################################################################################
# RATINGS     : BOOTSTRAP OF USER TO MOVIE RATINGS
# PREFERENCES : USER DATA INDICATING PREFERENCES TOWARDS GENRES (M users, P genres)
# X_GEN, Y_M2U: GENRES AS PREDICTORS OF MOVIE RATINGS FOR EACH USER (resulting in PREFERENCES) AS THETA VECTORS
# GEN_RATINGS : GIVEN USER PREFERENCES AS PREDICTORS OF X_GEN VALUES AS THETA VECTORS
# ###################################################################################################
DO_ITERATIVE_CONVERGENCE_FOR_RECOMMENDATION_PARAMETERS = function( ORIG_RATINGS, NUSERS=1000, NMOVIES=2000, debug=FALSE ) {
    NEWLINE(3); cat( HEADER )
    GENRE_RETVALS = GET_XY_FOR_GENRE_AS_X_AND_USER_RATING_AS_Y( ORIG_RATINGS, NUSERS=NUSERS, NMOVIES=NMOVIES, debug=debug )
    X_GEN = GENRE_RETVALS$X
    Y_M2U = GENRE_RETVALS$Y
    WHICH_USERS  <<- GENRE_RETVALS$WHICH_USERS
    WHICH_MOVIES <<- GENRE_RETVALS$WHICH_MOVIES

    X_GEN = scale( X_GEN, center=FALSE, scale=FALSE)
    Y_M2U = scale( Y_M2U, center=TRUE,  scale=FALSE)

    if ( debug ) {
        NEWLINE(3)
        cat( HEADER )
        SUMMARY( Y_M2U, w_str=TRUE )
        cat( HEADER )
        NEWLINE(3)
    }
    # ###################################################################################################

    # ###################################################################################################
    NEWLINE(3); cat( HEADER )
    PREFERENCES2 = GET_XY_FOR_USER_PREFERENCES_FOR_GENRE_MATRIX( X_GEN, Y_M2U, regparam=REGPARAM, debug=debug)
    # ###################################################################################################

    # ###################################################################################################
    NEWLINE(3); cat( HEADER )
    Y_U2M = t(Y_M2U)
    MOVIEGEN_FEATURES2 = GET_XY_FOR_PREFERENCES_AS_X_AND_USER_RATING_AS_Y( PREFERENCES2, Y_U2M )
    # ###################################################################################################

    # ###################################################################################################
    op = par( mfrow=c(3,2) )
    if ( DO_PDF ) pdf( 'coefficient_convergence.pdf' )
    print( "PDF FILE contains plots tracing the convergence of coefficients: coefficients_convergence.pdf" )
    op = par( mfrow=c(3,2) )

    MSE = data.frame( "GEN"=0, "P_MSE"=Inf, "M_MSE"=Inf )
    WHY = "MAXIMUM NUMBER OF ITERATIONS EXCEEDED"
    # ###################################################################################################
    # THIS IMPLEMENTS ITERATIVE CONVERGENCE WHICH IS A LOT SLOWER THAN TO DO IT ALL AT ONCE BUT ALLOWS
    # ACCESS TO EACH ITERATION OF THE CONVERGENCE TO PLOT THINGS SUCH AS COEFFICIENT TRAVERSAL AND STABILITY
    # ###################################################################################################
    for( iter in 2:N_ITER ) {
        PREFERENCES1       = GET_XY_FOR_USER_PREFERENCES_FOR_GENRE_MATRIX( MOVIEGEN_FEATURES2, Y_M2U, regparam=REGPARAM, debug=FALSE )
        MOVIEGEN_FEATURES1 = GET_XY_FOR_PREFERENCES_AS_X_AND_USER_RATING_AS_Y( PREFERENCES2,   Y_U2M )

        PREFERENCES2       = GET_XY_FOR_USER_PREFERENCES_FOR_GENRE_MATRIX( MOVIEGEN_FEATURES1, Y_M2U, regparam=REGPARAM, debug=FALSE )
        MOVIEGEN_FEATURES2 = GET_XY_FOR_PREFERENCES_AS_X_AND_USER_RATING_AS_Y( PREFERENCES1,   Y_U2M )

        PLOT_RETVALS       = DO_PLOTTING_ITERATION( PREFERENCES1, MOVIEGEN_FEATURES1 )
        PLOT_RETVALS       = DO_PLOTTING_ITERATION( PREFERENCES2, MOVIEGEN_FEATURES2 )

        P_MSE = DO_ANALYSIS_DIFFERENCES_BETWEEN_COEFFICIENTS( PREFERENCES2,       PREFERENCES1,       ptitle="PREFERENCES DIFF" )
        M_MSE = DO_ANALYSIS_DIFFERENCES_BETWEEN_COEFFICIENTS( MOVIEGEN_FEATURES2, MOVIEGEN_FEATURES1, ptitle="MOVIEGEN_DIFF" )

        MSE = rbind( MSE, data.frame( "GEN"=i, "P_MSE"=P_MSE, "M_MSE"=M_MSE ))
        print( sprintf( "ITERATION=%4d MSE(COEFF_DIFF BETWEEN GEN) P=%16.8g M=%16.8g", iter, P_MSE, M_MSE ) )

        idx = c(1:nrow(MSE))
        if ( nrow(MSE) > 3 ) idx = c( (nrow(MSE)-3) : nrow(MSE))
        MEAN_P_MSE = mean(MSE$P_MSE[idx])
        MEAN_M_MSE = mean(MSE$M_MSE[idx])

        METRICS <<- TIMESTAMP( paste( "CONVERGENCE_ITERATION", iter ) )

        if ( P_MSE<EPSILON  & M_MSE<EPSILON ) {                 
            cat(HEADER)
            WHY = "STABILITY TERMINATION CONDITION REACHED"
            if ( MEAN_P_MSE<EPSILON  & MEAN_M_MSE<EPSILON ) WHY = "STRONG STABILITY TERMINATION CONDITION REACHED"
            print( WHY )
            print( paste( P_MSE,      M_MSE ) )
            print( paste( MEAN_P_MSE, MEAN_M_MSE ) )
            cat(HEADER)
            break 
        }
    }
    # ###################################################################################################
    if ( DO_PDF ) { dev.off() ; graphics.off() }
    # ###################################################################################################

    ITER_RETVALS = list( 'PREFERENCES'=PREFERENCES2, 
                         'MOVIEGEN_FEATURES'=MOVIEGEN_FEATURES2,
                         'MSE_STATS'=MSE,
                         'Y_M2U'=Y_M2U,
                         'WHY'=WHY )

    return( ITER_RETVALS )
}
# ###################################################################################################


# ###################################################################################################
GENERATE_LOW_RANK = function( USER_PREFERENCES, MOVIE_FEATURES, NA_RM=TRUE ) {
    if ( NA_RM ) {
        NA_MOVIES = c()
        for ( MOVIE in rownames(MOVIE_FEATURES) ) {
            mfv = MOVIE_FEATURES[MOVIE,]
            if ( all(is.na(mfv)) ) NA_MOVIES = append( NA_MOVIES, MOVIE ) 
        }
        MOVIE_FEATURES[NA_MOVIES,] = 0
        SUMMARY( NA_MOVIES )
    
        NA_MOVIES = c()
        for ( MOVIE in rownames(MOVIE_FEATURES) ) {
            mfv = MOVIE_FEATURES[MOVIE,]
            if ( any(is.na(mfv)) ) NA_MOVIES = append( NA_MOVIES, MOVIE ) 
        }
        # SUMMARY( NA_MOVIES )
    }
    NORMALIZED_PREDICTIONS = USER_PREFERENCES %*% t(MOVIE_FEATURES)
}
# ###################################################################################################


# ###################################################################################################
# COLLABORATIVELY-FILTERED RATING GENERATION FOR USERS RATINGS AND MOVIE RATING-PREDICTIVE FEATURES
# ###################################################################################################
GENERATE_COLLABORATIVE_FILTERED_RATINGS = function( WHICH_RATINGS, USER_PREFERENCES, MOVIE_FEATURES, USER_MEAN_NORMALIZATIONS, sample_n=200, debug=TRUE ) {

    NORMALIZED_PREDICTIONS = GENERATE_LOW_RANK( USER_PREFERENCES, MOVIE_FEATURES )

    RECOMMENDATIONS_FOR_U2M_RATINGS  = apply( NORMALIZED_PREDICTIONS, 2, DENORMALIZE_MEAN_NORMALIZED_PREDICTIONS, 
                                                                         USER_MEANS=USER_MEAN_NORMALIZATIONS, 
                                                                         PRODUCTION_SYSTEM_CORRECTION=TRUE )

    colnames(RECOMMENDATIONS_FOR_U2M_RATINGS) = colnames(WHICH_RATINGS)
    rownames(RECOMMENDATIONS_FOR_U2M_RATINGS) = rownames(WHICH_RATINGS)

    if ( sample_n == 0 ) sample_n = nrow(WHICH_RATINGS)

    r_idxs = sample( rownames(WHICH_RATINGS), min(sample_n, nrow(WHICH_RATINGS)) )

    MSE = c()
    for ( MOVIE in colnames(USER_TO_MOVIE_RATINGS)) {
        u_idxs = names(which( !is.na(WHICH_RATINGS[,MOVIE]) ))

        print( sprintf( "%s : [%4d w/ selected ratings] : %s ", MOVIE, length(u_idxs), as.character(M[substr(MOVIE,2,10),"movie_title"]) )) 

        SUMMARY(RECOMMENDATIONS_FOR_U2M_RATINGS[,MOVIE])

        SUMMARY(WHICH_RATINGS[,MOVIE])

        SAMPLED_RECM = sprintf("%s[%.0f:%s]", r_idxs, 
                                     RECOMMENDATIONS_FOR_U2M_RATINGS[r_idxs, MOVIE], WHICH_RATINGS[r_idxs, MOVIE])

        SAMPLED_FITS = sprintf("%s[%.1f:%s]", u_idxs, 
                                     RECOMMENDATIONS_FOR_U2M_RATINGS[u_idxs, MOVIE], WHICH_RATINGS[u_idxs, MOVIE])

        FIT_MSE = NA
        if ( length(u_idxs) ) FIT_MSE = sum(RECOMMENDATIONS_FOR_U2M_RATINGS[u_idxs,MOVIE] - WHICH_RATINGS[u_idxs,MOVIE])^2 / length(u_idxs)
        MSE = append( MSE, FIT_MSE )

        cat(SUBHEADER)
        print( paste( "SAMPLE RECOMMENDATIONS  : ", CONCAT(SAMPLED_RECM) ))
        print( paste( "AVAILABLE USER FITS     : ", CONCAT(SAMPLED_FITS) ))
        print( paste( "MSE (WRT GIVEN MOVIE)   :  ", round(FIT_MSE,6) ))
        cat(HEADER)

        NEWLINE(1)
    }

    cat(HEADER)
    try( print(sprintf(" AVG(PER-MOVIE MSE) = %.6f", mean(MSE, na.rm=TRUE))))
    try( print(sprintf(" SD (PER-MOVIE MSE) = %.6f", sd(MSE,   na.rm=TRUE))))
    cat(HEADER)

    return (RECOMMENDATIONS_FOR_U2M_RATINGS)
}
# ###################################################################################################


# ###################################################################################################
# NUMBER OF STARS MAY ACTUALLY BE SLIGHTLY LARGER THAN FIVE
# ###################################################################################################
DENORMALIZE_MEAN_NORMALIZED_PREDICTIONS = function( USERS_WRT_MOVIE_PRED, USER_MEANS=c(), 
                                                   PRODUCTION_SYSTEM_CORRECTION=TRUE, TOP_RATING=5, LOW_RATING=0, debug=FALSE ) {

    NUMBER_STARS_PREDICTED_RATING = USERS_WRT_MOVIE_PRED + USER_MEANS

    # a 0:5 correction may be used here as values will be in range -0:5+some-extra
    if ( PRODUCTION_SYSTEM_CORRECTION ) {
        NUMBER_STARS_PREDICTED_RATING = apply( NUMBER_STARS_PREDICTED_RATING, 2, function(x) x = ifelse(x>TOP_RATING, TOP_RATING, x))
        NUMBER_STARS_PREDICTED_RATING = apply( NUMBER_STARS_PREDICTED_RATING, 2, function(x) x = ifelse(x<LOW_RATING, LOW_RATING, x))
        NUMBER_STARS_PREDICTED_RATING = round( NUMBER_STARS_PREDICTED_RATING )
    }

    if ( debug )
        SUMMARY( NUMBER_STARS_PREDICTED_RATING ) #, w_str=TRUE ) 

    return( NUMBER_STARS_PREDICTED_RATING ) 
}
# ###################################################################################################













    # ###################################################################################################
    # JRETVALS = BUILD_JESTER_DATASET()
    # J = as.data.frame(JRETVALS$J)
    # AGG_J = aggregate( J$AVG ~ as.factor(J$NR), J, mean )
    # ###################################################################################################








# ###################################################################################################
# ###################################################################################################
# ###################################################################################################
# ###################################################################################################
METRICS <<- INIT_METRICS()
METRICS <<- TIMESTAMP( "START" )


# ###################################################################################################
# ###################################################################################################
# ###################################################################################################
NEWLINE(10)






    # ###################################################################################################
    # ###################################################################################################
    # ###################################################################################################
    # GLOBALS PARAMETERS
    # ###################################################################################################
    N_USERS_TO_USE  = as.integer(1000 * 0.6)
    N_MOVIES_TO_USE = as.integer(1800 * 0.8)
        EPSILON      = 1E-1
        REGPARAM     = 3E-1
        N_ITER       = 50
        DO_PDF       = TRUE
        DROP_TRIGGER = 1
    WHICH_USERS  <<- c()
    WHICH_MOVIES <<- c()
    # ###################################################################################################
    # ###################################################################################################
    # ###################################################################################################

    # ###################################################################################################
    NEWLINE(3); cat( HEADER )
    ML = BUILD_MOVIELENS_DATASET()
    # ###################################################################################################
        ORIG_RATINGS = ML$user_movies
        USERS    = U = ML$users
        MOVIES   = M = ML$movies
        GENRES   = G = ML$genre
        OCCUP    = O = ML$occupation
        COUNTS   = C = ML$counts
        TRIPLETS = R = ML$ratings
    # ###################################################################################################
    
    # ###################################################################################################
    NEWLINE(3); cat( HEADER )
    i = 0
    for ( db in ML ) {
        i = i + 1
        cat ( HEADER )
        print( names(ML)[i] )
        cat ( HEADER )
        str(db)
        cat ( HEADER )
        NEWLINE(1)
    }
    # ###################################################################################################

    # ###################################################################################################
    ITER_RETVALS = DO_ITERATIVE_CONVERGENCE_FOR_RECOMMENDATION_PARAMETERS( ORIG_RATINGS, NUSERS=N_USERS_TO_USE, NMOVIES=N_MOVIES_TO_USE, debug=FALSE )
    METRICS <<- TIMESTAMP( paste( "ITERATIVE CONVERGENCE COMPLETED", i ) )
    # ###################################################################################################
        USER_PREFERENCES       =   ITER_RETVALS$'PREFERENCES'
        MOVIE_FEATURES         =   ITER_RETVALS$'MOVIEGEN_FEATURES'
        MSE_ON_COEFF_CHANGES   =   ITER_RETVALS$'MSE_STATS'
        MOVIE_TO_USER_RATINGS  =   ITER_RETVALS$'Y_M2U'
        USER_TO_MOVIE_RATINGS  = t(ITER_RETVALS$'Y_M2U')
        CONVERGENCE            =   ITER_RETVALS$'WHY'
        if (!is.null(attr(USER_TO_MOVIE_RATINGS,"scaled:center"))) USER_MEAN_NORMALIZATIONS  =   as.matrix(attr(USER_TO_MOVIE_RATINGS, "scaled:center"))
        if  (is.null(attr(USER_TO_MOVIE_RATINGS,"scaled:center"))) USER_MEAN_NORMALIZATIONS  =      rep(0, nrow(USER_TO_MOVIE_RATINGS))

        if (!is.null(attr(MOVIE_TO_USER_RATINGS,"scaled:center"))) MOVIE_MEAN_NORMALIZATIONS =   as.matrix(attr(MOVIE_TO_USER_RATINGS, "scaled:center"))
        if  (is.null(attr(MOVIE_TO_USER_RATINGS,"scaled:center"))) MOVIE_MEAN_NORMALIZATIONS =      rep(0, nrow(MOVIE_TO_USER_RATINGS))
    # ###################################################################################################

    # ###################################################################################################
    WHICH_RATINGS = ORIG_RATINGS[WHICH_USERS,WHICH_MOVIES]
    RECOMMENDATIONS_FOR_U2M_RATINGS = GENERATE_COLLABORATIVE_FILTERED_RATINGS( WHICH_RATINGS, 
                                                                     USER_PREFERENCES, 
                                                                     MOVIE_FEATURES, 
                                                                     USER_MEAN_NORMALIZATIONS, 
                                                                     sample_n=7, debug=TRUE )
    # ###################################################################################################
    

    # ###################################################################################################
    PREVIOUS_OPTIONS = options( digits=4 )
                       options( width=120 )
        NEWLINE(3); cat( HEADER )
            print( "USER PREFERENCES" )
            MDF_STATS(USER_PREFERENCES)
            SUMMARY( USER_PREFERENCES )
            cat( HEADER ); cat( HEADER )
        NEWLINE(3); cat( HEADER )
            print( "MOVIE FEATURES" )
            MDF_STATS(MOVIE_FEATURES)
            SUMMARY( MOVIE_FEATURES )
            cat( HEADER ); cat( HEADER )
        NEWLINE(3); cat( HEADER )
            print( "MSE WRT TO COEFFICIENT CHANGE ACROSS ITERATIONS" )
            print( MSE_ON_COEFF_CHANGES )
            cat( HEADER ); cat( HEADER )
        NEWLINE(3); cat( HEADER )
            print( "TIMING STATS" )
            print( METRICS )
            cat( HEADER ); cat( HEADER )
    options( PREVIOUS_OPTIONS )
    # ###################################################################################################


