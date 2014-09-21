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
source( 'utilities.R' )
source( 'fselect.R' )
# ######################################################################################################


# ###################################################################################################
# GLOBALS ***** GLOBALS ***** GLOBALS ***** GLOBALS ***** GLOBALS ***** GLOBALS ***** GLOBALS *****
# ###################################################################################################
PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX = matrix(0)
MY_DISTANCE_MATRIX_RETVALS = list()
# ###################################################################################################


# ###################################################################################################
OPTIMIZE_USER_TO_GENERE_PREFERENCES = function() {
}
# ###################################################################################################


# ###################################################################################################
OPTIMIZE_GENERE_FEATURES_PARAMETERS = function() {
}
# ###################################################################################################


# ###################################################################################################
DO_ITERATIVE_REFINEMENT = function( ) {
}
# ###################################################################################################


# ###################################################################################################
COMBINED_RECOMMENDER_SYSTEM = function() {
}
# ###################################################################################################


# ###################################################################################################
IDENTIFY_SIMILAR_USERS = function( ) {
}
# ###################################################################################################


# ###################################################################################################
GENERATE_FEATURE_RATING_RECOMMENDATIONS_FOR_USER = function () {
}
# ###################################################################################################


# ###################################################################################################
IDENTIFY_TOP_RECOMMENDATIONS_FOR_USER = function () {
}
# ###################################################################################################


# ###################################################################################################
DO_MEAN_NORMALIZATION = function() {
}
# ###################################################################################################


# ###################################################################################################
DIST = function( Xmat, blocksize=500 ) {
    VERIFY( "Xmat", Xmat )

    m = nrow(Xmat)
    n = ncol(Xmat)
    START = TRUE
    for (i in 1:(as.integer(m/blocksize)+1)) {
        # mini_df = Xmat%*%t(Xmat[i:(min(i+blocksize,m)),])
        imin=(i-1)*blocksize+1
        imax=min(i*blocksize,m)
        if (imin>m) break
        mini_df = DF_AS_MATRIX(Xmat[c(imin:imax),]) %*% t(DF_AS_MATRIX(Xmat))
        VERIFY( "mini_df", mini_df)
        if ( START ) {
            START = FALSE
            DISTANCE_MATRIX_INCREMENTAL <<- mini_df
        } 
        else {
            DISTANCE_MATRIX_INCREMENTAL <<- rbind( DISTANCE_MATRIX_INCREMENTAL , mini_df )
            VERIFY( sprintf("DISTANCE_MATRIX_INCREMENTAL-%s", i), DISTANCE_MATRIX_INCREMENTAL )
            cat( HEADER )
        }
    }

    # if ( m %% blocksize )  DISTANCE_MATRIX_INCREMENTAL <<- rbind( DISTANCE_MATRIX_INCREMENTAL , mini_df )

    VERIFY( "DISTANCE_MATRIX_INCREMENTAL", DISTANCE_MATRIX_INCREMENTAL )

    return( DISTANCE_MATRIX_INCREMENTAL )
}
# ###################################################################################################


# ###################################################################################################
GET_DISTANCE_MATRIX = function( xdf, dist_metric="euclidean", do_scaling=TRUE, do_plot=FALSE, transform="in_pca_domain,in_probas_domain", vargoal=0.995, ... ) {
    VERIFY( "X_INPUT", xdf)
    x_class = class( xdf )
    x = xdf
    if ( x_class == "data.frame" ) {
        x = DF_AS_MATRIX( xdf )
        METRICS <<- TIMESTAMP( "PREPROCESSING" )
    }

    VERIFY( "X_matrix", x)
    applied = c()
    if ( do_scaling ) {
        scalar_transform = scale( x, center=TRUE, scale=TRUE )
        applied = append( applied, "scaling_transform" )
        y = matrix(scalar_transform, nrow(xdf))
        METRICS <<- TIMESTAMP( "SCALING" )
    } else
        y = x

    VERIFY( "scaled_x", y)
    if ( str_detect(transform, "in_probas_domain")) {
        applied = append( applied, "gaussian_probabilities_transform" )
        y = 1.0 - as.matrix(apply( y, 2, PROB_X ), nrow(y))
        METRICS <<- TIMESTAMP( "GUASSIAN PROBS" )
    }

    VERIFY( "1-prob(scaled_x)", y)
    if ( str_detect(transform, "in_pca_domain")) {
        applied = append( applied, "pca_transform" )
        if (nrow(y)>5E4)
            pca = DO_PCA( y, vargoal=vargoal, do_scale=FALSE, ntries=3, nmax=min(1E3,nrow(y)), silent=TRUE )
        else
            pca = DO_PCA( y, vargoal=vargoal, do_scale=FALSE, silent=TRUE )
        y = pca$Z
        k = pca$which_k_to_us
        METRICS <<- TIMESTAMP( "PCA" )
    } else { 
        k = NA
        pca = list()
        pca[['Z']] = y
    }

    VERIFY( "pca(1-prob(scaled_x),vargoal)", y)
    if ( str_detect(dist_metric, "euclidean"    ) ) {
        applied = append( applied, "euclidean distances" )
        if ( nrow(y)^2 > 1E8 )
            z = DIST( y )
        else {
            z = dist( y,  method=dist_metric, upper=TRUE, diag=TRUE)
            z = as.matrix(z)
        } 
        METRICS <<- TIMESTAMP( "EUCLIDEAN DISTANCES" )
    }

    VERIFY( "dist(pca(1-prob(scaled_x),vargoal))", z)
    if ( do_plot ) {
        sampled_rows = sample(1:nrow(x),min(512,nrow(y)))
        zp = as.matrix(z)
        plot( zp[sampled_rows,] )
        METRICS <<- TIMESTAMP( "PLOT" )
    }

    print( applied )

    retvals = list( 'distances'=z, 'transforms_applied'=applied, 'pca_k'=k, 'pca_space'=pca$Z )
    return ( retvals )
}
# ###################################################################################################


# ###################################################################################################
DO_INIT_DISTANCE_MATRIX = function( x, ... ) {
    MY_DISTANCE_MATRIX_RETVALS <<- GET_DISTANCE_MATRIX ( x, ... )
    PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX <<- MY_DISTANCE_MATRIX_RETVALS$distances
    VERIFY( "PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX", PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX )
    return ( PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX )
}
# ###################################################################################################


# ###################################################################################################
GET_DISTANCE = function( X, i, j=0, ... ) {
    d_class = class(PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX)
    mx = nrow(PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX)
    my = ncol(PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX)
    if ( mx == 0 ) DO_INIT_DISTANCE_MATRIX( X, ... )
    if (j==0) j=c(1:my)
    ij_dist = PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX[i, j]
    return( ij_dist )
}
# ###################################################################################################


# ###################################################################################################
MIN_DISTANCE_FROM = function( X, i, do_plot=TRUE, ... ) {
    if ( class(PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX) != "dist" ) 
        DO_INIT_DISTANCE_MATRIX( X, ... )
    mx = nrow(as.matrix(PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX))
    ij_dist = c(as.matrix(PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX)[i, c(1:mx)])
    min_ij_val = 1E131
    min_ij_idx = NA
    for( j in 1:length(ij_dist) ) {
        if (ij_dist[j] <= min_ij_val) {
            if ( j!=i ) {
                min_ij_val = ij_dist[j]
                min_ij_idx = j
            }
        }
    }
    # min_ij_val = min(ij_dist)
    # min_ij_idx = which( ij_dist == min_ij_val )[1]
    Z = MY_DISTANCE_MATRIX_RETVALS$pca_space 
    if ( do_plot ) {
        dx= Z[i,1]/20  * rnorm(1,1,2)
        dy= Z[i,2]/20  * rnorm(1,1,2)
        plot(   x=Z[,1],            y=Z[,2],                                  pch=".", cex=1.5 )
        points( x=Z[i,1],           y=Z[i,2],          col="blue", bg="blue", pch=24,  cex=1.5 )
        points( x=Z[min_ij_idx,1],  y=Z[min_ij_idx,2], col="red",  bg="red",  pch=23,  cex=1.5 )
        text( x=Z[i,1]+dx,          y=Z[i,2],          "WRT", col="blue", cex=1.0 )
        text( x=Z[min_ij_idx,1]+dx, y=Z[min_ij_idx,2], "MIN", col="red",  cex=1.0 )
        if ( TRUE ) {
            try( { 
                    i1=s13$minidx
                    points( x=Z[i1,1],   y=Z[i1,2],       col="black", bg="green", pch=22, cex=1.0 )
                 })
        }
    }
    retvals = list( 'minidx'=as.integer(min_ij_idx), 'minval'=as.numeric(min_ij_val), 'minx'=X[min_ij_idx,])
    return( retvals )
}
# ###################################################################################################


# ###################################################################################################
METRICS <<- INIT_METRICS()
METRICS = TIMESTAMP( "START" )

    # ###################################################################################################
    M = 1999
    X = data.frame( x1=rnorm( M, 0, 3), x2=rnorm( M, 0, 5), x3=rnorm( M, 2, 3), x4=rnorm( M, 1, 5), 
             x5=rnorm( M, 2, 4), x6=rnorm( M, 2, 5), x7=rnorm( M, 1, 5), x8=rnorm( M, 3, 5))
    VERIFY( "X", X)
    # ###################################################################################################

    # ###################################################################################################
    s13 = NULL
    if ( TRUE ) {
        cat ( HEADER )
        DO_INIT_DISTANCE_MATRIX( X, dist_metric="euclidean", do_scaling=TRUE, do_plot=FALSE, transform="" )
        s13 = MIN_DISTANCE_FROM ( X, 2 )
        t1 = PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX[2,]
    }
    # ###################################################################################################


    # ###################################################################################################
    if ( TRUE ) {
        cat ( HEADER )
        DO_INIT_DISTANCE_MATRIX( X, dist_metric="euclidean", do_scaling=TRUE, do_plot=FALSE, transform="in_pca_domain,in_probas_domain", vargoal=0.55 )
        s33 = MIN_DISTANCE_FROM ( X, 2 )
        t3 = as.matrix(PRECOMPUTED_OPTIMIZED_TRANSFORMS_DISTANCE_MATRIX)[2,]
        cat ( HEADER )
        print( s33 )
        cat ( HEADER )
    }
    # ###################################################################################################


    # ###################################################################################################
    print( METRICS )
    # ###################################################################################################


MY_DISTANCE_MATRIX_RETVALS 
