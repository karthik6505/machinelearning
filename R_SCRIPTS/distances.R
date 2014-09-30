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
source( 'datasets.R' )
# #####################################################################################


# #####################################################################################
DO_DISTANCE_VALIDATION = TRUE
# #####################################################################################
    V = CHECK_COMMAND_ARGS( commandArgs(), "DO_DISTANCE_VALIDATION" )
    if ( V$'VALID'==TRUE & V$'FOUND'==TRUE ) 
        DO_DISTANCE_VALIDATION = V$'ARGVAL'
# #####################################################################################


# ###################################################################################################
# BROKEN
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

    VERIFY( "DISTANCE_MATRIX_INCREMENTAL", DISTANCE_MATRIX_INCREMENTAL )

    return( DISTANCE_MATRIX_INCREMENTAL )
}
# ###################################################################################################


# ###################################################################################################
GET_DISTANCE_PCA = function( y, vargoal ) { 
    if (nrow(y)>5E4)
        pca = DO_PCA( y, vargoal=vargoal, do_scale=FALSE, ntries=3, nmax=min(1E3,nrow(y)), silent=TRUE )
    else
        pca = DO_PCA( y, vargoal=vargoal, do_scale=FALSE, silent=TRUE )
    return ( pca )
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
        pca = GET_DISTANCE_PCA( y, vargoal, ... )
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
        # if ( nrow(y)^2 > 1E5 )
        #     z = DIST( y )
        # else {
        # } 
        z = dist( y,  method=dist_metric, upper=TRUE, diag=TRUE)
        z = as.matrix(z)
        METRICS <<- TIMESTAMP( "EUCLIDEAN DISTANCES" )
    } else {
        z = dist( y,  method="manhattan", upper=TRUE, diag=TRUE)
    }

    VERIFY( "dist(pca(1-prob(scaled_x),vargoal))", z)
    if ( do_plot ) {
        DO_PCA_FULL_PLOT( z, nmax=512, cex=1.0, pch="+" )
        METRICS <<- TIMESTAMP( "PLOT" )
    }

    print( applied )

    retvals = list( 'distances'=z, 'transforms_applied'=applied, 'pca_k'=k, 'pca_space'=pca$Z )
    return ( retvals )
}
# ###################################################################################################


# ###################################################################################################
GET_DISTANCE = function( X, i, j=0, D=matrix() ) {
    mx = nrow(D)
    my = ncol(D)
    if (j==0) j=c(1:my)
    if ( mx != 0 ) 
        ij_dist = D[i, j]
    return( ij_dist )
}
# ###################################################################################################


# ###################################################################################################
MIN_DISTANCE_FROM = function( X, i, D=matrix(), PCA=matrix(), do_plot=TRUE, debug_=FALSE, ... ) {
    mx = nrow(as.matrix(D))
    ij_dist = c(as.matrix(D)[i, c(1:mx)])
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

    if ( nrow(PCA)== 0 ) {
        if ( nrow(D)!= 0 ) Z = DO_PCA(D, ...)$Z
    } else 
        Z = PCA

    if ( nrow(Z) && do_plot )
        DO_PCA_NEIGHBORING_PLOT( i, min_ij_idx, Z )

    retvals = list( 'minidx'=as.integer(min_ij_idx), 'minval'=as.numeric(min_ij_val), 'minx'=X[min_ij_idx,])
    return( retvals )
}
# ###################################################################################################


# ###################################################################################################
COLOR_SHIFT = 24
SHIFTCOLOR = function( ) { COLOR_SHIFT<<-(COLOR_SHIFT+1)%%400 + 24 }
# ###################################################################################################


# ###################################################################################################
FIND_NEIGHBORING_POINTS = function( X, ix, D=matrix(), PCA=matrix(), new_plot=FALSE, color="brown", debug=FALSE ) {
    cat( HEADER )
    print( sprintf("LOOKING FOR SIMILAR ITEMS TO %s",  ix ) )

    x_ij_distances = GET_DISTANCE( X, ix, j=0, D=D )
        mu    = median( x_ij_distances)
        sigma = sd( x_ij_distances )

    xmin = names(which( x_ij_distances == sort(x_ij_distances)[2])[1])

    xtimes = 3; nearby=c(); i = 0
    while ( length(nearby)<2 ) {
        i = i + 1
        sigma_level = xtimes * sigma
        nearby  = names(which(x_ij_distances < (mu- sigma_level)))
        xtimes = xtimes * 4/5
        if ( debug ) print( paste( i, sigma_level, length(nearby), xtimes ))
    }

    print( sprintf("IDENTIFIED %d/%d=(%.2f%%) SIMILAR USERS/MOVIES AT %.2f SIGMAS AWAY FROM MEDIAN", 
                   length(nearby)-1, 
                   length(rownames(X)), 
                   (length(nearby)-1)/length(rownames(X))*100,
                    sigma_level/sigma ) )

    set_coloring = SHIFTCOLOR()
    for ( ij in nearby ) { 
        if ( ij == ix ) next
        DO_PCA_NEIGHBORING_PLOT( ix, ij, PCA, new_plot=FALSE, color=set_coloring )
        if ( debug ) print ( paste( ix, x_ij_distances[ij], ij )) 
    }

    retvals = list( 'from'=ix, 'closest'=xmin, 'mu'=mu, 'sigmas_away_from_median'=sigma_level/sigma, 'nearby'=nearby )

    return ( retvals )
}
# ###################################################################################################


# ###################################################################################################
DO_PCA_FULL_PLOT = function( Z, nmax=0, pch="+", ... ) {
    zp = as.matrix(Z)
    if ( nmax== 0 ) nmax = nrow(Z)
    nmax = min(nmax, 5E3, nrow(Z))
    sampled_rows = sample(rownames(Z),nmax)
    plot(x=as.matrix(Z[,1]), y=as.matrix(Z[,2]), ... )
}
# ###################################################################################################


# ###################################################################################################
DO_PCA_NEIGHBORING_PLOT = function( i, min_ij_idx, Z, new_plot=FALSE, color="green" ) {
    if ( nrow(Z) ) {
        dx= Z[i,1]/20  * rnorm(1,1,2)
        dy= Z[i,2]/20  * rnorm(1,1,2)

        x0 = Z[i,1]
        y0 = Z[i,2]
        x1 = Z[min_ij_idx,1]
        y1 = Z[min_ij_idx,2]

        segments( x0, y0, x1=x1, y1=y1, col=color, lwd=1 )

        if ( new_plot ) {
            x1lab = paste( i )
            points( x=x0, y=y0, col="blue", bg="blue", pch=24,  cex=0.7 ) # points( x=Z[i,1], y=Z[i,2], col="blue", bg="blue", pch=24,  cex=0.7 )
            text( x=x0+dx, y=y0, x1lab, col="blue", cex=0.4, srt=30 )     # text( x=Z[i,1]+dx, y=Z[i,2], x1lab, col="blue", cex=0.7 )
        }

        points( x=x1, y=y1, col=color, bg=color, pch=23, cex=0.7 )        # points(x=Z[min_ij_idx,1],y=Z[min_ij_idx,2],col=color,bg=color,pch=23,cex=0.7 )
        # x2lab = paste( "min=", min_ij_idx )
        # text( x=x1+dx, y=y1, x2lab, col=color, cex=0.4, srt=45 )        # text( x=Z[min_ij_idx,1]+dx, y=Z[min_ij_idx,2], x2lab, col=color, cex=0.7 )
    }
}
# ###################################################################################################


# ###################################################################################################
METRICS <<- INIT_METRICS()
METRICS = TIMESTAMP( "START" )

if ( DO_DISTANCE_VALIDATION ) {
    sink( 'output.distances.out', split=TRUE )
    # ###################################################################################################
    M = 3000
    X = data.frame( x1=rnorm( M, 0, 3), x2=rnorm( M, 0, 5), x3=rnorm( M, 2, 3), x4=rnorm( M, 1, 5), 
                    x5=rnorm( M, 2, 4), x6=rnorm( M, 2, 5), x7=rnorm( M, 1, 5), x8=rnorm( M, 3, 5))
    VERIFY( "X", X)
    # ###################################################################################################

    # ###################################################################################################
    s13 = NULL
    if ( TRUE ) {
        cat ( HEADER )
        MY_DISTANCE_MATRIX_RETVALS = GET_DISTANCE_MATRIX ( X, dist_metric="euclidean", do_scaling=TRUE, do_plot=FALSE, transform="" )
        D1  = as.matrix(MY_DISTANCE_MATRIX_RETVALS$distances)
        PCA1= MY_DISTANCE_MATRIX_RETVALS$pca_space
        s13 = MIN_DISTANCE_FROM ( X, 2, D=D1, PCA=PCA1 )
        t1  = D1[2,]
        cat ( HEADER )
        print( s13 )
        cat ( HEADER )
    }
    # ###################################################################################################

    # ###################################################################################################
    if ( TRUE ) {
        cat ( HEADER )
        MY_DISTANCE_MATRIX_RETVALS  = GET_DISTANCE_MATRIX ( X, dist_metric="euclidean", do_scaling=TRUE, do_plot=FALSE, transform="in_pca_domain,in_probas_domain", vargoal=0.55 )
        D3  = as.matrix(MY_DISTANCE_MATRIX_RETVALS$distances)
        PCA3= MY_DISTANCE_MATRIX_RETVALS$pca_space
        s33 = MIN_DISTANCE_FROM ( X, 2, D=D3, PCA=PCA3 )
        t3  = D3[2,]
        cat ( HEADER )
        print( s33 )
        cat ( HEADER )
    }
    # ###################################################################################################

    # ###################################################################################################
    print( METRICS )
    # ###################################################################################################

    # ###################################################################################################

    MY_DISTANCE_MATRIX_RETVALS 

    sink()

    MY_DISTANCE_MATRIX_RETVALS 
}

# ###################################################################################################
# ###################################################################################################
# ###################################################################################################

