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


# ######################################################################################################
LICENSETEXT = "These R code samples (version Sep/2014), Copyright (C) Nelson R. Manohar,
comes with ABSOLUTELY NO WARRANTY.  This is free software, and you are welcome to 
redistribute it under conditions of the GNU General Public License." 
message( LICENSETEXT )
message("")
# ######################################################################################################


# ######################################################################################################
OLD_CA = commandArgs()
commandArgs <- function() list(TEST_ENABLED=FALSE, DO_DISTANCE_VALIDATION=FALSE, RUN_VALIDATE_SGD=FALSE )
# ######################################################################################################


# ######################################################################################################
source( 'utilities.R' )
source( 'datasets.R' )
source( 'fselect.R' )
# ######################################################################################################




# #################################################################################################
TRANSITIVE_CLOSURE_FROM = function( ITEM, EPSILON, CID, debug=0 ) {
    if ( debug > 1) 
        print( paste( "INITIATING RECURSION ON ITEM", ITEM, CID ))

    if (all(VISITED != 0)) {
        if ( debug > 1 ) 
            print( paste( "FINISHED  RECURSIONS" ) )
        return
    }

    if (VISITED[ITEM] != 0) {
        if ( debug > 2) 
            print( sprintf( 'ITEM %s ALREADY VISITED', ITEM, CID ) )
    } else {
        VISITED[ ITEM ] <<- CID

        WHICH_ONES = GET_WHICH( ITEM )
        WHICH_ONES = which( VISITED[WHICH_ONES] == 0 )

            if ( debug > 3 ) 
                print( paste( "PROCESSING CLOSURE FOR ITEM", ITEM, CID, "LENGTH", length(WHICH_ONES) ))

            if ( length( WHICH_ONES > 0 ) ) {
                if ( debug > 4 ) 
                    print( paste( "2] CLOSURE:", WHICH_ONES ) )

                for ( subitem in WHICH_ONES ) {
                    TRANSITIVE_CLOSURE_FROM( subitem, EPSILON, CID )
                }

            } else {
                if ( debug > 3 ) 
                    print( paste( "2] CLOSURE:", WHICH_ONES, "DONE" ) )
            }
    }
    if ( debug > 1 )print( paste( "FINISHED   RECURSION ON ITEM", ITEM, CID ))

    print( paste( "ITEM=", ITEM, "CID=", CID, "VISITED=", length(which(VISITED==CID))))
    # print( which( VISITED == CID ) )

    return
}
# #################################################################################################


# ############################################################################################
VALIDATE_UPDATE = function( UPDATE, cid ) {
    if ( length(UPDATE) != 0 ) {
        ALREADY_VISITED = which( VISITED !=0 )
        if ( length( ALREADY_VISITED ) == 0 ) ALREADY_VISITED = c()
        UPDATE = setdiff( UPDATE, ALREADY_VISITED )
    } else {
        UPDATE = c()
    }
    return ( UPDATE )
}
# ############################################################################################


# ############################################################################################
UPDATE_WHICH = function( WHICH, UPDATE ) {
    WHICH = append( WHICH, UPDATE )
    WHICH = unique( WHICH )
    return ( WHICH )
}
# ############################################################################################


# ############################################################################################
GET_WHICH = function( topitem ) {
    WHICH  = as.numeric(which( DISTANCE_MATRIX_X[,topitem] < EPSILON ))
    return ( WHICH )
}
# ############################################################################################


# ############################################################################################
GET_UPDATE = function( WHICH ) {
    UPDATE = c()
    for( i in WHICH ) {
        UPDATE = unique(append( UPDATE, GET_WHICH( i ) ))
    }
    return ( UPDATE )
}
# ############################################################################################


# ############################################################################################
UPDATE_VISITED = function( WHICH, cid ) {
    VISITED[ WHICH ] <<- cid
}
# ############################################################################################


# ############################################################################################
#### 0 1
TRANSITIVE_CLOSURE = function ( topitem, cid, MIN_LEADSIZE=5, debug=FALSE ) {
    MIN_LEADSIZE = MIN_SIZE
    WHICH  = GET_WHICH ( topitem )
    UPDATE = GET_UPDATE ( WHICH )
    UPDATE = VALIDATE_UPDATE( UPDATE, cid )

    while ( length(UPDATE) > MIN_LEADSIZE ) {
        WHICH  = UPDATE_WHICH( WHICH, UPDATE )
        UPDATE_VISITED( WHICH, cid )

        UPDATE = GET_UPDATE ( WHICH )
        UPDATE = VALIDATE_UPDATE( UPDATE, cid )
    }
    if ( debug  ) print( paste( "LEN(WHICH)", length(WHICH), topitem, cid  ) )
    retvals = list( WHICH, length(WHICH), topitem, cid )
    return ( retvals )
}
# ############################################################################################


# ######################################################################################################
DO_DBSCAN_PLOT = function( CLUSTERS ) {
    NOISE   = which( VISITED == 0 )
    VALID   = 1:M

    if ( ncol(X) == 2 ) {
        plot( X[,1], X[,2], 
            pch=".", col="gray", cex=0.8, 
            main=sprintf("DBSCAN W/ EPS= %.3f M=%s |C|=%s", EPSILON, MIN_SIZE, length(table(VISITED)) ))
    } else {
        Z = DO_PCA(X, vargoal=0.95, ntries=2  )$Z
        plot( Z[,1], Z[,2], 
            pch=".", col="gray", cex=0.8, 
            main=sprintf("DBSCAN W/ EPS= %.3f M=%s |C|=%s", EPSILON, MIN_SIZE, length(table(VISITED)) ))
    }
    grid()

    if ( length(NOISE) > 0 ) 
        VALID = setdiff( VALID, NOISE )

    points( X[VALID,1], X[VALID,2], 
                pch=(10 + VISITED[VALID] %% 5), 
                col=(20 + VISITED[VALID] %% 256 ),
                bg =(40 + VISITED[VALID] %% 256 ),
                cex=1.2 )

    if ( length(NOISE) > 0 ) 
        points( X[NOISE,1], X[NOISE,2], 
                pch=".", 
                col="gray", 
                cex=0.5 )

}
# ######################################################################################################


# #################################################################################################
# for ( e in seq(0.05,0.6,0.05)) hist( rowSums(DISTANCE_MATRIX_X < e ))
# #################################################################################################
GET_INITIAL_EPSILON_ESTIMATE_FOR = function( X, D, epsilon=NA, NMIN=500, q=1.0, debug=FALSE ) {
    NMIN   = min(NMIN,nrow(X))
    MINCON = min(0.05*M, sqrt(M))

    SUBSAMPLED = sample(1:nrow(D), min(nrow(D),NMIN))
    DD = DISTANCE_MATRIX_X[SUBSAMPLED, ] 

    EPSILON = NA
    for ( EPSILON in  seq(STEP,1,STEP) ) {
        OUT_DEGREES = rowSums(ifelse( DD<EPSILON, 1, 0 ))
        OUTDEGREE_STAT = quantile(OUT_DEGREES, c(q))[1]
        print( paste ( EPSILON, OUTDEGREE_STAT, MINCON ) )
        if ( OUTDEGREE_STAT > MINCON ) break
    }
    EPSILON = ifelse( is.na(EPSILON), 0.3, EPSILON )
    if ( debug ) {
        cat ( HEADER )
        print( t(rowSums(ifelse( DISTANCE_MATRIX_X<EPSILON, 1, 0 ))) ) 
        cat ( HEADER )
    }

    if ( !is.na(epsilon) )
        if ( EPSILON < epsilon ) 
            EPSILON = epsilon

    return ( EPSILON )
}
# ######################################################################################################


# #################################################################################################
GET_CLUSTERED_X = function( MM=100, NN=5, N=2, SD_X, MU_X ) {
    cat ( HEADER )
    print( sprintf( "X comprises: %s samples sets from %s normal random sources across a %s dimensional space", MM, NN, N ) ) 
    print( sprintf( "MU[%s]=%.2f", 1:NN, MU_X ) )
    print( sprintf( "SD[%s]=%.2f", 1:NN, SD_X ) )
    cat ( HEADER )

    # a 2-dimensional example
    x <- rbind(matrix(rnorm(MM, mean = MU_X[1], sd = SD_X[1]), ncol = 2),
               matrix(rnorm(MM, mean = MU_X[2], sd = SD_X[2]), ncol = 2),
               matrix(rnorm(MM, mean = MU_X[3], sd = SD_X[3]), ncol = 2),
               matrix(rnorm(MM, mean = MU_X[4], sd = SD_X[4]), ncol = 2),
               matrix(rnorm(MM, mean = MU_X[5], sd = SD_X[5]), ncol = 2))
    colnames(x) <- c("x", "y")
    rownames(x) <- 1:nrow(x)
    X = x
    X = scale( X )
    return ( X )
}
# #################################################################################################


# ######################################################################################################
UPDATE_EPSILON = function( CHANGE, debug=FALSE ) { 
    EPSILON <<- EPSILON + CHANGE 

    if ( debug ) 
        print( paste( "EPSILON INCREASED", EPSILON ) )

    return ( EPSILON )
}
# ######################################################################################################


# ######################################################################################################
DO_DBSCAN = function( X, D, MIN_SIZE, ORDERING_IN_USE, debug=FALSE ) {
    CLUSTERS = c()
    CID = 1
    for ( TOPITEM in ORDERING_IN_USE ) {
      if ( any(VISITED == 0)) {
         # if ( any(VISITED[ which(DISTANCE_MATRIX_X[TOPITEM,] < (100*EPSILON)) ] == 0) ) {

            TRANSITIVE_CLOSURE( TOPITEM, CID )

            WHICH = which( VISITED == CID )
            if ( debug ) print( paste( "length(which(table(VISITED)<MIN_SIZE))", length(which(table(VISITED)<MIN_SIZE))))

            NOISE_CLUSTERS = names( which( table(VISITED)<=MIN_SIZE) )
            VALID_CLUSTERS = names( which( table(VISITED)>MIN_SIZE) )

            for ( i in NOISE_CLUSTERS )
                VISITED[ VISITED==as.integer(i) ] <<- 0

            if ( debug ) print( table(VISITED) )

            if ( length(WHICH) < MIN_SIZE ) { 
                for ( item in WHICH ) {
                    nearby = as.integer(names(which( DISTANCE_MATRIX_X[,item] < EPSILON )))
                    nearby_itemized = table(VISITED[nearby])
                    if ( max(nearby_itemized) > MIN_SIZE ) {
                        BEST_CID = as.integer(names(which( nearby_itemized == max(nearby_itemized) )))
                        UPDATE_VISITED( item, BEST_CID )
                    } else
                        UPDATE_VISITED( item, 0 )
                }
                # EPSILON = UPDATE_EPSILON( STEP/16)
            } 
            else {
                CLUSTERS = append( CLUSTERS, CID )
                CID = CID + 1
                if ( debug ) {
                    cat( HEADER )
                    print ( table ( VISITED ) )
                    cat( HEADER )
                    NEWLINE(3)
                }
            }

            NOISE = length( which(VISITED==0) )
            if ( NOISE > 2*sqrt(M) )                # it adapts heavily at beginning and slows down as it progresses 
                EPSILON = UPDATE_EPSILON( STEP/10 )

        } else
            break
    }

    DO_DBSCAN_PLOT( CLUSTERS )
    PRINT_SUMMARY( CLUSTERS )
    RETVALS = list( 'EPSILON'=EPSILON, 'MAPPINGS'=VISITED, 'CLUSTERS'=CLUSTERS, 'NOISE'=NOISE )

    return ( RETVALS )

}
# ######################################################################################################


# ######################################################################################################
PRINT_SUMMARY =function( CLUSTERS, debug=TRUE ) {
    cat( HEADER )
    C     = table ( VISITED )
    if ( debug ) {
        NOISE = length( which(VISITED==0) )
        CENTROIDS = GET_CENTROIDS_FOR( X, VISITED )
        TOTAL_WSS_SUM = sum(GET_MSE_FROM_CENTROIDS_FOR(X, VISITED, CENTROIDS=CENTROIDS, GET_WSS=TRUE ))
        TOTAL_MSE_SUM = sum(GET_MSE_FROM_CENTROIDS_FOR(X, VISITED, CENTROIDS=CENTROIDS, GET_WSS=FALSE ))
        AVG_BETWEEN_CLUSTER_DIST = mean(GET_DISTANCE_BETWEEN_CENTROIDS( CENTROIDS ))
        METRIC=round((TOTAL_WSS_SUM + TOTAL_MSE_SUM * length(C) - AVG_BETWEEN_CLUSTER_DIST ) * length(C),2)
        print( paste( "FINAL EPSILON=",   round(EPSILON,3),
                  "NOISE THRESHOLD=", MIN_SIZE, "NOISE=", NOISE, "|C|=", length(C), 
                  'SUM(WSS)=', round(TOTAL_WSS_SUM,2), 
                  'AVG(WSS)=', round(TOTAL_MSE_SUM,2),
                  'AVG(ICD)=', round(AVG_BETWEEN_CLUSTER_DIST,2),
                  'METRIC=', METRIC ))
    }
    print( C )
    cat( HEADER )
    NEWLINE(1)
}
# ######################################################################################################


# ######################################################################################################
GET_CENTROIDS_FOR = function( X, MAPPING ) {
    C    = table( MAPPING )
    CIDS = as.numeric(names(C))

    NC = length(CIDS)
    NF = ncol(X)
    CENTROIDS = MATRIX( NC, NF ) 

    for ( i in 1:NC) {
        WHICH_ONES    = which(MAPPING==CIDS[i])
        CENTROIDS[i,] = mean(X[WHICH_ONES,])
    }
    return ( CENTROIDS )
}
# ######################################################################################################


# ######################################################################################################
GET_MSE_FROM_CENTROIDS_FOR = function( X, MAPPING, CENTROIDS=matrix(), GET_WSS=TRUE ) {
    if ( nrow(CENTROIDS) == 0 )
        CENTROIDS = GET_CENTROIDS_FOR( X, MAPPING )
    NC   = nrow( CENTROIDS )
    WSS  = VECTOR(NC)
    C    = table( MAPPING )
    CIDS = as.numeric(names(C))
    for ( i in 1:NC) {
        if ( CIDS[i] == 0 ) next
        WHICH_ONES    = which(MAPPING==CIDS[i])
        if ( GET_WSS )
            WSS[i] = sum(rowSums( X[WHICH_ONES,] - CENTROIDS[i,])^2)
        else
            WSS[i] = sum(rowSums( X[WHICH_ONES,] - CENTROIDS[i,])^2) /length(WHICH_ONES)
    }
    return ( WSS )
}
# ######################################################################################################


# ######################################################################################################
GET_DISTANCE_BETWEEN_CENTROIDS = function( CENTROIDS ) {
    CD = as.matrix(dist( CENTROIDS, upper=TRUE, diag=TRUE ))
    return ( CD )
}
# ######################################################################################################












# ############################################################################################
DO_TEST= TRUE
# ############################################################################################


# ######################################################################################################
if ( DO_TEST ) {
    # ############################################################################################
    options( width=132 )
    sink( 'output.dbscan.out', split=T )
    graphics.off()
    pdf( 'plot_dbscan_cluster_assignments.pdf', 12, 8 )
    par( mfrow=c(2,2) )
    # ############################################################################################
    
    # ############################################################################################
    SD = 0.2
    SD_X <<- c(SD+.1, SD+0.1, SD+0.1, SD+0.1, SD+0.1 )
    MU_X <<- c(    0,      1,      2,      3,      4 )
    X      = GET_CLUSTERED_X( MM=400, NN=5, N=2, SD_X, MU_X )
    M      = nrow(X)
    N      = ncol(X)
    STEP   = 1E-2
    # ############################################################################################

    # ############################################################################################
    DISTANCE_MATRIX_X = as.matrix( dist( X, upper=TRUE, diag=TRUE ) )
    ORIGINAL_EPSILON  = GET_INITIAL_EPSILON_ESTIMATE_FOR( X, DISTANCE_MATRIX_X, epsilon=0.15, q=0.99 )
    ORDERING_IN_USE   = as.numeric(names( sort( rowSums(DISTANCE_MATRIX_X<ORIGINAL_EPSILON), decreasing=TRUE ) ))          
    # ############################################################################################

    # ############################################################################################
    NOISE_THRESHOLDS = c(5,10,15,20)
    for ( MIN_SIZE in NOISE_THRESHOLDS ) {
        EPSILON = ORIGINAL_EPSILON
        VISITED <<- VECTOR(M); rownames(VISITED) = rownames(X)
        DBSCAN_RETVALS = DO_DBSCAN( X, D, MIN_SIZE, ORDERING_IN_USE )
    }
    # ############################################################################################

    dev.off()
    sink()
}
# ######################################################################################################

