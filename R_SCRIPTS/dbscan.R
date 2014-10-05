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
    WHICH  = as.numeric(which( D[,topitem] < EPSILON ))
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
TRANSITIVE_CLOSURE = function ( topitem, cid, debug=FALSE ) {
    WHICH  = GET_WHICH ( topitem )
    UPDATE = GET_UPDATE ( WHICH )
    UPDATE = VALIDATE_UPDATE( UPDATE, cid )
    UPDATE_VISITED( WHICH, cid )
    while ( length(UPDATE) > 0 ) {
        UPDATE = GET_UPDATE ( WHICH )
        UPDATE = VALIDATE_UPDATE( UPDATE, cid )
        WHICH  = UPDATE_WHICH( WHICH, UPDATE )
        UPDATE_VISITED( WHICH, cid )
    }
    if ( debug  ) print( paste( "LEN(WHICH)", length(WHICH), topitem, cid  ) )
    retvals = list( WHICH, length(WHICH), topitem, cid )
    return ( retvals )
}
# ############################################################################################


# ######################################################################################################
DO_DBSCAN_PLOT = function( ) {
    COLORS  = 1:length(CLUSTERS)
    SYMBOLS = 1:length(CLUSTERS)

    if ( ncol(X) == 2 ) {
        plot( X[,1], X[,2], 
            pch=".", col="gray", cex=0.8, 
            main=sprintf("DBSCAN W/ EPS= %.3f M=%s |C|=%s", EPSILON, MIN_SIZE, length(table(VISITED))) )
    } else {
        Z = DO_PCA(X, vargoal=0.95, ntries=2  )$Z
        plot( Z[,1], Z[,2], 
            pch=".", col="gray", cex=0.8, 
            main=sprintf("DBSCAN W/ EPS= %.3f M=%s |C|=%s", EPSILON, MIN_SIZE, length(table(VISITED))) )
    }
    grid()

    VALID = 1:M
    NOISE = which( VISITED == 0 )
    if ( length(NOISE) > 0 ) VALID = setdiff( VALID, NOISE )

    points( X[VALID,1], X[VALID,2], 
                pch=SYMBOLS[VISITED[VALID]] %% 25, 
                col=COLORS [VISITED[VALID]],
                cex=0.8 )

    points( X[NOISE,1], X[NOISE,2], 
                pch=".", 
                col="gray", 
                cex=0.5 )

}
# ######################################################################################################


# #################################################################################################
GET_INITIAL_EPSILON_ESTIMATE_FOR = function( X, D, NMIN=500, MINCON=100, debug=FALSE ) {
    NMIN   = min(NMIN,nrow(X))
    MINCON = min(0.10*M, MINCON)

    SUBSAMPLED = sample(1:nrow(D), min(nrow(D),NMIN))
    DD = D[SUBSAMPLED, ] 

    EPSILON = NA
    for ( EPSILON in  seq(STEP,1,STEP) ) {
        RS = rowSums(ifelse( DD<EPSILON, 1, 0 ))
        MEAN_CON = max(RS)
        print( paste ( EPSILON, MEAN_CON ) )
        if ( MEAN_CON > MINCON ) break
    }
    EPSILON = ifelse( is.na(EPSILON), 0.3, EPSILON )
    if ( debug ) {
        cat ( HEADER )
        print( t(rowSums(ifelse( D<EPSILON, 1, 0 ))) ) 
        cat ( HEADER )
    }
    return ( EPSILON )
}
# ######################################################################################################


# #################################################################################################
GET_CLUSTERED_X = function( MM=500 ) {
    # a 2-dimensional example
    MM = 500
    x <- rbind(matrix(rnorm(MM, mean = 0, sd = 0.5), ncol = 2),
                   matrix(rnorm(MM, mean = 2, sd = 0.3), ncol = 2),
                   matrix(rnorm(MM, mean = 4, sd = 0.4), ncol = 2),
                   matrix(rnorm(MM, mean = 1, sd = 0.6), ncol = 2),
                   matrix(rnorm(MM, mean = 3, sd = 0.3), ncol = 2))
    colnames(x) <- c("x", "y")
    rownames(x) <- 1:nrow(x)
    X = x
    X = scale( X )
    return ( X )
}
# #################################################################################################




















# #################################################################################################
X      = GET_CLUSTERED_X( 500 )
M      = nrow(X)
N      = ncol(X)
STEP   = 1E-2
DEBUG  = FALSE
# ############################################################################################

# ############################################################################################
DO_TEST= TRUE
# ############################################################################################

if ( DO_TEST ) {
    options( width=132 )
    sink( 'output.dbscan.out', split=T )
    graphics.off()

    pdf( 'plot_dbscan_cluster_assignments.pdf', 12, 8 )
    par( mfrow=c(2,2) )
    
    D = as.matrix( dist( X, upper=TRUE, diag=TRUE ) )

    ORIGINAL_EPSILON = GET_INITIAL_EPSILON_ESTIMATE_FOR( X, D )
    ORDERING_IN_USE  = sample(1:M,M)
    NOISE_THRESHOLDS = c(0,1,3,5,10,20,30,40)
    
    for ( MIN_SIZE in NOISE_THRESHOLDS ) {
    
        EPSILON = ORIGINAL_EPSILON
    
        VISITED = VECTOR(M); 
            rownames(VISITED) = rownames(X)
        
        CLUSTERS = c()
        
        cid = 1
        for ( topitem in ORDERING_IN_USE ) {
    
            if ( any(VISITED == 0) ) {
                if ( FALSE )
                    TRANSITIVE_CLOSURE_FROM( topitem, EPSILON, cid )
                else
                    TRANSITIVE_CLOSURE( topitem, cid )
    
                WHICH = which( VISITED == cid )
    
                if ( DEBUG ) 
                    print( paste( "length(which(table(VISITED)<MIN_SIZE))", 
                                   length(which(table(VISITED)<MIN_SIZE))))

                NOISE_CLUSTERS = names( which( table(VISITED)<=MIN_SIZE) )
                for ( i in NOISE_CLUSTERS ) {
                    VISITED[ VISITED==as.integer(i) ] = 0
                }
    
                # VALID_CLUSTERS = names( which( table(VISITED)>MIN_SIZE) )
                if ( DEBUG ) 
                    print( table(VISITED) )
    
                if ( length(WHICH) < MIN_SIZE ) { 
                    VISITED[WHICH] = 0
                    EPSILON = EPSILON + STEP/50
                } else {
                    CLUSTERS = append( CLUSTERS, cid )
                    cid = cid + 1
                    if ( DEBUG ) {
                        cat( HEADER )
                        print ( table ( VISITED ) )
                        cat( HEADER )
                        NEWLINE(3)
                    }
                }
    
                if ( length( VISITED==0 ) > 50 ) {
                    EPSILON = EPSILON + STEP/10
                    if ( DEBUG ) 
                        print( paste( "EPSILON INCREASED", EPSILON ) )
                }
    
            } else
                break
        }
        # ############################################################################################
        
        # ############################################################################################
        cat( HEADER )
        print( paste( "FINAL EPSILON=", EPSILON, "NOISE THRESHOLD=", MIN_SIZE ))
        print( table ( VISITED ) )
        cat( HEADER )
        cat( HEADER )
        NEWLINE(1)
        # ############################################################################################
        
        # ############################################################################################
        # compute cluster means
        # compute wss
        # compute error
        # if error going down
        # otherwise, bump size
        # else end
        # get new centroids from kmeans code
        # get the error
        # ############################################################################################
        
        DO_DBSCAN_PLOT( )
    
        if ( length( VISITED==0 ) > 30 ) {
            print( paste( "EPSILON INCREASED", EPSILON ) )
            EPSILON = EPSILON + STEP/10
        }
    
    }
    dev.off()
    sink()
}
# ######################################################################################################

