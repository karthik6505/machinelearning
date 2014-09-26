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
# http://www.salemmarafi.com/code/collaborative-filtering-r/
# ######################################################################################################

# ######################################################################################################
# Load the libraries
library(arules)
library(arulesViz)
library(datasets)
# ######################################################################################################


# ######################################################################################################
# Load the data set
# data(Groceries)
# ######################################################################################################


# ######################################################################################################
# Formal class 'rules' [package "arules"] with 4 slots
#   ..@ lhs    :Formal class 'itemMatrix' [package "arules"] with 3 slots
#   .. .. ..@ data       :Formal class 'ngCMatrix' [package "Matrix"] with 5 slots
#   .. .. .. .. ..@ i       : int [1:526] 596 281 474 124 124 234 275 741 741 495 ...
#   .. .. .. .. ..@ p       : int [1:378] 0 1 2 3 4 5 6 7 8 9 ...
#   .. .. .. .. ..@ Dim     : int [1:2] 1682 377
#   .. .. .. .. ..@ Dimnames:List of 2
#   .. .. .. .. .. ..$ : NULL
#   .. .. .. .. .. ..$ : NULL
#   .. .. .. .. ..@ factors : list()
#   .. .. ..@ itemInfo   :'data.frame':	1682 obs. of  1 variable:
#   .. .. .. ..$ labels:Class 'AsIs'  chr [1:1682] "M1" "M2" "M3" "M4" ...
#   .. .. ..@ itemsetInfo:'data.frame':	0 obs. of  0 variables
#   ..@ rhs    :Formal class 'itemMatrix' [package "arules"] with 3 slots
#   .. .. ..@ data       :Formal class 'ngCMatrix' [package "Matrix"] with 5 slots
#   .. .. .. .. ..@ i       : int [1:377] 120 49 99 0 120 49 99 120 49 173 ...
#   .. .. .. .. ..@ p       : int [1:378] 0 1 2 3 4 5 6 7 8 9 ...
#   .. .. .. .. ..@ Dim     : int [1:2] 1682 377
#   .. .. .. .. ..@ Dimnames:List of 2
#   .. .. .. .. .. ..$ : NULL
#   .. .. .. .. .. ..$ : NULL
#   .. .. .. .. ..@ factors : list()
#   .. .. ..@ itemInfo   :'data.frame':	1682 obs. of  1 variable:
#   .. .. .. ..$ labels:Class 'AsIs'  chr [1:1682] "M1" "M2" "M3" "M4" ...
#   .. .. ..@ itemsetInfo:'data.frame':	0 obs. of  0 variables
#   ..@ quality:'data.frame':	377 obs. of  3 variables:
#   .. ..$ support   : num [1:377] 0.207 0.203 0.229 0.209 0.21 ...
#   .. ..$ confidence: num [1:377] 0.947 0.823 0.864 0.807 0.811 ...
#   .. ..$ lift      : num [1:377] 2.08 1.33 1.6 1.68 1.78 ...
#   ..@ info   :List of 4
#   .. ..$ data         : symbol T
#   .. ..$ ntransactions: int 943
#   .. ..$ support      : num 0.2
#   .. ..$ confidence   : num 0.8
# ######################################################################################################

# ##############################################################################################################
library(arules)
source( 'utilities.R' )

# #####################################################################################
TEST_ENABLED = TRUE
# #####################################################################################
    V = CHECK_COMMAND_ARGS( commandArgs(), "TEST_ENABLED" )
    if ( V$'VALID'==TRUE & V$'FOUND'==TRUE ) 
        TEST_ENABLED = V$'ARGVAL'
# #####################################################################################
# ##############################################################################################################


# ##############################################################################################################
PRINT_RULESET = function( rules0, debug=FALSE ) {
    cat(HEADER)
    summary(rules0)
    rules0<-sort(rules0, decreasing=TRUE,by="lift")
    inspect( rules0 )
    return
}
# ##############################################################################################################


# ##############################################################################################################
FIND_RULES_FOR_THIS_ITEM = function( RR, ITEM, SUPPORT, CONF, STEP=1/3, MIN_SUPPORT=0.01, debug=FALSE, silent=FALSE ) {
    CONTINUE = TRUE
    FOUND_ONE= FALSE

    rules = matrix()
    while ( CONTINUE ) {
        new_rules = apriori(data=RR, parameter=list(supp=SUPPORT,conf = CONF), appearance = list(default="rhs", lhs=ITEM), control=list(verbose=FALSE)) 

        if ( !FOUND_ONE & length( new_rules ) != 0 ) 
            FOUND_ONE=TRUE
        if ( FOUND_ONE &  length( new_rules ) == 0 ) 
            break

        if ( !FOUND_ONE ) 
            NEW_SUPPORT = (1-STEP) * SUPPORT
        if ( FOUND_ONE ) 
            NEW_SUPPORT = (1+STEP) * SUPPORT

        if ( NEW_SUPPORT < MIN_SUPPORT )
            break
        if ( NEW_SUPPORT > 1 )
            break

        if ( debug ) print( paste( "FOUND=", FOUND_ONE, "SUPPORT=", SUPPORT, "CONTINUE=", CONTINUE, length(new_rules) ))

        rules   = new_rules
        SUPPORT = NEW_SUPPORT
    }

    # see if confidence can be increased while retaining some knowledge about item
    while ( length(rules) > 0 ) {
        if ( debug ) print( paste( "FOUND=", FOUND_ONE, "SUPPORT=", SUPPORT, "CONTINUE=", CONTINUE, "CONF=", CONF, length(rules) ))

        CONF = (1+STEP) * CONF
        if ( CONF >= 1 )
            break

        new_rules = apriori(data=RR, parameter=list(supp=SUPPORT,conf = CONF), appearance = list(default="rhs", lhs=ITEM), control=list(verbose=FALSE) )
        if ( length(new_rules) == 0 ) 
            break

        rules    = new_rules
    }

    if ( !silent ) PRINT_RULESET( rules )

    return ( rules )
}
# ##############################################################################################################


# ##############################################################################################################
PRINT_FINDINGS = function( findings ) {
    itemsets_rhs = as(attr(findings,"rhs"), "list" )
    quality = as(attr(findings,"quality"), "list" )
    nmax = min(3,length(itemsets_rhs))
    for ( i in 1:nmax) {
        M2_MOVIE = itemsets_rhs[[i]][1]
        M2 = M2ID( M2_MOVIE )
        support = quality[[1]][i]
        confidence = quality[[2]][i]
        lift    = quality[[3]][i]
        print( sprintf( "%5s-->%5s: [%5.2f, %5.2f, %5.2f] %s", MOVIE, M2_MOVIE,
                       support, confidence, lift,
                       as.character(M[M2,'movie_title'] )))
    }
}
# ##############################################################################################################


# ##############################################################################################################
APPLY_MARKET_BASKET_ASSOCIATON_ANALYSIS_WRT = function( RR, MOVIE, SUPPORT=0.1, CONF=0.51, STEP=0.10, MIN_SUPPORT=0.01 ) { 
    RR = ifelse( is.na(RR), 0, RR )
    RR = ifelse( RR<5,   0, RR )
    RR = ifelse( RR<3.5, 0, 1 )

    M1 = M2ID( MOVIE )
    print( paste( MOVIE, as.character(M[M1,'movie_title'] )))

    findings = FIND_RULES_FOR_THIS_ITEM( RR, MOVIE, SUPPORT, CONF, STEP, MIN_SUPPORT, silent=TRUE )

    PRINT_FINDINGS( findings )

    return ( findings )
}
# ##############################################################################################################


# ##############################################################################################################
if ( TRUE )
    for ( MOVIE in sample(MOVIE_LABELS,400) ) {
        cat(HEADER)
        R4R = RECOMMENDATIONS_FOR_U2M_RATINGS
        findings = APPLY_MARKET_BASKET_ASSOCIATON_ANALYSIS_WRT( R4R, 
                                                                MOVIE, 
                                                                SUPPORT=0.1,
                                                                CONF=0.51, 
                                                                STEP=0.10, 
                                                                MIN_SUPPORT=2/nrow(R4R) )
        cat(HEADER)
        cat(HEADER)
        cat(HEADER)
        NEWLINE(3)
        # plot(findings, method="graph",interactive=FALSE)
    }
# ##############################################################################################################


