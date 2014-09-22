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


# ###############################################################################
source( 'utilities.R' )
source( 'plot_functions.R' )
# ###############################################################################


# ###############################################################################
# TODO
# pca implementation with verification (with subset and iterator if necessary for scalability)
# iterators for tests with variable sampling so as to address scalability and dominance of finding on repeated trials
# ###############################################################################


# ###############################################################################
# simple balanced anova with normality check
# ###############################################################################
# ANOVA( formula, data, 
# Call: aov(formula = tip ~ day - 1, data = data)
# Terms:                day Residuals
# Sum of Squares  2203.0066  455.6866
# Deg. of Freedom         4       240
# Residual standard error: 1.377931
# Estimated effects are balanced
# ###############################################################################
# 
#            Df Sum Sq Mean Sq F value Pr(>F)    
# day         4 2203.0   550.8   290.1 <2e-16 ***
# Residuals 240  455.7     1.9                   
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
#   dayFri   daySat   daySun  dayThur 
# 2.734737 2.993103 3.255132 2.771452 
# ###############################################################################
# List of 13
#  $ coefficients : Named num [1:4] 2.73 2.99 3.26 2.77                           ..- attr(*, "names")= chr [1:4] "dayFri" "daySat" "daySun" "dayThur"
#  $ residuals    : Named num [1:244] -2.2451 -1.5951 0.2449 0.0549 0.3549 ...    ..- attr(*, "names")= chr [1:244] "1" "2" "3" "4" ...
#  $ effects      : Named num [1:244] -11.92 -27.92 -28.38 -21.82 0.33 ...        ..- attr(*, "names")= chr [1:244] "dayFri" "daySat" "daySun" "dayThur" ...
#  $ rank         : int 4
#  $ fitted.values: Named num [1:244] 3.26 3.26 3.26 3.26 3.26 ...                ..- attr(*, "names")= chr [1:244] "1" "2" "3" "4" ...
#  $ assign       : int [1:4] 1 1 1 1
#  $ qr           :List of 5                                                      ..$ qr   : num [1:244, 1:4] -4.36 0 0 0 0 ...
#                                                                                 .. ..- attr(*, "dimnames")=List of 2
#                                                                                 .. .. ..$ : chr [1:244] "1" "2" "3" "4" ...
#                                                                                 .. .. ..$ : chr [1:4] "dayFri" "daySat" "daySun" "dayThur" 
#                                                                                 .. ..- attr(*, "assign")= int [1:4] 1 1 1 1
#                                                                                 .. ..- attr(*, "contrasts")=List of 1
#                                                                                 .. .. ..$ day: chr "contr.treatment"
#                                                                                 ..$ qraux: num [1:4] 1 1 1.11 1
#                                                                                 ..$ pivot: int [1:4] 1 2 3 4
#                                                                                 ..$ tol  : num 1e-07
#                                                                                 ..$ rank : int 4
#                                                                                 ..- attr(*, "class")= chr "qr"
#  $ df.residual  : int 240
#  $ contrasts    :List of 1                                                      ..$ day: chr "contr.treatment"
#  $ xlevels      :List of 1                                                      ..$ day: chr [1:4] "Fri" "Sat" "Sun" "Thur"
#  $ call         : language aov(formula = tip ~ day - 1, data = data)
#  $ terms        :Classes 'terms', 'formula' length 3 tip ~ day - 1              .. ..- attr(*, "variables")= language list(tip, day)
#                                                                                 .. ..- attr(*, "factors")= int [1:2, 1] 0 1
#                                                                                 .. .. ..- attr(*, "dimnames")=List of 2
#                                                                                 .. .. .. ..$ : chr [1:2] "tip" "day"
#                                                                                 .. .. .. ..$ : chr "day"
#                                                                                 .. ..- attr(*, "term.labels")= chr "day"
#                                                                                 .. ..- attr(*, "order")= int 1
#                                                                                 .. ..- attr(*, "intercept")= int 0
#                                                                                 .. ..- attr(*, "response")= int 1
#                                                                                 .. ..- attr(*, ".Environment")=<environment: R_GlobalEnv> 
#                                                                                 .. ..- attr(*, "predvars")= language list(tip, day)
#                                                                                 .. ..- attr(*, "dataClasses")= Named chr [1:2] "numeric" "factor"
#                                                                                 .. .. ..- attr(*, "names")= chr [1:2] "tip" "day"
#  $ model        :'data.frame':	244 obs. of  2 variables:                       ..$ tip: num [1:244] 1.01 1.66 3.5 3.31 3.61 4.71 2 3.12 1.96 3.23 ...
#                                                                                 ..$ day: Factor w/ 4 levels "Fri","Sat","Sun",..: 3 3 3 3 3 3 3 3 3 3 ...
#                                                                                 ..- attr(*, "terms")=Classes 'terms', 'formula' length 3 tip ~ day - 1
#                                                                                 .. .. ..- attr(*, "variables")= language list(tip, day)
#                                                                                 .. .. ..- attr(*, "factors")= int [1:2, 1] 0 1
#                                                                                 .. .. .. ..- attr(*, "dimnames")=List of 2
#                                                                                 .. .. .. .. ..$ : chr [1:2] "tip" "day"
#                                                                                 .. .. .. .. ..$ : chr "day"
#                                                                                 .. .. ..- attr(*, "term.labels")= chr "day"
#                                                                                 .. .. ..- attr(*, "order")= int 1
#                                                                                 .. .. ..- attr(*, "intercept")= int 0
#                                                                                 .. .. ..- attr(*, "response")= int 1
#                                                                                 .. .. ..- attr(*, ".Environment")=<environment: R_GlobalEnv> 
#                                                                                 .. .. ..- attr(*, "predvars")= language list(tip, day)
#                                                                                 .. .. ..- attr(*, "dataClasses")= Named chr [1:2] "numeric" "factor"
#                                                                                 .. .. .. ..- attr(*, "names")= chr [1:2] "tip" "day"
#                                                                                 - attr(*, "class")= chr [1:2] "aov" "lm"
# ###############################################################################
DO_SIMPLE_ANOVA = function( formula_txt, data ) {
    formula_f = formula(eval(parse(text=formula_txt)))
    t = aov( tip ~ day -1, data=data )
    summary(t)
    t$coeff
    return( t )
}
# ###############################################################################


# ###############################################################################
COMPARE_EFFECTS_OF_FACTOR_LEVELS_FOR = function( x, y ) {
}
# ###############################################################################


# ######################################################################################################
DO_REGRESSION_DIAGNOSTICS = function( X_TEST_SCALED, Y_TEST_SCALED, SGD_Y_TEST, PREDICT_VARNAME="Y", W_INTERCEPT=TRUE, debug=TRUE ) {
    cat(HEADER)
    op <- par( mfrow=c(2,3) )
        eterm = Y_TEST_SCALED-SGD_Y_TEST
        ci    = sd(eterm)
        plot( Y_TEST_SCALED,  SGD_Y_TEST, pch="+", cex=0.7, main="PREDICTED vs. TRUE VALUES" )
            text( SGD_Y_TEST     +rnorm(1,0,1E-3), 
                  Y_TEST_SCALED  +rnorm(1,0,1E-3), rownames(SGD_Y_TEST ), c="black", cex=0.5 )
            abline(c(0,1),          col="black",lw=2)
            abline(c(mean(eterm),1),col="blue", lw=1)
            abline(c(ci, 1),        col="red",  lw=2)
            abline(c(-ci,1),        col="red",  lw=2)

        DO_HIST(   Y_TEST_SCALED, nbins=32, ptitle="Histogram of Y" )
        DO_HIST(   SGD_Y_TEST,    nbins=32, ptitle="Histogram of YP" )
        DO_QQPLOT( Y_TEST_SCALED-SGD_Y_TEST, nbins=32, use_par=FALSE )
    par( op )

    # ######################################################################################################
    print( "ARE THESE REPRESENTATIVE OF SIMILAR/SAME SOURCES?" )
    FINDINGS = ARE_THESE_REPRESENTATIVE_OF_SAME_SOURCE( Y_TEST_SCALED, 
                                                        SGD_Y_TEST,    nmax=0, confidence_level=0.999 )
    cat( HEADER)
    # ######################################################################################################

    # ######################################################################################################
    # fixme: put eval parse with theta and the columsn of x
    # ######################################################################################################
    XY_TEST = EXTEND_DF( as.data.frame(X_TEST_SCALED), as.data.frame(Y_TEST_SCALED), colname="Y")

    if ( !W_INTERCEPT )
        FIT = glm( Y ~ . - 1, data = XY_TEST )
    else
        FIT = glm( Y ~ .,     data = XY_TEST )

    sum_txt = summary( EXTEND_DF( as.data.frame(XY_TEST), as.data.frame(SGD_Y_TEST), colname="YP") )
    fit_txt = summary( FIT )

    NEWLINE(3) ; cat(HEADER)
    print ( sum_txt )
    cat(HEADER)
    NEWLINE(3) ; cat(HEADER)
    print ( fit_txt )
    cat(HEADER)

    retvals = list( 'FIT'=FIT, 'FINDINGS'=FINDINGS )
    return ( retvals )
}
# ######################################################################################################



