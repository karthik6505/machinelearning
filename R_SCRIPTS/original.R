
# ########################################################################################
# AUTHOR: NELSON R. MANOHAR
# DATE:   SEP/2014
# LANGUAGE: R
# INTENDED-USE: CODE-SAMPLE, LIBRARY BUILDING
# CODEBASE: MACHINE LEARNING
# FILENAME: 
# ########################################################################################


# #####################################################################################
# load libraries
# loading csv
# clean csv: complete cases
# preprocessing csv: scaling (numerical features)
# preprocessing csv: binarize categorial data
# optional: stratified subsampling 
# optional: build feature subsets
# iterator: correlated features within feature subset
# describe features from csv 
# feature selection
# plot
# #####################################################################################


# #####################################################################################
COLNAME = function( dfname, i ) {
    df_colnames = colnames( eval(parse(text=dfname)) )
    col_name = paste( dfname, "$", df_colnames[i], sep="")
    return ( col_name )
}
# #####################################################################################
COLDATA = function( dfname, i ) {
    col_name = COLNAME ( dfname, i )
    col_data = eval(parse(text=col_name))
    return ( col_data )
}
# #####################################################################################
COLTYPE = function( col_data ) {
    col_class= class(col_data)
    return ( col_class )
}
# #####################################################################################
GET_XY_COLUMN_CLASS = function( colnamestr, df_name="" ) {
    col_objname_class = class( eval(parse(text=paste(df_name,"$", colnamestr, sep=""))))
    return ( col_objname_class )
}
# COLS_DATATYPES = mapply( GET_XY_COLUMN_CLASS, colnames(XYnew), df_name="XYnew" )
# #####################################################################################


# #####################################################################################
# http://stat.ethz.ch/R-manual/R-devel/library/graphics/html/hist.html
# #####################################################################################
PLOT_FEATURE = function( xx, xbins=16, ptitle="" ) {
    op <- par(mfrow = c(1, 4))
        # plot1
        utils::str(hist(xx,        col = "gray",     labels = TRUE, main=paste("histogram", ptitle)))

        # plot2
        r = hist(sqrt(abs(xx)),  col = "lightblue",labels = TRUE, main=paste("histogram", "sqrt", ptitle), border = "pink")
        text(r$mids, r$density, r$counts, adj = c(.5, -.5), col = "blue3")
        sapply(r[2:3], sum)
        sum(r$density * diff(r$breaks)) # == 1
        lines(r, lty = 3, border = "purple") # -> lines.histogram(*)

        # plot3: ## Comparing data with a model distribution should be done with qqplot()!
        set.seed(14)
        # x <- rchisq(100, df = 4)
        x <- rnorm(100)
        # CHECK CHISQ PLOT AND DF
        # qqplot(x, qchisq(ppoints(xx), df = 4)); abline(0, 1, col = 2, lty = 2)
        qqplot(x, qnorm(ppoints(xx))); abline(0, 1, col = 2, lty = 2)

        # plot4: ## if you really insist on using hist() ... :
        hist(xx, freq = FALSE, ylim = c(0, 0.42), main=paste( "histogram", ptitle ))
        # curve(dchisq(x, df = 4), col = 2, lty = 2, lwd = 2, add = TRUE)
        curve(dnorm(x, mean=mean(xx), sd=sd(xx)), add=TRUE, col="darkblue", lwd=2) 
    par(op)
}
# #####################################################################################


# #####################################################################################
GET_DF_COLNAME = function( df, i ) {
    this_colname = attr(df,"dimnames")[[2]][i]
    return ( this_colname )
}
# #####################################################################################


# #####################################################################################
DO_BASIC_FEATURE_ANALYSIS_PLOT = function( xdf, colnum, numbins=32 ) {
    xxx   = xdf[,colnum]
    title = GET_DF_COLNAME( xdf, colnum)
    PLOT_FEATURE( xxx, xbins=numbins, ptitle=title )
}
# #####################################################################################


# #####################################################################################
RANDOM_ROWS = function( idx, m=100 ) {
    random_rows = sort(sample( idx, m ))
    return ( random_rows )
}
# #####################################################################################


# #####################################################################################
EXTEND_DF = function( X, Y ) { 
    XX = data.frame( X, Y )
    colnames(XX) = c(colnames(X), "Y" )
    rownames(XX) = rownames(X)
    return ( XX )
}
# #####################################################################################


# #####################################################################################
# http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_(ggplot2)/
# #####################################################################################
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
MULTIPLOT <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  require(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

  if (numPlots==1) {
    print(plots[[1]])
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
# #####################################################################################


# #####################################################################################
RECODE = function( x, xnbins=10, debug=FALSE ) {
    q = seq(0,1,1/xnbins)
    p = quantile( x, q )
    pbins = attr(p, "names")
    for ( i in 1:xnbins ) {
        ql = p[pbins[i]]
        qh = p[pbins[i+1]]
        qlset = length(x[ x>=ql & x<qh ])
        if ( qlset ) {
            x[ x>=ql & x<qh ] = ql
        }
        if ( debug ) {
            print ( paste ( ql, qh ) )
            print ( x[ x>=ql & x<qh ] )
        }
    }
    x = factor(x, levels=p, labels=pbins)
    return( x )
}
# #####################################################################################





























# #####################################################################################
# http://cran.r-project.org/doc/contrib/Lemon-kickstart/kr_scrpt.html
# #####################################################################################
INITIAL.DIR <- getwd()          # store the current directory
# setwd("/home/nrm/WORKSPACE/R")# change to the new directory
# detach("package:nlme")        # unload the libraries
# sink("R.out")                 # set the output file
# sink()                        # close the output file
# detach("package:nlme")        # unload the libraries
# setwd(INITIAL.DIR)            # change back to the original directory
# #####################################################################################


# #####################################################################################
library(utils)
library(stats)
library(ggplot2)
library(ggthemes)
library(vcd)
# #####################################################################################


# #####################################################################################
DEBUG = TRUE
HEADER = paste( "==================================================" )
SUBHEADER = paste( "    ----------------------------------------------" )
# #####################################################################################


# #####################################################################################
# LOAD CSV
# #####################################################################################
DATAFILE = "mldata/autos/imports-85.data"
XY = read.csv ( DATAFILE, header=FALSE, na.strings="?", stringsAsFactors=TRUE, sep="," )
# #####################################################################################


# #####################################################################################
# M x N (M samples by N features)
# #####################################################################################
N = ncol(XY) 
M = nrow(XY)
P = M # as.integer(M/10)
# #####################################################################################


# #####################################################################################
# SELECT COMPLETE CASES, WITHOUT MISSING VALUES
# #####################################################################################
COMPLETE_SAMPLES = complete.cases(XY)[0:P]
XYnew = XY[COMPLETE_SAMPLES,]
print( summary(XYnew) )
print( HEADER )
# #####################################################################################


# #####################################################################################
COLNAMES = colnames(XYnew)
COLS_DATATYPES = mapply( class, XYnew )
COLS_FACTORS   = grep( 'factor',  COLS_DATATYPES )
COLS_INTEGERS  = grep( 'integer', COLS_DATATYPES )
COLS_NUMERICS  = grep( 'numeric', COLS_DATATYPES )
COLS_UNKNOWNS  = c(1:N, -COLS_FACTORS, -COLS_INTEGERS, -COLS_NUMERICS )
# #####################################################################################
if ( DEBUG ) {
    print ( HEADER )
    print ( paste( "DATASET:COLUMN TYPE", COLNAMES, COLS_DATATYPES) )
    print ( SUBHEADER )
    print ( paste( "CATEGORICAL COLUMNS", COLS_FACTORS ) )
    print ( SUBHEADER )
    print ( paste( "NUMERICAL COLUMNS",   COLS_NUMERICS ) )
    print ( SUBHEADER )
    print ( paste( "INTEGER COLUMNS",     COLS_INTEGERS ) )
    if ( !is.null ( COLS_UNKNOWNS ) ) {
        print ( SUBHEADER )
        print ( paste( "MISCELLANEOUS COLUMNS", COLS_UNKNOWNS ) )
    }
    print ( HEADER )
}
# #####################################################################################


# #####################################################################################
PREDICT = 25
Y   = XYnew[,PREDICT]
XYr = XYnew[,setdiff(COLS_NUMERICS,PREDICT)]
XYf = XYnew[,setdiff(COLS_FACTORS, PREDICT)]
XYi = XYnew[,setdiff(COLS_INTEGERS,PREDICT)]
# #####################################################################################


# #####################################################################################
XYrnew = scale( XYr, center=TRUE, scale=TRUE )
XYrnew_centers = attr(XYrnew, "scaled:center" )
XYrnew_scales  = attr(XYrnew, "scaled:scale" )
colnames(XYrnew) = colnames(XYr)
rownames(XYrnew) = rownames(XYr)
utils::str(XYrnew)
# #####################################################################################


# #####################################################################################
XYinew = scale( XYi, center=TRUE, scale=TRUE )
XYinew_centers = attr(XYinew, "scaled:center" )
XYinew_scales  = attr(XYinew, "scaled:scale" )
colnames(XYinew) = colnames(XYi)
rownames(XYinew) = rownames(XYi)
utils::str(XYinew)
# #####################################################################################



# #####################################################################################
for (i in 1:ncol(XYinew)) {
    DO_BASIC_FEATURE_ANALYSIS_PLOT( XYinew, i, numbins=32 )
}
# #####################################################################################


# #####################################################################################
# example
# #####################################################################################
XXi = mapply( RECODE, XYi, xnbins=4)
XXi = data.frame( matrix( XXi, nrow=nrow(XYi)) )
colnames(XXi) = colnames(XYi)
rownames(XXi) = rownames(XYi)
# #####################################################################################


# #####################################################################################
# for (i in 1:ncol(XYinew)) { ggplot( as.data.frame(XYinew), aes( y=Y, x=as.factor(RECODE(XYinew[,i],xnbins=4) ))) + geom_violin() }
# #####################################################################################


# #####################################################################################
# #####################################################################################
SAMPLED_ROWS = RANDOM_ROWS( rownames(XYr), m=min(30,length(rownames(XYr))))
# #####################################################################################


# #####################################################################################
# #####################################################################################
if ( 0 ) {
    library('GGally')
    pdf("ggpairs-numeric.pdf", onefile=TRUE, 10, 7)
        ggpairs(EXTEND_DF(XYr, Y)[SAMPLED_ROWS,], params=list(corSize=7, base_size=7))
    dev.off()
    pdf("ggpairs-integer.pdf", 10, 7)
        ggpairs(EXTEND_DF(XYi, Y)[SAMPLED_ROWS,], params=list(corSize=7, base_size=7))
    dev.off()
    pdf("ggpairs-integer-asfactors.pdf", 10, 7)
        ggpairs(EXTEND_DF(XXi, Y)[SAMPLED_ROWS,], params=list(corSize=7, base_size=7))
    dev.off()
    pdf("ggpairs-factors.pdf", 10, 7)
        ggpairs(EXTEND_DF(XYf, Y)[SAMPLED_ROWS,], params=list(corSize=7, base_size=7))
    dev.off()
    graphics.off()
}
# #####################################################################################


# #####################################################################################
VIOLIN_DENSITY_PLOT = function( XYdf, i )  {
    Y = XYdf[,ncol(XYdf)]
    p1 = ggplot( as.data.frame(XYdf), aes( y=Y, x=RECODE(XYdf[,i],xnbins=2)))  + geom_violin() + theme_wsj(base_size=8)
    p2 = ggplot( as.data.frame(XYdf), aes( y=Y, x=RECODE(XYdf[,i],xnbins=3)))  + geom_violin() + theme_wsj(base_size=8)
    p3 = ggplot( as.data.frame(XYdf), aes( y=Y, x=RECODE(XYdf[,i],xnbins=4)))  + geom_violin() + theme_wsj(base_size=8)
    p4 = ggplot( as.data.frame(XYdf), aes( y=Y, x=RECODE(XYdf[,i],xnbins=8))) + geom_violin() + theme_wsj(base_size=8)
    MULTIPLOT(p1, p2, p3, p4, cols=2)
    return ( list(p1, p2, p3, p4 ) )
}
# #####################################################################################


# #####################################################################################
# Because of scoping bug in ggplot, it is necessary that the data frame be at the global 
# scoping level, for this reason it is critical that the name matches to the local variable
# name; hence a new data frame is created here to make it work
# #####################################################################################
XYdf = DO_STRATIFIED_SUBSAMPLING( XYinew, Y, 8, nmax=30 )
# XYdf = EXTEND_DF( XYinew, Y )[SAMPLED_ROWS,]
pdf("feature_exploration_discrete_features.pdf")
    for (i in 1:ncol(XYdf)) {
        plist = VIOLIN_DENSITY_PLOT( XYdf, i )
    }
dev.off()
graphics.off()
# #####################################################################################


# #####################################################################################
XYdf = DO_STRATIFIED_SUBSAMPLING( XYrnew, Y, 8, nmax=30 )
# XYdf = EXTEND_DF( XYrnew, Y )[SAMPLED_ROWS,]
pdf("feature_exploration_continous_features.pdf")
    for (i in 1:ncol(XYdf)) {
        plist = VIOLIN_DENSITY_PLOT( XYdf, i )
    }
dev.off()
graphics.off()
# #####################################################################################


# #####################################################################################
# stratified sampling function needed
# #####################################################################################
GET_SAMPLES_WITH_YLEVEL = function(  XYdf, y_level, nmax=100 ) {
    idx = XYdf$Y == y_level
    print ( y_level )
    y_level_rows = XYdf[ idx, ]
    yidx = rownames( y_level_rows )
    y_level_idx = RANDOM_ROWS( yidx, m=min(nmax,length(yidx)))
    SUBSAMPLED_Y = XYdf[ y_level_idx, ]
    return ( SUBSAMPLED_Y )
}
# #####################################################################################


# #####################################################################################
STRATIFIED_SUBSAMPLING = function( X, Y, nb=4, nmax=100 ) {
    Y_as_factor = RECODE(Y,nb)
    XYdf = EXTEND_DF( X, Y_as_factor )
    y_levels = levels( Y_as_factor )
    START = TRUE
    for ( y_level in y_levels ) {
        idx = XYdf$Y == y_level
        print ( y_level )
        y_level_rows = XYdf[ idx, ]
        yidx = rownames( y_level_rows )
        y_level_idx = RANDOM_ROWS( yidx, m=min(nmax,length(yidx)))
        if ( START == TRUE ) { 
            SUBSAMPLED_Y_LEVEL_ROWS = XYdf[ y_level_idx, ]
            START = FALSE
        } else {
            subsampled_y_level_rows = XYdf[ y_level_idx, ]
            SUBSAMPLED_Y_LEVEL_ROWS = rbind( SUBSAMPLED_Y_LEVEL_ROWS, subsampled_y_level_rows ) 
        }
        txt = summary( SUBSAMPLED_Y_LEVEL_ROWS )
        print ( txt )
    }
    return ( SUBSAMPLED_Y_LEVEL_ROWS  )
}
# #####################################################################################


# #####################################################################################
DO_STRATIFIED_SUBSAMPLING = function( X, Y, nb=4, nmax=100 ) {
    Y_as_factor = RECODE(Y,nb)
    XYdf = EXTEND_DF( X, Y_as_factor )
    y_levels = levels( Y_as_factor )
    START = TRUE
    for ( y_level in y_levels ) {
        if ( START == TRUE ) { 
            SUBSAMPLED_Y_LEVEL_ROWS = GET_SAMPLES_WITH_YLEVEL(  XYdf, y_level, nmax )
            START = FALSE
        } else {
            subsampled_y_level_rows = GET_SAMPLES_WITH_YLEVEL(  XYdf, y_level, nmax )
            SUBSAMPLED_Y_LEVEL_ROWS = rbind( SUBSAMPLED_Y_LEVEL_ROWS, subsampled_y_level_rows ) 
        }
        txt = summary( SUBSAMPLED_Y_LEVEL_ROWS )
        print ( txt )
    }
    return ( SUBSAMPLED_Y_LEVEL_ROWS  )
}
# #####################################################################################


# #####################################################################################
# association plot automatically selecting the smallest size factors for quick inspection 
# assoc( Y ~ V8+V5, data=XYdf, shade=TRUE)
# #####################################################################################
ASSOC_PLOT = function( XY, Y, nb=4, plot_output="" ) {
    if ( plot_output != "" ) pdf(plot_output)
    # SAMPLED_ROWS2 = RANDOM_ROWS( rownames(XYr), m=min(100,length(rownames(XYr))))
    # XYdf = EXTEND_DF( XY, RECODE(Y,nb) )[SAMPLED_ROWS2,]
    XYdf = DO_STRATIFIED_SUBSAMPLING( XY, Y, nb, nmax=100 )
    txt = summary( XYdf )
    print( txt )
    XYnlevels = mapply( nlevels, XY )
    xn = sort(XYnlevels)
    print( xn )
    for (i in seq(1,length(xn)-3,1)) {
        for (j in seq(i+1,length(xn)-4)) {
        idx  = c(i, j)
        idx_names = names(xn[ idx ])
        if (sum(is.na(idx_names)) > 0) break
        idx_names = c( idx_names, "Y" )
        XYdf1 = XYdf[ idx_names ]
        print( names(XYdf1) )
        assoc( XYdf1, data=XYdf1, shade=TRUE)
        }
    }
    if ( plot_output != "" ) {
        dev.off()
        graphics.off()
    }
}
ASSOC_PLOT( XYf, Y, 4, plot_output="feature_exploration_factor_features.pdf" )

# #####################################################################################


