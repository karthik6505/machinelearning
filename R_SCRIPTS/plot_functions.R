# ########################################################################################
# AUTHOR: NELSON R. MANOHAR
# DATE:   SEP/2014
# LANGUAGE: R
# INTENDED-USE: CODE-SAMPLE, LIBRARY BUILDING
# CODEBASE: MACHINE LEARNING
# FILENAME: 
# ########################################################################################


source('utilities.R')


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
# http://stats.stackexchange.com/questions/52293/r-qqplot-how-to-see-whether-data-are-normally-distributed
# ###############################################################################
DO_QQPLOT = function( x, nbins=32, use_par=TRUE, ptitle="Residuals" ) {
    require(e1071)
    if ( use_par ) op <- par(mfrow = c(1, 3))
        # plot1: # qq-plot: you should observe a good fit of the straight line
        qqnorm(x, xlab=ptitle)
        qqline(x)
     
        # plot2: # p-plot: you should observe a good fit of the straight line
        probplot(x, qdist=qnorm, xlab=ptitle)
    
        # plot3: # fitted normal density
        DO_HIST( x, nbins=nbins, ptitle="Histogram of Residuals" )
    if ( use_par) par(op)
}
# ###############################################################################


# ###############################################################################
DO_HIST = function( x, nbins=32, ptitle="Histogram of Residuals" ) {
    hist(x, freq=FALSE, breaks=nbins, main=ptitle)
    f.den <- function(t) dnorm(t, mean=mean(x), sd=sqrt(var(x)))
    curve(f.den, add=TRUE, col="darkblue", lwd=2)
}
# ###############################################################################


# #####################################################################################
# #####################################################################################
DO_BASIC_FEATURE_ANALYSIS_PLOT = function( xdf, colnum, numbins=32 ) {
    xxx   = xdf[,colnum]
    title = GET_DF_COLNAME( xdf, colnum)
    PLOT_FEATURE( xxx, xbins=numbins, ptitle=title )
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
# #####################################################################################
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
# #####################################################################################
VIOLIN_DENSITY_PLOT = function( XYdf, i )  {
    Y = XYdf[,ncol(XYdf)]
    p1 = ggplot( as.data.frame(XYdf), aes( y=Y, x=RECODE(XYdf[,i],xnbins=2)))  + geom_violin() + theme_wsj(base_size=8)
    p2 = ggplot( as.data.frame(XYdf), aes( y=Y, x=RECODE(XYdf[,i],xnbins=3)))  + geom_violin() + theme_wsj(base_size=8)
    p3 = ggplot( as.data.frame(XYdf), aes( y=Y, x=RECODE(XYdf[,i],xnbins=4)))  + geom_violin() + theme_wsj(base_size=8)
    p4 = ggplot( as.data.frame(XYdf), aes( y=Y, x=RECODE(XYdf[,i],xnbins=8)))  + geom_violin() + theme_wsj(base_size=8)
    MULTIPLOT(p1, p2, p3, p4, cols=2)
    return ( list(p1, p2, p3, p4 ) )
}
# #####################################################################################


# #####################################################################################
# association plot automatically selecting the smallest size factors for quick inspection 
# assoc( Y ~ V8+V5, data=XYdf, shade=TRUE)
# #####################################################################################
ASSOC_PLOT = function( XY, Y, nb=4, nmax=100, plot_output="" ) {
    if ( plot_output != "" ) pdf(plot_output)
    # SAMPLED_ROWS2 = RANDOM_ROWS( rownames(XYr), m=min(100,length(rownames(XYr))))
    # XYdf = EXTEND_DF( XY, RECODE(Y,nb) )[SAMPLED_ROWS2,]
    XYdf = DO_STRATIFIED_SUBSAMPLING( XY, Y, nb, nmax )
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
# #####################################################################################


# #####################################################################################
DO_PERFORMANCE_CURVE_PLOT = function( x, y, xlab, ylab, ... ) {
    plot( x=x, y=y, type='b', pch=22, bg='brown', xlab=xlab, ylab=ylab, ... )
    title( "LEARNING CURVE (for_given_clf)" )

    grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")

    y_delta = (max(y)-min(y))/length(y)
    for( i in 1:length(y) ) {
        xx=x[i]
        yy=y[i] + y_delta 
        text( xx, yy, as.character( round(yy,2) ), col="blue", cex=0.5 ) 
    }

}
# #####################################################################################


