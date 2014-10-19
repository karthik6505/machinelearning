# README #

This README would normally document whatever steps are necessary to get your application up and running.



### MOST IMPORTANT DIRECTORIES ###
    1. Main directory is R_SCRIPTS.
    2. Main directory is EMSEMBLE for PYTHON CLASSIFIER ANALYTICS
    3. Main directory is LINEAR_REGRESSION for PYTHON REGRESSION ANALYTICS



### R_ANALYTICS (R_SCRIPTS DIRECTORY): ###
    R analytics contain machine learning and predictive analytics in R
    for various general areas. Of course, these have not yet been optimized
    and thus are limited by your environment R's runtime limitations.
    Some of these scripts are useful for educational purposes, so as to
    illustrate, detailed inner-working graphics as how some of these
    algorithms decision making and progress occurs over time. On most,
    detailed error metrics are provided to facilitate automated 
    decision-making.

#### 28 -rw-rw-r-- 1 nrm nrm 25772 Oct 18 18:09 an_agglomerative_clusterer.R ####
    An agglomerative clustering (for demonstrative purposes only, please
    see disclaimers). Perhaphs useful for those in bioinformatics, philogeny, etc
    seeking to understand the evolution of the clustering arrangements
    and those whose interest is primarely on the interdependencies and 
    arrangement of the branches as it automatically focuses on those (i.e.,
    less overplotting as leaves are not plotted). Nevertheless, detailed
    cluster assignment mappings are displayed during the agglomerative process.

#### 16 -rw-rw-r-- 1 nrm nrm 14433 Oct 18 06:57 decision_trees.R ####
    Under development. Basic decision trees.

#### 28 -rw-rw-r-- 1 nrm nrm 25898 Oct 17 07:37 regression.R ####
    Basic multivariate linear regression with assumptions checking.

#### 24 -rw-rw-r-- 1 nrm nrm 21059 Oct 17 03:57 marginals.R ####
    Basic recursive partitioning decision trees. Under development.

####  4 -rw-rw-r-- 1 nrm nrm  1661 Oct 15 01:39 model_comparison.R ####
    To be developed.

#### 24 -rw-rw-r-- 1 nrm nrm 13787 Oct  4 22:54 dbscan.R    ####
    Implements a version of dbscan which instead uses transitive closure which 
    allows to do heuristics about the search space and adaptations to it.
    For example, it allows to discover the value of EPSILON that fits
    the data. Experimental as all in here as written in past 24 hrs.

#### 36 -rw-rw-r-- 1 nrm nrm 26314 Oct  4 19:58 clustering_methods.R ####
    Implements a diagnostics version of kmeans which provides feedback about
    the convergence of kmeans within iterations given a value of K.
    Experimental as all in here as written in past 24 hrs.

#### 16 -rw-rw-r-- 1 nrm nrm  4610 Oct  3 02:56 recommender_igraphs.R ####
    Implements plotting of adjacency matrices produced by the recommender system 
    using igraph. Currently based on adjacency non-sparse matrix representation 
    which does NOT scale well with dataset size for this application as 

#### 24 -rw-rw-r-- 1 nrm nrm 13757 Oct  3 02:33 recommender_diagnostics.R ####
    Implements basic diagnostics plots for the recommender system.

#### 48K -rw-rw-r-- 1 nrm nrm  38K Sep 25 20:20 recommender_systems.R ####
    Implements an iterative convergence collaborative filtering and 
    recommendation system, tailored for the movielens dataset. 
    1. Collaborative filtering is done via iterative convergence between 
       Theta parameters and X-feature parameters. 
    2. Recomendations are done using euclidean (at this time) distances 
       wrt shortest-path neighbors at one and two degree of separations.

#### 56K -rw-rw-r-- 1 nrm nrm  46K Sep 24 18:23 stochastic_gradient_descent.R ####
    Performs gradient descent, stochastic gradient descent, fminunc, and
    normal equations with or without regularization over numerical datasets.

#### 24K -rw-rw-r-- 1 nrm nrm  14K Sep 25 19:56 distances.R ####
    Implements by wraping distance computations after various 
    transformations: pca, probability, and scaling transforms for
    numerical and/or categorical datasets.

#### 36K -rw-rw-r-- 1 nrm nrm  26K Sep 23 21:03 anomaly_detection.R ####
    Implements anomaly detection over a numerical dataset wrt to
    1. Gaussian univariate (independent features) 
    2. Gaussian multivariate (otherwise)

#### 24K -rw-rw-r-- 1 nrm nrm  16K Sep 19 21:16 t_tests.R ####
    Implements simpler/selected t_tests statistical tests procedures with 
        1. iterative or not wrappers 
        2. over full or subsampled datasets.

#### 20K -rw-rw-r-- 1 nrm nrm 9.7K Sep 26 13:25 basket_rules.R ####
    Performs heuristic optimization via grid search for Market Basket Analysis 
    to identify the highest confidence/support RHS for the specified LHS.

#### 36K -rw-rw-r-- 1 nrm nrm  26K Sep 25 17:53 datasets.R ####
    Generates and load datasets into expected format for the analytics.

#### 36K -rw-rw-r-- 1 nrm nrm  27K Sep 24 18:24 fselect.R ####
    Wraps up some selected fSelect.R feature selection algorithms for
    numerical and categorical datasets on classification and/or regression 
    problems

#### 32K -rw-rw-r-- 1 nrm nrm  23K Sep 24 16:23 utilities.R ####
    Wraps ups various common utilities used by various of these modules.

#### 12K -rw-rw-r-- 1 nrm nrm  497 Sep 11 20:30 classifiers.R ####
    being developed.

#### 24K -rw-rw-r-- 1 nrm nrm  14K Sep 23 01:14 regression.R ####
    being developed.

#### 16K -rw-rw-r-- 1 nrm nrm 4.6K Sep 23 21:03 aggregate.R ####
    not yet developed. will be a database wrapper for analyzing
    datasets with or without database aid.

#### 20K -rw-rw-r-- 1 nrm nrm 8.3K Sep 22 23:27 plot_functions.R ####
    wraps ups visualization scripts, some reusing and/or adapting
    plotting code available on the web, all with the url-ref/citations
    to the original site.

#### 16K -rw-rw-r-- 1 nrm nrm 7.2K Sep 19 23:59 learning_curves.R ####
    not yet developed. Instead, for learning curves, see 
    stochastic_gradient_descent.R

#### 20K -rw-rw-r-- 1 nrm nrm  11K Sep 19 21:16 data.R ####
    deprecated.

#### 12K -rw-rw-r-- 1 nrm nrm 1.7K Sep 19 21:14 copyrigth.R ####
    GNU license

#### 12K -rw-rw-r-- 1 nrm nrm 2.4K Sep 18 23:15 exception.R ####
    not yet developed. provides wrapper to exception processing



### What is this repository for? ###

The goal is to provide access to some quickly developed code-samples I put over a few days so as to facilitate discussion.

* Quick summary *
    TO CASUAL VISITORS:
    1. Please do not branch YET from this codebase as the code is CURRENTLY way 
       too preliminary; it's is just a matter of a few days old (Sep/26/2014);
       i.e., version 0.00b. 

    2. However, you are welcome to BROWSE at this time the codebase.
       If you find it or a part useful and decide to recycle it, please 
       follow accordance to the provided GNU license along with an URL
       reference to the original [codebase] (https://bitbucket.org/nelsonmanohar/machinelearning)

* Version: *
    Again, just to be clear: version 0.001b.

* [Learn Markdown] *
    (https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact



