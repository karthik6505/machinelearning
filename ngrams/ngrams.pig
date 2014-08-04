-- SKELETON for ngram parser tutorial
-- NRM

-- --------------------------------------------------------------------------------------------------
-- RELATION n5grams_dataset as per description of google
-- n5gram                                                                YYYY    H   P   B
-- angler skill the common mudskipp                                 	 2011 	 1 	 6 	 1
-- soon was the debt repaid                                         	 1972 	 1 	 4 	 4
-- and ranch without depend on                                      	 1971 	 1 	 3 	 1
-- weight of 238 wherea lead                                        	 1995 	 1 	 9 	 3
-- in new zealand is veri                                           	 1970 	 1 	 4 	 2
-- --------------------------------------------------------------------------------------------------


-- --------------------------------------------------------------------------------------------------
-- datasets
-- --------------------------------------------------------------------------------------------------
-- NG = LOAD 'hdfs:///user/hduser/ngrams/n5gram_dataset.txt' 
-- NG = LOAD 'ngrams/googlebooks-eng-us-all-5gram-20120701-re' 
-- NG = LOAD 'ngrams/google-re.txt'
-- USING PigStorage('\t') AS ( ngram:chararray, year:int, pages:int, books:int ); 
-- --------------------------------------------------------------------------------------------------



-- --------------------------------------------------------------------------------------------------
-- load the dataset
-- --------------------------------------------------------------------------------------------------
NG = LOAD 'ngrams/google-re.txt' USING PigStorage('\t') AS ( ngram:chararray, year:int, pages:int, books:int ); 


-- --------------------------------------------------------------------------------------------------
-- apply simple/typical volume reductions to the data, for example, filter all 5grams with less than
-- 5 hits per year to make the dataset manageable without too much loss at the heavy tail
-- --------------------------------------------------------------------------------------------------
NG = DISTINCT NG;
NG = FILTER NG BY pages > 20; -- NOTE: here, pages is directly accessible
NG = FOREACH NG GENERATE ngram, year, pages, books, pages/(books+1E-06) as density;
ILLUSTRATE NG; 
DESCRIBE   NG;


-- --------------------------------------------------------------------------------------------------
-- generate potential predictive features by computing values for each ngram as an object
--      for example, given a time series across previous years of these values, could you predict whether
--      the ngram will be hot or not, next year? in this case, this equates to two classes, top_ngrams
--      versus others, and the predictive features composed and shared by these. ideally, these are
--      composed at this stage.
-- --------------------------------------------------------------------------------------------------
-- STEP 1: group by ngrams
-- --------------------------------------------------------------------------------------------------
NG_BY_NGRAM = GROUP NG BY ngram;
DESCRIBE   NG_BY_NGRAM;
ILLUSTRATE NG_BY_NGRAM;


-- --------------------------------------------------------------------------------------------------
-- STEP 2: generate potential predictive features by computing values for ngram year
-- --------------------------------------------------------------------------------------------------
NGRAM_FREQS = FOREACH NG_BY_NGRAM GENERATE FLATTEN(group), 
                            SUM(NG.pages) AS psum, SUM(NG.books) AS bsum, AVG(NG.density) as davg,
                            SUM(NG.pages)/(SUM(NG.books)+1E-6) AS p2bden,
                            SUM(NG.pages)/SUM(NG.books)/(COUNT(NG.year)+1E-6) AS p2byden,
                            COUNT(NG.year) AS ycnt, MAX(NG.year)- MIN(NG.year) AS ydif;
DESCRIBE   NGRAM_FREQS;
ILLUSTRATE NGRAM_FREQS;




-- --------------------------------------------------------------------------------------------------
-- A COMPUTATION TO DETERMINE THE TOP X=10 NGRAMS PER YEAR (A CLASS TO BE PREDICTED, FOR EXAMPLE)
-- --------------------------------------------------------------------------------------------------
-- STEP1: group by year and ngram
-- --------------------------------------------------------------------------------------------------
NG_BY_YEAR_NGRAM = GROUP NG BY ( year, ngram );
DESCRIBE   NG_BY_YEAR_NGRAM;
ILLUSTRATE NG_BY_YEAR_NGRAM;


-- --------------------------------------------------------------------------------------------------
-- STEP 2: generate the ranking metric for each group and flatten
-- --------------------------------------------------------------------------------------------------
NG_BY_YEAR_NGRAM_RANKS = FOREACH NG_BY_YEAR_NGRAM GENERATE group.$0 as year, group.$1 as ngram, 
                            SUM(NG.pages) as psum,
                            AVG(NG.density) as davg, 
                            SUM(NG.pages)/(SUM(NG.books)+1E-6) as ametric;
DESCRIBE   NG_BY_YEAR_NGRAM_RANKS;
ILLUSTRATE NG_BY_YEAR_NGRAM_RANKS;


-- --------------------------------------------------------------------------------------------------
-- STEP 3: just group again by year 
-- --------------------------------------------------------------------------------------------------
NG_BY_YEAR_RANKS = GROUP NG_BY_YEAR_NGRAM_RANKS BY year;
DESCRIBE   NG_BY_YEAR_RANKS;
ILLUSTRATE NG_BY_YEAR_RANKS;


-- --------------------------------------------------------------------------------------------------
-- STEP 4: for each year, sort based on the cost metric, the associated bag tuples and retrieve the top ones
-- --------------------------------------------------------------------------------------------------
TOP_NGRAMS = FOREACH NG_BY_YEAR_RANKS {
    RESULTS = ORDER NG_BY_YEAR_NGRAM_RANKS BY davg, psum desc;
    TOP_RESULTS = LIMIT RESULTS 10;
    GENERATE FLATTEN( TOP_RESULTS );
}
DESCRIBE   TOP_NGRAMS;
ILLUSTRATE TOP_NGRAMS;


-- --------------------------------------------------------------------------------------------------
-- OPTIONAL STEP 5: find among above, which are reocurring across years and on which years but reduce 
-- output by applying some data reductions of interest (e.g., 5 years on top and ordered by dillution
-- which years what range can be itemized here
-- --------------------------------------------------------------------------------------------------
TOPNGRAM_YEARS = GROUP TOP_NGRAMS BY TOP_RESULTS::ngram;
TOPNGRAM_YEAR_COUNTS = FOREACH TOPNGRAM_YEARS GENERATE group as ngram, 
                        MAX( TOP_NGRAMS.TOP_RESULTS::year ) - MIN( TOP_NGRAMS.TOP_RESULTS::year ) as year_range , 
                        COUNT( TOP_NGRAMS.TOP_RESULTS::year ) as total_years;
TOPNGRAM_REPEATS = FILTER TOPNGRAM_YEAR_COUNTS BY total_years > 5;
TOPNGRAM_REPEATS_METRICS = FOREACH TOPNGRAM_REPEATS GENERATE ngram, 
                        total_years, year_range, 
                        total_years / (year_range + 1E-6) as year_dilution;
TOPNGRAM_REPEATS_SORTED = ORDER TOPNGRAM_REPEATS_METRICS BY year_dilution desc;
DESCRIBE    TOPNGRAM_REPEATS_SORTED;
ILLUSTRATE  TOPNGRAM_REPEATS_SORTED;
-- --------------------------------------------------------------------------------------------------





-- --------------------------------------------------------------------------------------------------
-- FOR EACH NGRAM FIND THE TOP YEAR AT WHICH IT PEAKED
-- --------------------------------------------------------------------------------------------------
-- STEP1: group by year and ngram
-- --------------------------------------------------------------------------------------------------
NG_BY_YEAR_NGRAM = GROUP NG BY ( year, ngram );
DESCRIBE   NG_BY_YEAR_NGRAM;
ILLUSTRATE NG_BY_YEAR_NGRAM;


-- --------------------------------------------------------------------------------------------------
-- STEP 2: generate the ranking metric for each group and flatten
-- --------------------------------------------------------------------------------------------------
NG_BY_YEAR_NGRAM_RANKS = FOREACH NG_BY_YEAR_NGRAM GENERATE group.$0 as year, group.$1 as ngram, 
                            SUM(NG.pages) as psum,
                            AVG(NG.density) as davg, 
                            SUM(NG.pages)/(SUM(NG.books)+1E-6) as ametric;
DESCRIBE   NG_BY_YEAR_NGRAM_RANKS;
ILLUSTRATE NG_BY_YEAR_NGRAM_RANKS;


-- --------------------------------------------------------------------------------------------------
-- STEP 3: just group again by ngram, now this contains for each year, comparison metrics
-- --------------------------------------------------------------------------------------------------
NG_BY_YEAR_RANKS = GROUP NG_BY_YEAR_NGRAM_RANKS BY ngram;
DESCRIBE   NG_BY_YEAR_RANKS;
ILLUSTRATE NG_BY_YEAR_RANKS;


-- --------------------------------------------------------------------------------------------------
-- STEP 4: for each year, sort based on the cost metric, the associated bag tuples and retrieve the top ones
-- --------------------------------------------------------------------------------------------------
TOP_NGRAMS = FOREACH NG_BY_YEAR_RANKS {
    RESULTS = ORDER NG_BY_YEAR_NGRAM_RANKS BY davg, psum desc;
    TOP_RESULTS = LIMIT RESULTS 3;
    GENERATE FLATTEN( TOP_RESULTS );
}
DESCRIBE   TOP_NGRAMS;
ILLUSTRATE TOP_NGRAMS;












