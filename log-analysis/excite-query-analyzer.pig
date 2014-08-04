/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

-- Query Phrase Popularity (Hadoop cluster)

-- This script processes a search query log file from the Excite search engine and finds search phrases that occur with particular high frequency during certain times of the day. 


-- Register the tutorial JAR file so that the included UDFs can be called in the script.
REGISTER /home/hduser/WORKSPACE/DATASETS//tutorial.jar;

-- Use the  PigStorage function to load the excite log file into the “raw” bag as an array of records.
-- Input: (user,time,query) 
-- raw0 = LOAD 'excite.log.bz2' USING PigStorage('\t') AS (user, time, query);
-- raw  = LOAD 'excite_small.log' USING PigStorage('\t') AS (user, time, query);
raw  = LOAD 'excite_small.log' USING PigStorage('\t') AS (user, time, query);

-- Call the NonURLDetector UDF to remove records if the query field is empty or a URL. 
clean1 = FILTER raw BY org.apache.pig.tutorial.NonURLDetector(query);

-- Call the ToLower UDF to change the query field to lowercase. 
-- clean1: {user: bytearray,time: bytearray,query: bytearray}
clean2 = FOREACH clean1 GENERATE user, time, org.apache.pig.tutorial.ToLower(query) as query;

-- Because the log file only contains queries for a single day, we are only interested in the hour.
-- The excite query log timestamp format is YYMMDDHHMMSS.
-- Call the ExtractHour UDF to extract the hour (HH) from the time field.
-- clean2: {user: bytearray,time: bytearray,query: chararray}
houred = FOREACH clean2 GENERATE user, org.apache.pig.tutorial.ExtractHour(time) as hour, query;

-- houred: {user: bytearray,hour: chararray,query: chararray}
-- Call the NGramGenerator UDF to compose the n-grams of the query.
ngramed1 = FOREACH houred GENERATE user, hour, org.apache.pig.tutorial.NGramGenerator(query) as ngram;

-- here we may want to reflect a weight for an ngram being repeated by a user in an hour
-- ngramed1: {user: bytearray,hour: chararray,ngram: chararray}
ngramed3 = GROUP ngramed1 BY (user, hour, ngram);

-- ngramed3: {group: (user: bytearray,hour: chararray,ngram: chararray),ngramed1: {(user: bytearray,hour: chararray,ngram: chararray)}}
ngramed4 = FOREACH ngramed3 GENERATE group.$0 as user, group.$1 as hour, group.$2 as ngram, COUNT($1) as usrweight;

-- ngramed4: {user: bytearray,hour: chararray,ngram: chararray,usrweight: long}
ngramed2prime = DISTINCT ngramed4;

-- ngramed1: {user: bytearray,hour: chararray,ngram: chararray}
-- Use the  DISTINCT command to get the unique n-grams for all records.
ngramed2 = DISTINCT ngramed1;

-- ngramed2prime: {user: bytearray,hour: chararray,ngram: chararray,usrweight: long}
-- ngramed2: {user: bytearray,hour: chararray,ngram: chararray}
-- hour_frequency1: {group: (ngram: chararray,hour: chararray),ngramed2: {(user: bytearray,hour: chararray,ngram: chararray)}}
-- Use the  GROUP command to group records by n-gram and hour. 
hour_frequency1 = GROUP ngramed2 BY (ngram, hour);

-- hour_frequency2: {group::ngram: chararray,group::hour: chararray,count: long}
-- Use the  COUNT function to get the count (occurrences) of each n-gram. 
hour_frequency2 = FOREACH hour_frequency1 GENERATE group.$0 as ngram, group.$1 as hour, COUNT($1) as count; 

-- Use the  GROUP command to group records by n-gram only. 
-- Each group now corresponds to a distinct n-gram and has the count for each hour.
uniq_frequency1 = GROUP hour_frequency2 BY ngram;

-- uniq_frequency1: {group: chararray,hour_frequency2: {(group::ngram: chararray,group::hour: chararray,count: long)}}
-- For each group, identify the hour in which this n-gram is used with a particularly high frequency.
-- Call the ScoreGenerator UDF to calculate a "popularity" score for the n-gram.
uniq_frequency2 = FOREACH uniq_frequency1 GENERATE group, FLATTEN(org.apache.pig.tutorial.ScoreGenerator($1));

-- uniq_frequency2: {group: chararray,org.apache.pig.tutorial.scoregenerator_hour_frequency2_823::hour: chararray,org.apache.pig.tutorial.scoregenerator_hour_frequency2_823::score: double,org.apache.pig.tutorial.scoregenerator_hour_frequency2_823::count: long,org.apache.pig.tutorial.scoregenerator_hour_frequency2_823::mean: double}
-- Use the  FOREACH-GENERATE command to assign names to the fields. 
uniq_frequency3 = FOREACH uniq_frequency2 GENERATE $0 as ngram, $1 as hour, $2 as score, $3 as count, $4 as mean;

-- uniq_frequency3: {hour: chararray,ngram: chararray,score: double,count: long,mean: double}
-- Use the  FILTER command to move all records with a score less than or equal to 2.0.
filtered_uniq_frequency = FILTER uniq_frequency3 BY score > 2.0;

-- Use the  ORDER command to sort the remaining records by hour and score. 
ordered_uniq_frequency = ORDER filtered_uniq_frequency BY hour, score;

-- Use the  PigStorage function to store the results. 
-- Output: (hour, n-gram, score, count, average_counts_among_all_hours)
rmf script1-hadoop-results
STORE ordered_uniq_frequency INTO 'script1-hadoop-results' USING PigStorage();
