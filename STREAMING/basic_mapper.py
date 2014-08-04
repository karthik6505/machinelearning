#! /usr/bin/env python
'''
    @FILE: basic_mapper.py
    @AUTHOR: NRM
'''

# ######################################################################
# http://blog.cloudera.com/blog/2013/01/a-guide-to-python-frameworks-for-hadoop/
# ######################################################################

import sys
import os
import re

# ######################################################################
DEBUG = 0
HADOOP_SEP = "\t"
LOCAL_SEP  = ","
SUBST_CHAR = "_"
# ######################################################################


# ######################################################################
def token_cleaner( token, hadoop_sep=HADOOP_SEP, local_sep=LOCAL_SEP, debug=DEBUG):
    token = token.replace( hadoop_sep, SUBST_CHAR ) 
    token = token.replace( local_sep,  SUBST_CHAR ) 
    if debug>1: print >>sys.stderr, "[DEBUG]  [%32s]: [%s]" % ( "TOKEN", token )
    return token[:]
# ######################################################################

# ######################################################################
def generate_key( keyvals, sep=",", hadoop_sep=HADOOP_SEP, local_sep=LOCAL_SEP, debug=DEBUG):
    token = sep.join( [token_cleaner(str(x)) for x in keyvals] )
    token = token.replace( local_sep,  SUBST_CHAR ) 
    if debug>1: print >>sys.stderr, "[DEBUG]  [%32s]: [%s]" % ( "GENERATED KEY", token )
    return token[:]
# ######################################################################

# ######################################################################
def emit( key, emit_val, hadoop_sep=HADOOP_SEP, local_sep=LOCAL_SEP, debug=DEBUG):
    emit_key = token_cleaner( str(key) )
    output_tuple = "%s%s%s" % ( emit_key, hadoop_sep, emit_val )
    if debug>0: print >>sys.stderr, "[DEBUG]  [%32s]: [%s]" % ( "TUPLE", output_tuple )
    print >>sys.stdout, output_tuple
    return output_tuple[:]
# ######################################################################





 

DEBUG        = 0
YEAR_PATTERN = re.compile ( "([0-2]\d{3})" )
DM_METRICS   = { 'invalid_length': 0,
                 'invalid_fcount': 0,
                 'invalid_ngrams': 0,
                 'invalid_tuples': 0 }


# ######################################################################
def update_metrics( mtype, incr=1, debug=DEBUG ):
    if mtype in DM_METRICS: 
        DM_METRICS[mtype] = DM_METRICS[mtype] + incr
        if debug:
            print "[ERROR]    %32s : %s " % ( mtype.upper(), DM_METRICS[mtype] )
# ######################################################################


# ######################################################################
# dataset specific
# ######################################################################
def mapper_preprocess( line, hadoop_sep ):
    good_record, items = False, []

    year = YEAR_PATTERN.findall( line )
    if len(year) < 1:
        update_metrics( 'invalid_length' )
        return ( good_record, items )

    data = line.split(hadoop_sep)
    items = [ x.strip() for x in data if x.strip() ]
    if len(items) < 5:
        update_metrics( 'invalid_fcount' )
        return ( good_record, items )

    good_record = True
    return ( good_record, items )
# ######################################################################



# ######################################################################
def mapper( ngram_size, hadoop_sep=HADOOP_SEP, max_errcount=-1 ):
    for line in sys.stdin:
        ( ok, items ) = mapper_preprocess( line, hadoop_sep )
        if not ok:
            continue

        # unpack data
        try:
            ngram = [ x.strip() for x in items[0].split() if x ]
            if len(ngram)<ngram_size:
                update_metrics( 'invalid_ngrams' )
                continue
        except:
            update_metrics( 'invalid_tuples' )
            continue

        try:
            year  = int(items[1])
            count = int(items[2])
            pages = int(items[3])
            books = int(items[4])
        except:
            update_metrics( 'invalid_fcount' )
            continue

        # build key and emit
        keyvals = [ generate_key( ngram, sep=":" ), year ]
        emit_key = generate_key( keyvals, sep="," )
        emit_val = str( [ count, pages, books ] )
        emit( emit_key, emit_val )
    return


# ######################################################################
if __name__ == "__main__":
    # determine value of n in the current block of ngrams by parsing the filename
    try:
        input_file = os.environ['map_input_file']
    except:
        input_file = "./n5grams.txt"

    expected_ngram_size = int(re.findall(r'([\d]+)gram', os.path.basename(input_file))[0])

    ret = mapper( expected_ngram_size )

    
    if DEBUG:
        print >>sys.stderr, "_" * 80
        for k in DM_METRICS.keys():
            print >>sys.stderr, k, DM_METRICS[k]
        print >>sys.stderr, "_" * 80
# ######################################################################



# ######################################################################
'''
START DATASET/TUPLE DESCRIPTION
    INPUT: [ ngram_vector YEAR #HITS #PAGES #BOOKS ]
        KEY: ng1:ng2:..:ngn     (":" joined) 
        VAL: ( #HITS, #PAGES, #BOOKS )

    OUTPUT: [ KEY, VAL ]
        KEY: ng1,ng2,..,ngn     ("," joined) 
        VAL: ( #HITS, #PAGES, #BOOKS )

    # ######################################################################
    SOURCE: https://aws.amazon.com/datasets/8172056142375670

    DESCRIPTION:
        There are a number of different datasets available. Each dataset is a 
        single n-gram type (1-gram, 2-gram, etc.) for a given input corpus 
        (such as English or Russian text).

        We store the datasets in a single object in Amazon S3. The file is 
        in sequence file format with block level LZO compression. The sequence 
        file key is the row number of the dataset stored as a LongWritable and 
        the value is the raw data stored as TextWritable.

        The value is a tab separated string containing the following fields:
            n-gram - The actual n-gram
            year - The year for this aggregation
            occurrences - The number of times this n-gram appeared in this year
            pages - The number of pages this n-gram appeared on in this year
            books - The number of books this n-gram appeared in during this year

        The n-gram field is a space separated representation of the tuple.

    EXAMPLE:
        analysis is often described as  1991 1 1 1
        n50      n51 n52   n53      n54 YY  H P B
        |<--------------------------->|

        analysis:is:often:described:as,1991 ( 1 1 1 )
        |<--------mapper key------------->| |<----->|
END DATASET
'''
# ######################################################################


