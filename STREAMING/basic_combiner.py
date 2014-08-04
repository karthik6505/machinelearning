#! /usr/bin/env python
'''
    @FILE: basic_combiner.py
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


# ######################################################################
def combiner(hadoop_sep=HADOOP_SEP, max_errcount=-1, debug=0):
    old_key, new_key = None, None
    METRICS = ( 0, 0, 0 )
    for line in sys.stdin:
        data = line.split(hadoop_sep)
        items = [ x.strip() for x in data if x ]
        recv_key  = items[0]                         # key
        recv_vals = items[1].split(hadoop_sep)       # iterator of mapper values

        # if running in MR, entries are sorted from shuffer
        new_key = recv_key
        if new_key != old_key:
            if old_key:
                emit_key = old_key 
                emit_val = METRICS
                emit( emit_key, emit_val, hadoop_sep=hadoop_sep )
            old_key = new_key[:]
            METRICS = [ 0, 0, 0 ]

        # retrieve data-tuple from hardcoded format that was put in (tuple inside list)
        recv_vals = eval(recv_vals[0])
        views_per_year = int(recv_vals[0])
        pages_per_year = int(recv_vals[1])
        books_per_year = int(recv_vals[2])

        # perform the computation of this combiner
        UPDATE = [views_per_year, pages_per_year, books_per_year ]
        METRICS = [ x+y for (x,y) in zip(METRICS, UPDATE ) ]

    # print last record, as it was not triggered
    emit_val = UPDATE
    emit( emit_key, emit_val, hadoop_sep=hadoop_sep )
    return


# ######################################################################
if __name__ == "__main__":
    combiner()















# ######################################################################



# ######################################################################
'''
START DATASET/TUPLE DESCRIPTION
    INPUT: [ KEY, VALUES ]
        KEY: ng1,ng2,..,ngn     ("," joined) 
        VALLUES: [ ( #HITS, #PAGES, #BOOKS ), 
                   ( #HITS, #PAGES, #BOOKS ), ...
                   ( #HITS, #PAGES, #BOOKS ) ]

    OUTPUT: [ KEY, VALUES ]
        KEY: ng1,ng2,..,ngn     ("," joined) 
        VALUES: [ ( SUM(#HITS), SUM(#PAGES), SUM(#BOOKS) ) ]

    # ######################################################################
    SOURCE: https://aws.amazon.com/datasets/8172056142375670

    DESCRIPTION:

    EXAMPLE:

END DATASET
'''
# ######################################################################
