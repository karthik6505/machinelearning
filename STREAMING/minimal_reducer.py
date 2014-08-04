#! /usr/bin/env python
'''
    @FILE: minimal_reducer.py
    @AUTHOR: NRM
'''

# ######################################################################
# http://blog.cloudera.com/blog/2013/01/a-guide-to-python-frameworks-for-hadoop/
# ######################################################################

import sys
import os
import re
HADOOP_SEP = "\t"



# ######################################################################
def reducer(hadoop_sep=HADOOP_SEP, max_errcount=-1, debug=0):
    METRICS = {}
    for line in sys.stdin:
	print >>sys.stdout, line.split("\t")
    return


# ######################################################################
if __name__ == "__main__":
    reducer()















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
