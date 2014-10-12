#!/usr/bin/env python
__author__ = "Nelson R. Manohar"
__date__   = "Jun-25-2014"
__version__= "0.1b"

'''
 * Returns a ...  * <p>
 * This method always returns...
 *
 * @author      Nelson R. Manohar - (classes and interfaces only, required)
 * @version     v0.1b             - (classes and interfaces only, required. See footnote 1)
 * @param       text              - (methods and constructors only)
 * @return      aval              - (methods only)
 * @throws      IOexception       - (@exception is a synonym added in Javadoc 1.2)
 * @see         http://linkedin.com/in/nelsonmanohar
'''

import argparse
import sys


# #########################################################################################
LINE = "_" * 80


# #########################################################################################
# basic utility functions
# #########################################################################################
def argprint( argstring="argname", argval=None ):
    argval = "%s: %s" % ( argstring, argval )
    return argval


# #########################################################################################
def getitems( line, sep ):
    items = [ x.strip() for x in line.split(sep) ]
    return items[:]


# #########################################################################################
def getline( fp ):
    line = fp.readline()
    line.strip()
    return line[:]


# #########################################################################################
def nextline( fp, ln=-1 ):
    ln = ln + 1
    dataline = getline(fp)
    return ( ln, dataline[:] )


# #########################################################################################
def print_header( header, skipcols=[] ):
    print LINE
    skip_msg = "        *** COLUMN WILL BE DROPPED FROM OUTPUT"
    for i, col in enumerate(header):
        msg = ""
        if i in skipcols: msg = skip_msg
        print "%42s" % argprint( argstring=header[col], argval="at colnum %s" % col ), msg
    print LINE
    return header


# #########################################################################################
def get_header( filename="", sep="," ):
    with open( filename, "r" ) as fp:
        headerline = getline(fp)
        headeritems = getitems( headerline, sep )
        header = dict( zip( range(len(headeritems)), headeritems ) )
    return header, headeritems[:]


# #########################################################################################
def debug_line( line, ln, items, n ):
    retval = False
    m = len(items)
    if m == n: 
        retval = True
    if not retval:
        print >>sys.stderr, ln, m, "NOT_OK"
        print "line[%s] = %s" % (ln, line)
        print "line[%s] = %s" % (ln, items)
    else:
        pass # print >>sys.stderr, ln, m, "OK"
    return retval


# #########################################################################################
def drop_skipcols( items, skipcols=[], headeritems=[] ):
    validcols = sorted(set(range(len(items))).difference( set(skipcols)))
    if headeritems: 
        itemmatch = zip( headeritems, items ) 
        dropped = [(i,x) for i,x in enumerate(itemmatch) if i in skipcols]
        kept    = [(i,x) for i,x in enumerate(itemmatch) if i in validcols ]
        if args.verbose > 1:
            for x in dropped:
                print "DROPPED:", "%3s:     %-42s" % ( x[0], x[1] )
            for x in kept:
                print "KEPT:   ", "%3s:     %-42s" % ( x[0], x[1] )
    new_items = kept
    new_itemmatch = dict( kept )
    return ( new_items[:], new_itemmatch.copy() )


# #########################################################################################
def get_headerschema( filename="", sep=":" ):
    with open(filename, "r" ) as hfp:
        hlines = hfp.readlines()
        headeritems = [ getitems( hline, sep ) for hline in hlines ]
        header = dict( zip( range(len(headeritems)), headeritems ) )
    return header, headeritems[:]


# #########################################################################################
def pretty_print( itemmatch, mod=1 ):
    for i, x in enumerate(itemmatch):
        if i % mod == 0: print
        print argprint( x, itemmatch[x] ), 
    print


# #########################################################################################
def print_skipcols( header, skipcols ):
    print "THESE COLUMNS WILL BE DROPPED"
    skipped = [ ( x, header[x] ) for x in header if x in skipcols ]
    print skipped
    return skipped


# #########################################################################################
def clean_string( field, fdefault="", sep=",", repval="_" ):
    if sep not in ['\'', '"', ',', ':', ';']: pass
    field = repr(field)
    field = field.replace('\'', repval )
    field = field.replace('"',  repval )
    field = field.replace(',',  repval )
    field = field.replace(':',  repval )
    field = field.replace(';',  repval )
    field = field.replace(sep,  repval )
    if fdefault and field == repval+repval:
        field = fdefault
    return field


# #########################################################################################
def print_tuple( items, sep=",", PREFIX="" ):
    '''
    [(0, (['title', '0', 'chararray'], "'$1000 a Touchdown'")), 
     (1, (['year', '1', 'int'], '1939')), 
     (2, (['length', '2', 'int'], '71')), ... ]
    '''
    out_items =  []
    for i,x in enumerate( items ):
        coln = x[0]
        colf = x[1][0]
        fname = colf[0] 
        fnum  = colf[1] 
        ftype = colf[2]
        fdefault = ""
        if len(colf)>3: fdefault = colf[3]
        fval  = x[1][1]

        nfval = fval
        if ftype == 'chararray': 
            nfval = clean_string( fval, fdefault ) 
        elif ftype == 'int': 
            nfval = clean_string( fval, repval="" ) 
            if nfval == "": nfval = 0
            nfval = int(nfval)
        elif ftype == 'float': 
            nfval = clean_string( fval, repval="" ) 
            if nfval == "": nfval = 0.0
            nfval = float(nfval)
        else:
            print 'invalid field specification in tuple'
        if args.verbose > 2:
            print "%3s: %50s -->  %-12s" % ( i, x, nfval )
        out_items.append( nfval )
    out_tuple = sep.join([ str(x) for x in out_items])
    if args.verbose > 0: print PREFIX, out_tuple
    if args.outfile:
        with open( args.outfile, "a" ) as outfp: 
            print >>outfp, out_tuple
    return ( out_tuple, out_items[:] )


# #########################################################################################
def linestepper( filename="", sep=",", header=True, headerschema="", skipcols=[], PREFIX="" ):
    if args.outfile:
        with open( args.outfile, "w" ) as outfp: 
            pass

    if not headerschema: 
        header, headeritems = get_header( filename=filename, sep=sep )
        n = len(header)
    else:
        header, headeritems = get_headerschema( headerschema, sep=":" )
        n = len(header)
    print_header( header, skipcols )
    print headeritems
    print LINE

    with open( filename, "r" ) as fp:
        if header: lineno, line = nextline( fp, 0 )
        lineno, line = nextline( fp, 0 )
        while line:
            if args.verbose>1: print LINE
            items = getitems( line, sep )
            status = debug_line( line, lineno, items, n )
            if not status: break
            ( new_items, new_itemmatch ) = drop_skipcols( items, skipcols, headeritems )
            print_tuple( new_items, sep=",", PREFIX=PREFIX )

            lineno, line = nextline( fp, lineno )
            if args.maxlines and lineno > args.maxlines:
                break


# #########################################################################################
if __name__ == "__main__":
    print >>sys.stderr, "STARTING..."

    parser = argparse.ArgumentParser()

    if parser:
        parser.add_argument("-v", "--verbose", help="output verbosity level",    type=int, choices=[0,1,2,3] )
        parser.add_argument("-n", "--maxlines",help="num. data lines to process",type=int )
        parser.add_argument("-c", "--cols",    help='columns to skip on parsing',type=int, nargs='+' )
        parser.add_argument("-o", "--outfile", help='file store for parsed tuples' )

    args = parser.parse_args()

    if args:
        print LINE
        print "GIVEN ARGS:", args
        print LINE

        if args.maxlines:  print argprint( "num lines to scan", args.maxlines )
        if args.verbose:   print argprint( "debug verbosity",   args.verbose )
        if args.cols:      print argprint( "columns to drop",   args.cols)
        if args.outfile:   print argprint( "output file",       args.outfile)
        print LINE

    skipcols = []  # skipcols=set(range(6,10)).union(set(range(10,16)))
    if args.cols: skipcols=set(args.cols)

    # linestepper( filename="movies.tab", headerschema="movies.header", sep="\t", skipcols=skipcols )
    linestepper( filename="testdata.manual.2009.06.14.csv", 
                 headerschema="testdata.manual.header", sep='","', skipcols=skipcols, PREFIX=">>>" )

    print >>sys.stderr, "DONE"

