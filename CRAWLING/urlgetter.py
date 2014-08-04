#!/usr/bin/env python

import re 
import sys
import urllib
import urllib2
import time

# ###################################################################################
def get_url_list( filename="" ):
    lines = []
    if filename:
        with open(filename, "r") as fp:
            lines = fp.readlines()
            lines = [x.strip() for x in lines if x.strip()]
    return lines


# ###################################################################################
def do_post( url="", values={} ):
    html = ""
    if url:
        data = urllib.urlencode(values)
        req = urllib2.Request(url, data)
        response = urllib2.urlopen(req)
        html = response.read()
    return html








# ###################################################################################
def get_url( url="" ):
    html = ""
    if url: 
        try:
            response = urllib2.urlopen(url)
            html = response.read()
        except:
            print "[ERROR] [%s] could not be fetched" % url
    return html


# ###################################################################################
def write_file( filename="", data="", debug=1 ):
    n = len(data)
    try:
        with open( filename, "w" ) as fp:
            fp.write( data )
            if debug>1: print "[STATUS]: [%s] bytes written" % n
    except:
        if debug>0: print "[ERROR]: [%s] could not write to disk" % filename
    return n


# ###################################################################################
def get_gutenberg_book( booknumber="" ):
    gutenberg_url = "http://www.gutenberg.org/ebooks/"
    book_url = "%s%s" % ( gutenberg_url, booknumber )
    book_metadata = get_url( book_url )  
    book_url = "%s%s" % ( book_url, ".txt.utf-8" )
    book_data = get_url ( book_url )
    return ( book_metadata, book_data )

    
# ###################################################################################
def store_gutenberg_book( booknumber="", debug=1 ):
    book_metadata, book_data = "", ""
    if booknumber:
        try:
            ( book_metadata, book_data ) = get_gutenberg_book( booknumber )
        except:
            if debug>0: print "[ERROR] could not retrieve book"
            return False

        write_file( "%s.txt" % booknumber, book_data )
        write_file( "%s.dat" % booknumber, book_metadata )
        return True


# ###################################################################################
def retrieve_guttenberg_books( items=[], delay=3, debug=1 ):
    nsuccess = 0
    if items:
        for booknumber in items:
            ret = store_gutenberg_book( booknumber )
            if ret: nsuccess += 1
            time.sleep( delay )
        if debug: print "[INFO] crawl retrieved %s books out of %s" % ( nsuccess, len(items) )
    return

# ###################################################################################
def misc():
    args = sys.args 
    if args:
        filename = args[1]
    url = 'http://www.someserver.com/cgi-bin/register.cgi'
    values = {'name' : 'Michael Foord',
          'location' : 'Northampton',
          'language' : 'Python' }

# ###################################################################################

    
if __name__ == "__main__":
    books = [ 345, 120, 4363, 730, 10, 74, 161, 2701, 236, 6130, 100, 2591, 2600, 174, 1232, 768, 12 ,
              98, 28054, 1497, 8800, 5200, 521, 45, 844, 1400, 22381, 84, 829, 55, 16, 35, 158 ]

    retrieve_guttenberg_books( items=books, delay=3, debug=1 )

