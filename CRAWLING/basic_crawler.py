#!/usr/bin/env python
# -*- coding: utf-8 -*- 

'''
-------------------------------------------------------------------------------------------
FROM: How to make a web crawler [Python and requests], Forked from Basic Python.
HTTP://runnable.com/UqqXuSGIpqAeAAPR/how-to-make-a-web-crawler-for-python-and-requests
USED: originally, an example for trying to collect e-mail addresses from a website
AUTHOR: cezary
MODIFIED BY: Nelson & Hobbes (Holy Spirit aka True and Only Builder of the Pyramids)
-------------------------------------------------------------------------------------------
'''
print (__doc__)


# ##########################################################################################
import sys
import re
import time
import os
import requests
import urlparse
from os import path
# ##########################################################################################


# ##########################################################################################
# HTML <a> regexp
# Basic e-mail regexp:
# letter/number/dot/comma @ letter/number/dot/comma . letter/number
# Matches href="" attribute
# ##########################################################################################
EMAIL_RE = re.compile(r'([\w\.,]+@[\w\.,]+\.\w+)')
FILE_RE  = re.compile(r'([\w\.,]+@[\w\.,]+\.\w+)')
LINK_RE  = re.compile(r'href="(.*?)"')
FPATH_RE = re.compile('.+tp://(.*)?/(.*)')
VISITED  = {}
SKIPPED  = {}
FINDINGS = {}
METRICS  = {}
BASEURL  = None
DEBUG    = False
DELAY    = 0.5
# ##########################################################################################


# ##########################################################################################
def add_timestamp( uritask, t ):
    global METRICS
    if uritask not in METRICS: METRICS[ uritask ] = [ ]
    METRICS[ uritask ].append( t )

 
# ##########################################################################################
def write_file( response, filecontents, filename, filetype, where="./" ):
    try:
        filepath = urlparse.urlparse(filename)
        filepathname = where + str(filepath.netloc) + str(filepath.path)
        try:
            os.makedirs( os.path.dirname(filepathname) )
        except Exception as exception:
            pass
        with open( filepathname, "w" ) as fp:
            filecontents = filecontents.encode('utf8', 'replace')
            print "writing to filepath:", filepathname, len(filecontents), "bytes:\n"
            fp.write(filecontents)
            if DEBUG: 
                print '-' * 80
                print filecontents
                print '-' * 80
                print '-' * 80
    except Exception as exception:
        print >>sys.stderr, 'exception reported, file writing...', type(exception).__name__
    return


# ##########################################################################################
def is_valid_file( ext, invalid_files=[] ):
    if ext[1].lower() in invalid_files: 
        return 0
    return 1


# ##########################################################################################
def stats():
    print
    print '-' * 80
    print "SKIPPED REVISITATIONS"
    print '-' * 80
    for i,v in enumerate(sorted(VISITED.keys())):
        print "%s \t %s \t %s" % ( i, VISITED[v], v )
    print '-' * 80
    print "SKIPPED OUTWARD URI"
    print '-' * 80
    for i,v in enumerate(sorted(SKIPPED.keys())):
        print "%s \t %s \t %s" % ( i, SKIPPED[v], v )
    print '-' * 80
    print "URI RETRIEVAL TIMES (ACCOUNTS FOR RECURSION)"
    for i,v in enumerate(sorted(METRICS.keys())):
        try:
            print "%s \t %24.8f \t %s" % ( i, METRICS[v][1]-METRICS[v][0], v )
        except:
            print "%s \t %24s \t %s" % ( i, METRICS[v], v )
    print '-' * 80


# ##########################################################################################
def print_findings( findings, ftype="emails" ):
    print
    print '-' * 80
    print "Scrapped e-mail addresses:"
    print '-' * 80
    for i, f in enumerate(sorted(findings)):
         print "%s \t %s \t %s" % ( i, findings[f], f )
    print '-' * 80


# ##########################################################################################
def crawl(url, maxlevel, look_for, within_site_only=True, invalid_files=[]):
    global BASEURL
    global FINDINGS

    if not BASEURL:
        BASEURL = url[:]
        print "Crawling Base URL: %s" % BASEURL

    print ".",
    if url in SKIPPED:
        SKIPPED[url] = SKIPPED[url]+1
        print >>sys.stderr, "...known site-leap: %s" % url
        return FINDINGS

    if within_site_only and BASEURL not in url:
        SKIPPED[url] = 1
        print >>sys.stderr, "...skipping site-leap: %s" % url
        return FINDINGS

    if url in VISITED: 
        print >>sys.stderr, "...already been to: %s" % url
        VISITED[url] = VISITED[url] + 1
        return FINDINGS

    ext = path.splitext(url)
    if not is_valid_file(ext, invalid_files=invalid_files):
        print >>sys.stderr, "...skipping url with invalid ext: %s" % url
        return FINDINGS

    # Limit the recursion, we're not downloading the whole Internet
    if(maxlevel == 0):
        return FINDINGS

    print "VISITING", url
    VISITED[url] = 1
    add_timestamp( url, time.time() )
    time.sleep(DELAY)

    # Get the webpage
    try:
        req = requests.get(url)
    except Exception as exception:
        print >>sys.stderr, 'exception reported, fetch...', type(exception).__name__
        add_timestamp( url, time.time() )
        return FINDINGS

    # Check if successful
    if(req.status_code != 200):
        print >>sys.stderr, "%s error retrieving file: %s" % ( req.status_code, url )
        add_timestamp( url, time.time() )
        return FINDINGS

    RESULT = []
    filecontents = req.text[:]
    filename = url[:]
    filetype = ext[:]

    # Find and follow all the links
    links = LINK_RE.findall(req.text)
    for link in links:
        # Get an absolute URL for a link
        link = urlparse.urljoin(url, link)
        new_result = crawl(link, maxlevel - 1, look_for, within_site_only, invalid_files )
        if new_result:
            RESULT += new_result

    # Find all emails on current page
    if "emails" in look_for.lower():
        RESULT.append( EMAIL_RE.findall(req.text) )
        write_file( req, filecontents, filename, filetype )

    if "files" in look_for.lower():
        RESULT.append( FILE_RE.findall(req.text) )
        write_file( req, filecontents, filename, filetype )

    for r in RESULT:
        r = str(r)
        if r not in FINDINGS: FINDINGS[r] = 0
        FINDINGS[r] = FINDINGS[r] + 1

    add_timestamp( url, time.time() )
    return FINDINGS


# ##########################################################################################
if __name__ == "__main__":
    argc, args = len(sys.argv), sys.argv

    WHERE = 'http://nelsonmanohar.x10host.com/'
    DEPTH = 6
    if argc>0 and args[1]: WHERE = args[1]
    if not WHERE.endswith('/'): WHERE += '/'
    if argc>1 and args[2]: DEPTH = int(args[2])
    print WHERE, DEPTH

    findings = crawl( WHERE, DEPTH, "emails", 
                      within_site_only=True, 
                      invalid_files=[".gif", ".png", "jpg", ".tif", ".flv", ".css"])

    stats()

    print_findings( findings )


