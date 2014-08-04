#!/usr/bin/env python
#-*- coding: utf-8 -*



# ##############################################################################
import os
import sys
import time
import socket
import GeoIP
import re
import mimetypes
from urlparse import urlparse
from datetime import datetime
# ##############################################################################



# ##############################################################################
# GeoLite License --- https://pypi.python.org/pypi/GeoIP/
# ##############################################################################
# The GeoLite databases are distributed under the Creative Commons Attribution-ShareAlike 3.0 
# Unported License. The attribution requirement may be met by including the following 
# in all advertising and documentation mentioning features of or use of this database:
# 
#   This product includes GeoLite data created by MaxMind, available from
#       <a href="http://www.maxmind.com">http://www.maxmind.com</a>.
# ##############################################################################
# import pygeoip
# from pygeoip import geoiplite2
#
# match = geolite2.lookup('64.242.88.10')
# <IPInfo ip='64.242.88.10' country='US' continent='NA' subdivisions=frozenset(['MO']) timezone='America/Chicago' location=(38.65, -90.5334)>
#
# gi = GeoIP.open("/home/nrm/WORKSPACE/GEOIP/GeoLiteCity.dat", GeoIP.GEOIP_STANDARD)
# gir = gi.record_by_name("www.google.com")
# gir = gi.record_by_addr("24.24.24.24")
#
# with open_database('/home/nrm/WORKSPACE/SRC/GEOIP/GeoLite2-City.mmdb') as db:
#    match = db.lookup_mine()
# ##############################################################################


# ##############################################################################
HADOOP_FILENAME=None
HOST2IP_DBPATH=None
GEODBQRY = GeoIP.open("/home/nrm/WORKSPACE/GEOIP/GeoLiteCity.dat", GeoIP.GEOIP_STANDARD)
SKIP_WORDS = [ 'dsl.', 'dialup', 'dialin' ]
TIME_FORMAT="%a %Y/%m/%d %I:%M%p"
LOG_REGEX = re.compile( "(\w{3}) (\d{4})/(\d{2})/(\d{2}) (.*)" )
IP, HOST = {}, {}
HOSTS={}
DEBUG=0
LINENO = 0
# ##############################################################################


# ##############################################################################
# ##############################################################################
def get_timestamp():
    tstamp = time.time()
    return tstamp

def get_timetuple():
    # datetime.now() == datetime.fromtimestamp(time.time())
    dt_tuple = datetime.now()
    return dt_tuple

def format_timetuple( dt_tuple ):
    if dt_tuple:
        stderr_time_format = dt_tuple.strftime( TIME_FORMAT )
        return stderr_time_format
    return ''

def format_timetimestamp( tstamp ):
    dt_tuple = datetime.fromtimestamp( tstamp )
    stderr_time_format = format_timetuple( dt_tuple )
    return stderr_time_format 

def debug( out, level=0, dtype="INFO", tabc="." ):
    stderr_time_format = format_timetuple( get_timetuple() )
    tab = tabc * level 
    msg = "[%s]%s[%s]: [%s]" % ( stderr_time_format, tab, dtype, out )
    print >>sys.stderr, msg
    return msg
# ##############################################################################


# ##############################################################################
def get_ip( haddr ):
    ip = None
    haddr = haddr.strip()
    if haddr in IP:
        ip = IP[haddr] 
    else:
        try:
            items = haddr.split('.')
            if len(items)>3:
                haddr_new = ".".join(haddr.split('.')[-3:])
                ip = socket.gethostbyname( haddr_new )
            else:
                ip = socket.gethostbyname( haddr )
        except:
            try:
                haddr_new = ".".join(haddr.split('.')[-3:])
                ip = socket.gethostbyname( haddr_new )
            except:
                ip = "N/A:%s" % haddr
        IP[haddr] = ip[:]
    return ip


# ##############################################################################
def get_host( ip, known_host="" ):
    host = None
    ip = ip.strip()
    if ip in HOST:
        host = HOST[ip] 
    else:
        try:
            host = socket.gethostbyaddr(ip)
        except:
            host = ip[:]
            ip = get_ip( host )
        HOST[ip] = host[:]
    return host


# ##############################################################################
def clean_gdata_defaults( gdata ):
    if gdata:
        for k in gdata:
            if gdata[k] == None:
                gdata[k] = "N/A"
    else:
        gdata = {'city': 'N/A', 'region_name': 'N/A', 'region': 'N/A', 'area_code': 000, 'time_zone': 'N/A', 
                 'longitude': 0.0000000000000, 'metro_code': 000, 'country_code3': 'N/A', 'latitude': 0.0000000000000, 
                 'postal_code': '00000', 'dma_code': 000, 'country_code': 'N/A', 'country_name': 'N/A'}
    return gdata.copy()


# ##############################################################################
def lookup_hostname_subseq( host, index=3, splitby="." ):
    gdata = {}
    h_items = host.split(splitby)
    if len(h_items)>index:
        name_subseq = ".".join(h_items[-index:])
        gdata = GEODBQRY.record_by_name(name_subseq)
    return gdata


# ##############################################################################
def get_geoip_lookup_data( ip_host ):
    gdata = GEODBQRY.record_by_addr(ip_host)
    if not gdata:
        gdata = GEODBQRY.record_by_name(ip_host)
    if not gdata:
        gdata = lookup_hostname_subseq( ip_host, index=3, splitby="." )
    if not gdata:
        gdata = lookup_hostname_subseq( ip_host, index=2, splitby="." )
    if not gdata:
        gdata = lookup_hostname_subseq( ip_host, index=3, splitby="_" )
    gdata = clean_gdata_defaults( gdata )
    return gdata.copy()


# ##############################################################################
def emit_output_schema( out ):
    print "\t".join( [ str(x) for x in out ] ) 


# ##############################################################################
# INPUT_SCHEMA = {"host":"fw1.millardref.com","identity":"-","user":"-","time":"[09/Mar/2004:08:18:19 -0800]","request":"\"GET /favicon.ico HTTP/1.1\"","status":"200","size":"1078"}
def ingest_input_schema( schema ):
    d_items = eval( schema )
    host     = d_items['host']
    identity = d_items['identity']
    user     = d_items['user']
    timestamp= d_items['time'][1:-1]
    request  = d_items['request']
    status   = d_items['status']
    size     = d_items['size']
    return ( host, identity, user, timestamp, request, status, size )     


# ##############################################################################
def emit_default_tuple( val="0", n=25 ):
    out = [ val ] * n
    emit_output_schema( out )
    return out


# ##############################################################################
def line_processor( line ):
    if sum([ 1 for x in SKIP_WORDS if x in line ]):
        debug( "PRIVACY SKIPPED: %s" % LINENO, level=0, dtype="INFO", tabc="." )
        emit_default_tuple()
        return

    if "NULL" in line.split('\t'):
        debug( "CRAP SKIPPED: %s" % LINENO, level=0, dtype="INFO", tabc="." )
        emit_default_tuple()
        return

    try:
        ( host, identity, user, timestamp, request, status, size )  = ingest_input_schema( line )
        debug( "KEYPAIR: %s" % LINENO, level=0, dtype="INFO", tabc="." )
    except:
        line  = line.strip()
        items = line.split('\t')
        host     = items[0]
        identity = items[1]
        user     = items[2]
        timestamp= items[3][1:-1]
        request  = items[4]
        status   = items[5]
        size     = items[6]

    ip = host
    # ip = get_ip( host ) 
    # host = get_host( ip ) 

    time_date = timestamp.split(' ')[0]
    wday, year, month = "N/A", "N/A", "N/A"
    try:
        dt = datetime.strptime( time_date, "%d/%b/%Y:%H:%M:%S")
        log_time = format_timetuple( dt )  # "%a %Y/%m/%d %I:%M%p"
        match_items = LOG_REGEX.findall( log_time ) 
        if match_items:
            wday, year, month = match_items[0][0], match_items[0][1], match_items[0][2]
    except:
        debug( "EXCEPTION: %s format unexpected @ LINENO %s" % (time_date, LINENO), level=0, dtype="INFO", tabc="." )

    gdata = get_geoip_lookup_data( host )

    # ParseResult(scheme='http', netloc='www.cwi.nl:80', path='/%7Eguido/Python.html', params='', query='', fragment='')
    uri = ''
    uri = urlparse(request)


    mtype = mimetypes.guess_type( request.split(' HTTP/1.')[0].lower())[0] 
    if not mtype:
        mtype = "file/obj"

    out = [ gdata['city'], gdata['area_code'], gdata['region_name'], gdata['postal_code'], gdata['country_code3'], 
            ip[:], host[:], 
            identity[:], user[:], wday[:], year[:], month[:], timestamp[:],  
            mtype[:],
            size[:],
            status[:],
            uri.scheme, uri.netloc, uri.path, uri.params, uri.query, uri.fragment,
            request[:],
            gdata['longitude'], gdata['latitude'] ]

    emit_output_schema( [ str(x) for x in out[:] ] )

    return




# ##############################################################################
# ##############################################################################
try:
    HADOOP_FILENAME = os.environ["map_input_file"]
    HOST2IP_DBPATH  = "/user/hduser/apachelogs/host2ip.db"
except:
    HADOOP_FILENAME = None
    HOST2IP_DBPATH  = "./host2ip.db"
debug( "HADOOP_FILENAME: %s" % HADOOP_FILENAME )

for line in sys.stdin:
    unicode_line = unicode(line, 'utf8')
    line = unicode_line.encode('utf8', 'replace')

    LINENO = LINENO + 1
    try:
        line_processor( line )
        debug( "PROCESSED: %s bytes @ LINENO %s" % (len(line), LINENO), level=0, dtype="INFO", tabc="." )
    except:
        emit_default_tuple()
        debug( "SKIPPED  : %s bytes @ LINENO %s" % (len(line), LINENO), level=0, dtype="INFO", tabc="." )
        try:
            debug( "LINE %s = %s" % (LINENO, line ), level=1, dtype="INFO", tabc="." )
        except:
            debug( "LINE %s = %s" % (LINENO, repr(line) ), level=1, dtype="INFO", tabc="." )
        debug( "%s" % ('-' * 80), level=1, dtype="INFO", tabc="." )


