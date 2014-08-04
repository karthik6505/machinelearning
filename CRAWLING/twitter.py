import urllib
import json
import oauth2
import urllib2
import urllib3
import tweepy

url = 'http://search.twitter.com/search.json?q=microsoft'
response = urllib.urlopen( url )
pyresponse = json.load( response )
print pyresponse
for k in pyresponse:
    print k, pyresponse[k]

# ############################################################################
def do_urlget( ):
    pass


# ############################################################################
def write_file():
    pass

# ############################################################################
def setup_credentials():
    pass

# ############################################################################
def get_tweets( user="", key="" ):
    pass

# ############################################################################
def run():
    pass


class twitter_downloader( Object ):
    def __init__( self, user="", password="" ): 
        self._credentials = {}
    def get_tweets( self ):
        pass

if __name__ == "__main__":
    run()

