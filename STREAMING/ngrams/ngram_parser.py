#! /usr/bin/env python

import re
import random
import string
import nltk

# #########################################################################################
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
try:
    STEMMER = PorterStemmer()
except:
    STEMMER = None
# #########################################################################################

sep     = "\t"

# #########################################################################################
def clean_token( word ):
    word = "".join([ x for x in word if x.strip() and x not in string.whitespace ])
    return word


# #########################################################################################
def parser( filename='text', debug=0 ):
    with open( 'text', 'r' ) as fp:
        DATA   = fp.read()
        DATA = DATA.lower()
        ASCII_DATA = "".join([ x for x in DATA if x in string.printable and 
                                                ( x in string.whitespace or 
                                                  x in string.letters or 
                                                  x in string.digits )])

        WORDS = nltk.word_tokenize(ASCII_DATA)
        WORDS = [ SnowballStemmer("english", ignore_stopwords=False).stem(word) for word in WORDS ]
        WORDS = [ word for word in WORDS if len(word)>1 ]

        NGRAMS = {}
        for i, word in enumerate(WORDS):
            if i<5: continue
            ngram_words = [ clean_token(x) for x in [ WORDS[i-4], WORDS[i-3], WORDS[i-2], WORDS[i-1], WORDS[i]  ] ]
            if len(ngram_words) < 5: continue
            ngram = " ".join( ngram_words )
            if ngram in NGRAMS:
                NGRAMS[ ngram ] =  NGRAMS[ ngram ] + 1
            else:
                NGRAMS[ ngram ] =  1

    for ngram in NGRAMS:
        year = random.randint(1995, 2004)
        pages = random.randint(3, 10)
        books = random.randint(1, 3)
        print "%-64s %s %4d %s %d %s %d %s %d" % ( ngram, sep, year, sep, NGRAMS[ngram], sep, pages, sep, books ) 


# #########################################################################################
if __name__ == "__main__":
    parser( filename="text" )
