# <table><tr><th><img src="/icons/blank.gif" alt="[ICO]"></th><th><a href="?C=N;O=D">Name</a></th><th><a href="?C=M;O=A">Last modified</a></th><th><a href="?C=S;O=A">Size</a></th><th><a href="?C=D;O=A">Description</a></th></tr><tr><th colspan="5"><hr></th></tr>
# <tr><td valign="top"><img src="/icons/back.gif" alt="[DIR]"></td><td><a href="/ml/machine-learning-databases/">Parent Directory</a></td><td>&nbsp;</td><td align="right">  - </td><td>&nbsp;</td></tr>
# <tr><td valign="top"><img src="/icons/unknown.gif" alt="[   ]"></td><td><a href="poker-hand-testing.data">poker-hand-testing.data</a></td><td align="right">25-Feb-2007 22:46  </td><td align="right"> 23M</td><td>&nbsp;</td></tr>
# <tr><td valign="top"><img src="/icons/unknown.gif" alt="[   ]"></td><td><a href="poker-hand-training-true.data">poker-hand-training-true.data</a></td><td align="right">25-Feb-2007 22:46  </td><td align="right">599K</td><td>&nbsp;</td></tr>
# <tr><td valign="top"><img src="/icons/text.gif" alt="[TXT]"></td><td><a href="poker-hand.names">poker-hand.names</a></td><td align="right">25-Feb-2007 22:45  </td><td align="right">5.8K</td><td>&nbsp;</td></tr>

import sys
import re

p = re.compile ('(.*)">(.*)</a>')
t = p.findall('/ml/machine-learning-databases/">Parent Directory</a>')
print t 

for line in sys.stdin:
    items = line.split( 'href="' )
    if len(items) == 2:
        url = items[1]
        try:
            t = p.findall(url)[0]
        except:
            continue
        print t[0], t[1]
    print '-' * 80
