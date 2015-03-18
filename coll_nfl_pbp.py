#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''Script or Module Title
    
    This section should be a summary of important information to help the editor
    understand the purpose and/or operation of the included code.
    
    List of classes: -none-
    List of functions:
        main
'''

# built-in modules
import urllib2
from pprint import pprint
import re

# third-party modules
#import urltools
from bs4 import BeautifulSoup


#==============================================================================
def main(*args):
    url = 'http://www.pro-football-reference.com/boxscores/201409040sea.htm'
    pg = urllib2.urlopen(url)
    pgtxt = pg.read()
    pg.close()
    pgsoup = BeautifulSoup(pgtxt)
    pbptbl = pgsoup.find('table', id='pbp_data')
    # row headings
    # 0        1     2     3     4         5       6      7      8    9    10
    # Quarter  Time  Down  ToGo  Location  Detail  VisSc  HmSc   EPB  EPA  Win%
    # Note: scores are FOLLOWING the play
    rows = pbptbl('tr')
    
    # Write to csv
    with open('pbp_data_test.csv', 'w') as f:
        f.write('Quarter,Time,Down,ToGo,Pos,Description,VisSc,HmSc\n')
        for r in rows:
            if re.search(r'thead', ' '.join(r['class'])): continue
            cells = r('td')
            for i in range(len(cells)-3):
                if cells[i].string is not None:
                    f.write( str(cells[i].string) )
                else:
                    s = re.sub(r'^<[^<>]*?>|</[^<>]*?>$', '"', str(cells[i]))
                    s = re.sub(r'<[^<>]*?>\s*</[^<>]*?>', '', s)
                    s = re.sub(
                        r'<a [^<>]*?players[^<>]*?/(\w+)\.htm[^<>]*>'
                        + r'[^<>]*</a>', r'[p:\1]', s
                    )
                    f.write(s)
                if i < len(cells)-3-1: f.write(',')
                else: f.write('\n')
            # END for
        # END for
    # END with
# END main

#==============================================================================
if __name__ == '__main__':
    main()
# END if