#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''Internet Data Collection Module for NFL
    
    Module dependencies:
        sys
    List of classes: -none-
    List of functions:
        main
'''

# built-in modules
import sys
import traceback
import re
import urllib2
from math import *
import time
import multiprocessing as mp
import pickle
from pprint import pprint
import random

# third-party modules
try:
    import NFL_class as NFL
    import urltools
    from bs4 import BeautifulSoup
except:
    sys.path.append('../../../ampPy/')
    
    import urltools
    import NFL_class as NFL
    from bs4 import BeautifulSoup
    
    print '*** added "../../../ampPy/" to sys.path ***'
# END try

#===============================================================================
def main(*args):
    for year in range(2009, 2015):
        print 'downloading ' + str(year)
        games = download_schedule_espn(year, 'all', csv=True, debug=False)
        print 'saving pickle'
        # Save results for later
        pickle_fn = 'NFL_games_{}.pypickle'.format(year)
        f = open(pickle_fn, 'wb')
        pickle.dump(games, f)
        f.close()
    # END for
# END main

#===============================================================================
def download_schedule_espn(year, return_filter='all', csv=False, debug=False):
    '''NFL Schedule/Results Downloader For ESPN.com
    
    TODO: hosted bool is not explicitly set, assumed True
    
    Args:
        year (str|int): Year to download in "yyyy" format.
        return_filter (str): Specifies which games should be returned.
                            Acceptible values include: all, final, upcoming.
        debug (bool): Saves a copy of the downloaded games and has enables more
                    runtime output in the terminal
    Returns:
        (list) [Game_1, Game_2, Game_3, ...]
    '''
    
    # Normalize year argument
    if isinstance(year, int):
        year = str(year)
        if len(year) > 4:
            raise ValueError('"{}" is not a valid year'.format(year))
    # END if
    
    # Download all webpages using multiprocessing
    # Regular Season
    url_template = (
            "http://scores.espn.go.com/nfl/scoreboard" +
            "?seasonYear={}&seasonType=2".format(year) +
            "&weekNumber={}"
        )
    urls = [url_template.format(i) for i in range(1,17+1)]
    # Post-Season
    url_template = (
            "http://scores.espn.go.com/nfl/scoreboard" +
            "?seasonYear={}&seasonType=3".format(year) +
            "&weekNumber={}"
        )
    urls += [url_template.format(i) for i in [1,2,3,5]]
    # testing snipit
    #random.shuffle(urls)
    #urls = [urls[0]]
    print "Downloading {} data".format(year)
    webpages = urltools.open_urls(urls, silence=False)
    print 'Downloading complete (found {} pages)'.format(len(webpages))
    
    # Parse all webpages into game-dicts
    #print 'Parsing webpages into Game objects'
    #worktime = time.time()
    #games = []
    #for url in webpages.keys():
    #    games.extend(
    #        parse_espn_weekly_results_page(webpages[url], url, return_filter)
    #    )
    ## END for
    #worktime = time.time() - worktime
    #print 'Parsing completed in {}:{:0.1f}'.format( int(worktime/60),
    #                                                worktime%60
    #                                              )
    
    # Parse all webpages into game-dicts
    print 'Parsing webpages into Game objects'
    worktime = time.time()
    results = []
    # Create worker pool and start all jobs
    if debug:
        numproc = 1
    else:
        numproc = mp.cpu_count()+1
    # END if
    worker_pool = mp.Pool(processes=numproc)
    for url in webpages.keys():
        results.append(
            worker_pool.apply_async(
                parse_espn_weekly_results_page,
                args=(webpages[url], url, return_filter, debug)
           )
        )
    # END for
    worker_pool.close()
    # Wait here for all work to complete
    worker_pool.join()
    games = []
    for resultobj in results:
        try:
            some_games = resultobj.get()
        except:
            continue
        # END try
        games.extend(some_games)
    # END for
    del results
    worktime = time.time() - worktime
    print 'Parsing completed in {}:{:0.0f}'.format( int(worktime/60),
                                                    worktime%60
                                                  )
    
    # Remove duplicate games
    # On ESPN.com if the postseason page has not been created for the year of
    # insterst then requests for postseason pages will actually return the
    # current week's page
    games.append(None)
    gmset = set()
    while games[0] is not None:
        gm = games.pop(0)
        if str(gm) not in gmset:
            games.append(gm)
            gmset.add(str(gm))
    # END while
    games.pop(0)
    
    # (Debug) Save games as csv file
    if debug or csv:
        print len(games)
        col_headers = sorted(games[0].__dict__.keys())
        # Remove private attributes from column headers
        col_headers.append(None)
        while col_headers[0] is not None:
            s = col_headers.pop(0)
            if s[0] == '_':
                continue
            col_headers.append(s)
        # END while
        col_headers.pop(0)
        # Write file
        save_name = 'schedule_{}.csv'.format(year)
        with open(save_name, 'w') as f:
            line = ''
            for header in col_headers:
                line += header + ','
            #line += 'edge weight (V2H),'
            f.write(line[:-1] + '\n')
            
            for gm in sorted(games, key=lambda g: g.date):
                line = ''
                for header in col_headers:
                    if header in gm.__dict__:
                        line += str(gm.__dict__[header]) + ','
                    else:
                        line += ','
                    # END if
                #line += '{0:0.3f},'.format(value_game(gm)['V2H'])
                #line += ','
                f.write(line[:-1] + '\n')
            # END for
        # END with
        print 'saved "{}"'.format(save_name)
    # END if
    
    return games
# END download_schedule_espn

def parse_espn_weekly_results_page(pgtxt, url, return_filter, debug=False):
    if debug: print ''
    print 'parsing '+url[30:]
    try:
        games = _parse_espn_weekly_results_page( pgtxt, url,
                                                      return_filter,
                                                      debug=debug
                                                    )
    except Exception as err:
        print 'skipping week page:\n    ' + url[30:]
        print '    ' + str(type(err)) + ': ' + str(err)
        games = []
    # END try
    return games
# END parse_espn_weekly_results_page

def _parse_espn_weekly_results_page( pgtxt, url, return_filter,
                                          debug=False
                                        ):
    # Turn the page into "soup"
    htmlsoup = BeautifulSoup(pgtxt)
    
    # Read off the season week number
    wk = htmlsoup.find(
        'a', class_=re.compile('selected'),
        href=re.compile(r'\?seasonYear.*?weekNumber')
    )
    # Post-season convertion
    postlkup = {
        'Wild Card': 18, 'Divisional': 19, 'Conference': 20,
        'Super Bowl': 21
    }
    if str(wk.string) in postlkup:
        wk = postlkup[str(wk.string)]
    else:
        wk = int(wk.string)
    # END if
    
    # Pre-collect some information in case game is upcoming
    # collect game days
    game_days = htmlsoup.find_all('h4', class_='games-date')
    game_day_containers = htmlsoup.find_all('div', class_='gameDay-Container')
    if debug:
        print 'found {} ({}) game days on week {}'.format(
            len(game_days), len(game_day_containers), wk
        )
    game_worklist = []
    for i in range(len(game_days)):
        # Example date string: Thursday, September 11, 2014
        # remove day of the week
        dstr = re.sub(r'^\w+, ', '', game_days[i].string)
        date = time.strptime(dstr, "%B %d, %Y")
        # Extract all html game containers
        game_chunks = game_day_containers[i].find_all(
            'div', id=re.compile(r'\d+-gameContainer')
        )
        if debug:
            print '{} games on {}'.format( len(game_chunks),
                                           game_days[i].string
                                         )
        for gamesoup in game_chunks:
            vstm = NFL.NFL.normalize_name(
                gamesoup.find(class_='team visitor').find('a').string
            )
            hmtm = NFL.NFL.normalize_name(
                gamesoup.find(class_='team home').find('a').string
            )
            game_worklist.append( (date, vstm, hmtm, gamesoup) )
        # END for
    # END for
    
    # Parse each game
    games = []
    if debug:
        print 'found {} games'.format(len(game_worklist))
    for date, vstm, hmtm, gamesoup in game_worklist:
        gm = {
            'wk': wk,
            'date': int(time.strftime('%Y%m%d', date)),
            'final': False,
            'vs': vstm, 'hm': hmtm,
        }
        # download the boxscore page for the game
        boxscore_link_mat = re.search(
            r'href="([^"]+)">Box Score<', str(gamesoup)
        )
        if not boxscore_link_mat:
            games.append(NFL.Game(gm))
            continue
        # END if
        
        boxscore_link = ( 'http://scores.espn.go.com' +
                          boxscore_link_mat.group(1)
                        )
        f = urllib2.urlopen(boxscore_link)
        boxsoup = BeautifulSoup( f.read() )
        f.close()
        
        try:
            if re.search('final-state', str(gamesoup['class'])):
                if return_filter == 'all' or return_filter == 'final':
                    gm['final'] = True
                else:
                    continue
                # END if
            elif return_filter == 'final':
                continue
            # END if
            if debug:
                print 'Final? ' + str(gm['final'])
            gm['final'] = gm['final']
            
            # Get the game date
            # Example date string: 6:25 PM ET, February 2, 2014
            datediv = boxsoup.find('div', class_='game-time-location')
            dstr = re.sub(r' ET,', '', datediv.p.string)
            # TODO: test this parsing format code
            try:
                date = time.strptime(dstr, "%I:%M %p %B %d, %Y")
            except ValueError as err:
                print datediv.p.string
                raise err
            # END try
            
            # Get the name of the visiting team
            t = gamesoup.find('div', class_='team visitor')
            if gm['final']:
                t_score = int(t.find('li', class_='final').string)
                gm["vssc"] = t_score
            
            # Get the name of the home team
            t = gamesoup.find('div', class_='team home')
            if gm['final']:
                t_score = int(t.find('li', class_='final').string)
                gm["hmsc"] = t_score
            
            if gm['final']:
                div_stattable = boxsoup.find_all(
                    'div', class_=re.compile(r'.*?mod-open-gamepack')
                )[1]
                # Passing Yards
                vs_pass, hm_pass = re.search(r'''
                    Passing \s*?< .*?
                    <td>\s* (\d+) \s*</td> .*?
                    <td>\s* (\d+) \s*</td>
                    ''', str(div_stattable), re.VERBOSE
                ).groups()
                gm["vs_pass"] = int(vs_pass)
                gm["hm_pass"] = int(hm_pass)
                if debug:
                    print 'found passing yards'
                # Passing Completions and Attempts
                try:
                    passes = re.search(r'''
                        Comp[\s-]*Att .*?
                        <td>\s* (\d+)-(\d+) \s*</td> .*?
                        <td>\s* (\d+)-(\d+) \s*</td>
                        ''', str(div_stattable), re.VERBOSE
                    ).groups()
                except AttributeError as err:
                    print str(div_stattable)
                    raise err
                # END try
                passes = list(passes)
                for i in range(len(passes)):
                    passes[i] = int(passes[i])
                gm['vs_cmp'] = passes[0]
                gm['vs_att'] = passes[1]
                gm['hm_cmp'] = passes[2]
                gm['hm_att'] = passes[3]
                if debug:
                    print 'found passing att & cmp counts'
                # Sack counts and yards
                sacks = re.search(r'''
                    Sacks\s*-\s*Yards\s Lost .*?
                    <td>\s* (\d+)-(\d+) \s*</td> .*?
                    <td>\s* (\d+)-(\d+) \s*</td>
                    ''', str(div_stattable), re.VERBOSE
                ).groups()
                gm["vs_sk"] = int(sacks[0])
                gm["vs_skyd"] = int(sacks[1])
                gm["hm_sk"] = int(sacks[2])
                gm["hm_skyd"] = int(sacks[3])
                if debug:
                    print 'found sack counts and yards'
                # Rushing Yards
                vs_rush, hm_rush = re.search(r'''
                    Rushing \s*?< .*?
                    <td>\s* (\d+) \s*</td> .*?
                    <td>\s* (\d+) \s*</td>
                    ''', str(div_stattable), re.VERBOSE
                ).groups()
                gm["vs_rush"] = int(vs_rush)
                gm["hm_rush"] = int(hm_rush)
                if debug:
                    print 'found rushing yards'
                # Rushing Attempts
                try:
                    vs_runs, hm_runs = re.search(r'''
                        Rushing[\s-]*Attempts .*?
                        <td>\s* (\d+) \s*</td> .*?
                        <td>\s* (\d+) \s*</td>
                        ''', str(div_stattable), re.VERBOSE
                    ).groups()
                except AttributeError as err:
                    print str(div_stattable)
                    raise err
                # END try
                gm["vs_runs"] = int(vs_runs)
                gm["hm_runs"] = int(hm_runs)
                if debug:
                    print 'found rushing attempt counts'
                
                vsdrives, hmdrives = re.search(r'''
                    Total \s Drives .*?
                    <td>\s* (\d+) \s*</td> .*?
                    <td>\s* (\d+) \s*</td>
                    ''', str(div_stattable), re.VERBOSE
                ).groups()
                gm["vsdrives"] = int(vsdrives)
                gm["hmdrives"] = int(hmdrives)
                if debug:
                    print 'found drive counts'
                
                vsint, hmint = re.search(r'''
                    Interceptions \s thrown .*?
                    <td>\s* (\d+) \s*</td> .*?
                    <td>\s* (\d+) \s*</td>
                    ''', str(div_stattable), re.VERBOSE
                ).groups()
                gm["vs_int"] = int(vsint)
                gm["hm_int"] = int(hmint)
                if debug:
                    print 'found int counts'
                
                vsto, hmto = re.search(r'''
                    Turnovers .*?
                    <td>\s* (\d+) \s*</td> .*?
                    <td>\s* (\d+) \s*</td>
                    ''', str(div_stattable), re.VERBOSE
                ).groups()
                gm["vsto"] = int(vsto)
                gm["hmto"] = int(hmto)
                if debug:
                    print 'found TO counts'
                
                # Defensive / Special Teams TDs
                try:
                    vsdefTD, hmdefTD = re.search(r'''
                        Defensive. /. Special. Teams. TDs .*?
                        <td>\s* (\d+) \s*</td> .*?
                        <td>\s* (\d+) \s*</td>
                        ''', str(div_stattable), re.VERBOSE
                    ).groups()
                    gm["vsdefTD"] = int(vsdefTD)
                    gm["hmdefTD"] = int(hmdefTD)
                    if debug:
                        print 'found def/special TD counts'
                except AttributeError as err:
                    # Some earlier years did not record this stat
                    pass
                # END try
            # stat table is in:
            # <div class="mod-container mod-open mod-open-gamepack">
            # Turnovers found in
            # <tr class="odd" align="right" style="font-weight: bold; font-size: 11px;">
            #  <td class="bi" style="text-align:left">Turnovers</td>
            #  <td>3</td>
            #  <td>1</td>
            # </tr>
        except IndexError as err:
            printstr = 'error'
            pprint(gm)
            raise err
        except ValueError:
            printstr = 'Error encountered parsing a game\n'
            printstr += 'gm = ' + repr(gm)
            print printstr
            #quit()
        except Exception as err:
            print sorted(gm.keys())
            raise err
        # END try
        games.append(NFL.Game(gm))
    # END for
    
    return games
# END _parse_espn_weekly_results_page

#===============================================================================
def download_ANS_rankings(url):
    pg = urllib2.urlopen(url)
    pgtxt = pg.read()
    pg.close()
    
    rows = re.findall(
        r'<tr class="myClass">.+?logocell.+?</tr>', pgtxt, re.DOTALL
    )
    
    gwp_lkup = {}
    for r in rows:
        try:
            name = re.search(r'>\s*([A-Z]{2,3})\s*<', r).group(1)
        except:
            print 'Error finding name for row:'
            print '\t"{}"'.format(r)
            continue
        # END try
        
        try:
            num_cols = re.findall(r'>\s*([.\d]+)\s*<', r)
            gwp = float(num_cols[2])
        except:
            print 'Error finding gwp for row:'
            print '\t"{}"'.format(r)
            continue
        # END try
        
        gwp_lkup[NFL.NFL.normalize_name(name)] = gwp
    # END for
    
    return gwp_lkup
# END download_ANS_rankings

#===============================================================================
if __name__ == '__main__':
    main(*sys.argv[1:])
# END if