#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''NFL Regular Update Script
    
    This section should be a summary of important infomation to help the editor
    understand the purpose and/or operation of the included code.
    
    List of classes: -none-
    List of functions:
        main
'''

# built-in modules
import sys
import time
from math import *
import re
import os
from pprint import pprint
import pickle

# third party modules
from scipy.special import erf, erfinv

# personal modules
import rank_NFL
from NFL_class import *
import collectors_NFL_v2 as clct
import mcmc_analyzer
try:
    import ampCSV
except:
    sys.path.append('../../ampPy/')
    print '*** added "../../ampPy/" to sys.path ***'
    
    import ampCSV
# END try

#===============================================================================
def main(*args):
    force_download = False
    debug = False
    if '-f' in args:
        force_download = True
    if '-d' in args:
        debug = True
    
    # year as 'yyyy'
    year = NFL.as_seasonyear( time.strftime('%Y%m') )
    #year = 2012
    if year is None: year = int(time.strftime('%Y')) - 1
    print 'Running update script for {} season'.format(year)
    
    # Get all final scores for this season
    tms = NFL.team_names
    tms = [Team(nm) for nm in tms]
    tm_lkup = {}
    for t in tms:
        tm_lkup[str(t)] = t
    print 'No. teams = {}'.format(len(tms))
    
    # Load in the season schedule
    pickle_fn = 'NFL_games_{}.pypickle'.format(year)
    if pickle_fn in os.listdir('.') and not force_download:
        print 'loading pickle'
        f = open(pickle_fn, 'rb')
        games = pickle.load(f)
        f.close()
    else:
        print 'downloading'
        games = clct.download_schedule_espn(year, 'all', debug=debug)
        print 'saving pickle'
        # Save results for later
        pickle_fn = 'NFL_games_{}.pypickle'.format(year)
        f = open(pickle_fn, 'wb')
        pickle.dump(games, f)
        f.close()
    # END if
    
    # Sort games into finish and future, and make Team objects consistent
    fut_games = []
    games.append(None)
    while games[0] is not None:
        g = games.pop(0)
        g.hm = tm_lkup[str(g.hm)]
        g.vs = tm_lkup[str(g.vs)]
        if g.final:
            games.append(g)
            #print '{:l}'.format(g)
        else:
            fut_games.append(g)
        # END if
    # END while
    games.pop(0)
    print '{} final games'.format(len(games))
    print '{} future games'.format(len(fut_games))
    
    # Create fake game results against fake team to establish prior
    real_games = list(games)
    f = open('2014_preseason_rankings.csv')
    avgtm = Team('Average Joes')
    lines = re.split('\s+', f.read())
    for ln in lines:
        nm, lg = re.search(r'^([A-Z]{3}),(.+)', ln).groups()
        tm = tm_lkup[NFL.normalize_name(nm)]
        #tm = NFL.normalize_name(nm)
        p = exp(float(lg)) / (exp(float(lg)) + 1)
        pnts = 11.87*sqrt(2.0)*erfinv(2.0*p - 1.0)
        #print '    {:.<24s} {:>.0f}'.format(str(tm), pnts)
        gmdict = {
            'final': pnts, 'date': '0901', 'wk': 0, 'hosted': False,
            'vs': tm, 'vssc': pnts, 'vsto': 0, 'vsdrives': 1, 'vsdefTD': 0,
            'hm': avgtm, 'hmsc': 0, 'hmto': 0, 'hmdrives': 1, 'hmdefTD': 0,
        }
        games.append(Game(gmdict))
    # END for
    f.close()
    
    # Count up "Significant Wins" Record
    for g in games:
        if type(g) == dict:
            pprint(g)
            quit()
        try:
            g.hm.__bigwins
        except AttributeError:
            g.hm.__bigwins = WLTRecord()
        # END try
        try:
            g.vs.__bigwins
        except AttributeError:
            g.vs.__bigwins = WLTRecord()
        # END try
        x = g.value()
        if x < (1-0.65):
            # home loss
            g.hm.__bigwins.addL()
            g.vs.__bigwins.addW()
        elif 0.65 < x:
            # home win
            g.hm.__bigwins.addW()
            g.vs.__bigwins.addL()
        else:
            # tie
            g.hm.__bigwins.addT()
            g.vs.__bigwins.addT()
        # END if
    # END for
    
    # Rank teams using the random walker method
    print 'ranking...'
    # Calculate Random-Walker rankings
    tm_rw_vals = rank_NFL.rw_rank(games)
    tm_rw_vals = rank_NFL.normalize_ranks(tm_rw_vals)
    tm_rw_vals_orig = dict(tm_rw_vals)
    ordered_tms = sorted(
        tm_rw_vals.keys(), key=lambda t: tm_rw_vals[t], reverse=True
    )
    rank = 1
    for tm in ordered_tms:
        tm.v = tm_rw_vals[tm]
        tm_rw_vals[tm] = (rank, tm_rw_vals[tm])
        rank += 1
    # END for
    
    # Calculate Bradley-Terry rankings
    try:
        tm_bt_vals = rank_NFL.bt_rank(games, adjusted=True)
    except ZeroDivisionError:
        print 'Warning: Continuing with RW ranks substitued for BT ranks!'
        tm_bt_vals = dict(tm_rw_vals_orig)
    # END try
    tm_bt_vals = rank_NFL.normalize_ranks(tm_bt_vals)
    ordered_tms = sorted(
        tm_bt_vals.keys(), key=lambda t: tm_bt_vals[t], reverse=True
    )
    rank = 1
    for tm in ordered_tms:
        tm_bt_vals[tm] = (rank, tm_bt_vals[tm])
        rank += 1
    # END for
    
    # Calculate remaining wins and strength of schedule
    fut_wins = {}
    for t in tms:
        fut_wins[t] = 0.0
    # END for
    for g in fut_games:
        p_hm_win = g.predict()
        fut_wins[g.hm] += p_hm_win
        fut_wins[g.vs] += 1 - p_hm_win
    # END for
    
    csv_data = [
        [ 'Division', 'Team','RW Rank', 'RW Strength', '(logRHO)',
          'Future ExWins',
          'BT Rank', 'BT Strength', '(logRHO)', 'BgW', 'BgL', 'BgT', 'Bg%'
        ]
    ]
    f = open('java_insert.java', 'w')
    for tm in sorted(tm_rw_vals.keys()):
        if tm == 'AVG':
            continue
        row = [
            tm.group, tm,
            tm_rw_vals[tm][0], '{0:0.2f}'.format(tm_rw_vals[tm][1]),
            '{0:0.2f}'.format( log(tm_rw_vals[tm][1], sqrt(1.27)) ),
            '{:0.1f}'.format( fut_wins[tm] ),
            tm_bt_vals[tm][0], '{0:0.2f}'.format(tm_bt_vals[tm][1]),
            '{0:0.2f}'.format( log(tm_bt_vals[tm][1], sqrt(1.27)) ),
            tm.__bigwins.W, tm.__bigwins.L, tm.__bigwins.T,
            '{:0.3f}'.format(tm.__bigwins.Wpct)
        ]
        csv_data.append(row)
        tm_w = tm_rw_vals[tm][1]
        f.write(
            'gwp_lkup.put("{:l}", {:0.3f});\n'.format(tm, tm_w/(tm_w+1))
        )
    # END for
    f.close()
    ampCSV.saveCSVHTML(csv_data, 'NFL_rankings_{}.csv.html'.format(year))
    
    # Save some schedule details
    rank_NFL.write_schedule_details(games, tm_rw_vals_orig, year)
    print 'successfully wrote schedule details'
# END main

#===============================================================================
if __name__ == '__main__':
    main(*sys.argv[1:])
# END if
