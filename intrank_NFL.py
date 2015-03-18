#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''NFL Interception Modeling Script
    
    This section should be a summary of important infomation to help the editor
    understand the purpose and/or operation of the included code.
    
    List of classes: -none-
    List of functions:
        main
'''

# built-in modules
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
    # year as 'yyyy'
    year = NFL.as_seasonyear( time.strftime('%Y%m') )
    #year = 2013
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
    print 'loading {}'.format(pickle_fn)
    if pickle_fn in os.listdir('.'):
        f = open(pickle_fn, 'rb')
        games = pickle.load(f)
        f.close()
    else:
        print 'no pickle found'
        quit()
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
        else:
            fut_games.append(g)
        # END if
    # END while
    games.pop(0)
    print '{} final games'.format(len(games))
    print '{} future games'.format(len(fut_games))
    
    # Attemp MCMC sampling of BT model for interception rates
    print 'Begin Monte Carlo'
    pickle_fn = 'NFL_int_mcmc_results_{}.pypickle'.format(year)
    results = rank_NFL.bt_int_mcmc_solve(games)
    f = open(pickle_fn, 'wb')
    pickle.dump(results, f)
    f.close()
    mcmc_analyzer.int_mcmc_analyzer(pickle_fn)
# END main

#===============================================================================
if __name__ == '__main__':
    main()
# END if
