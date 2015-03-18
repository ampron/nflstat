#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''NFL Simulator
    
    List of classes: -none-
    List of functions:
        main
'''

# built-in modules
import random
import math
import pickle
from pprint import pprint

# third-party modules
import numpy as np

# custom modules
import NFL_class_v2 as NFLmod

#===============================================================================
def main():
    sn = NFLmod.Season()
    
    # Load in the season schedule
    f = open('NFL_games_2014.pypickle', 'rb')
    games = pickle.load(f)
    f.close()
    
    # Sort games into finish and future, and make Team objects consistent
    fut_games = []
    for _ in range(len(games)):
        g = games.pop(0)
        g.hm = sn.lkup_tm(str(g.hm))
        g.vs = sn.lkup_tm(str(g.vs))
        if g.final:
            games.append(g)
        else:
            fut_games.append(g)
        # END if
    # END for
    print '{} final games'.format(len(games))
    print '{} future games'.format(len(fut_games))
    
    # count up current wins
    curr_records = {tm: NFLmod.WLTRecord() for tm in sn.teams}
    for gm in games:
        if not gm.tie:
            curr_records[gm.winner].addW()
            curr_records[gm.loser].addL()
        else:
            curr_records[gm.hm].addT()
            curr_records[gm.vs].addT()
        # END if
    # END for
    # Print the divisional standings
    sn.teams.sort(key=lambda t: curr_records[t], reverse=True)
    sn.teams.sort(key=lambda t: t.group)
    div = ''
    for tm in sn.teams:
        if div != tm.group:
            div = tm.group
            print ''
            print div
            print 22*'-'
        # END if
        print u'{0:22s}: {1}'.format(str(tm), curr_records[tm])
    # END for
    print ''
    
    # Set-up team strengths
    S = [
        ('DEN', 0.71),
        ('MIA', 0.68),
        ('GB', 0.64),
        ('IND', 0.59),
        ('SEA', 0.59),
        ('NE', 0.58),
        ('SF', 0.58),
        ('KC', 0.57),
        ('DAL', 0.56),
        ('CLE', 0.56),
        ('BUF', 0.55),
        ('NO', 0.54),
        ('BAL', 0.54),
        ('DET', 0.54),
        ('CIN', 0.53),
        ('PHI', 0.52),
        ('PIT', 0.51),
        ('WAS', 0.49),
        ('ARI', 0.49),
        ('TEN', 0.48),
        ('HOU', 0.47),
        ('SD', 0.46),
        ('CHI', 0.46),
        ('CAR', 0.45),
        ('NYG', 0.41),
        ('MIN', 0.38),
        ('NYJ', 0.38),
        ('JAC', 0.38),
        ('STL', 0.37),
        ('ATL', 0.34),
        ('OAK', 0.33),
        ('TB', 0.32),
    ]
    for nm, p in S: (sn.lkup_tm(nm)).__w = p/(1-p)
    
    # Simulate the remainder of the season
    N_sim = 3000
    all_outcomes = []
    for i_sn in range(N_sim):
        outcome = []
        for gm in fut_games:
            p_hm = 1.27 * gm.hm.__w / gm.vs.__w
            if random.random() < p_hm:
                # Home team wins
                outcome.append(
                    NFLmod.Game(gm.vs, gm.hm, gm.date, vssc=0, hmsc=1)
                )
            else:
                # Visiting team wins
                outcome.append(
                    NFLmod.Game(gm.vs, gm.hm, gm.date, vssc=1, hmsc=0)
                )
            # END if
        # END for
        all_outcomes.append(outcome)
    # END for
    print 'done simulating seasons'
    
    # Save simulated results in a csv/html file
    #f_txt = 'Date,Week,Visitor,Score,Home,Score' + r'\\n'
    #for gm in sim_gms:
    #	f_txt += '{0[date]},{0[wk]},{0[vs]},{0[vssc]},{0[hm]},{0[hmsc]}'.format(gm)
    #	f_txt += r'\\n'
    # END for
    #f = open('../../ampPy/example_chimera_table.csv.html')
    #template = f.read()
    #f.close()
    #f_txt = re.sub(r"(?<=var csv_data = ')[^']*?(?=')", f_txt, template)
    #f = open('NFL_simulated_season_results.csv.html', 'w')
    #f.write(f_txt)
    #f.close()
# END main

#===============================================================================
def logit(p): return np.log(p/(1.0-p))
def invlogit(x): return 1.0 / (1.0 + np.exp(-x))

#===============================================================================
if __name__ == '__main__': main()