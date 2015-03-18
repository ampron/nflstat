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
from pprint import pprint

# third-party modules
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.special import erf, erfinv
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

# custom modules
import rank_NFL
from NFL_class import NFL, Game, Team
import collectors_NFL as coll

#===============================================================================
def main():
    tms = NFL.team_names
    tms = [Team(nm) for nm in tms]
    tm_lkup = {}
    for t in tms:
        tm_lkup[str(t)] = t
    
    # Load in the season schedule
    # TODO: fix downloading function to use new objects
    all_gms = coll.download_schedule_espn()
    for g in all_gms:
        g.hm = tm_lkup[str(g.hm)]
        g.vs = tm_lkup[str(g.vs)]
    #all_gms = [Game(gm) for gm in all_gms]
    all_gms.sort(key=lambda d: d.wk)
    print 'len(all_gms) = ' + str(len(all_gms))
    
    i_sn = 0
    win_dist = []
    sig = 16.0
    real_betas = []
    calc_betas = []
    
    while True:
        # Set up the team strength distribution
        #tm_strength = rand_tm_strengths(tms)
        for t in tms:
            t.set_rand_strength()
        
        # Simulate ONE season
        sim_gms, sim_wins = simulate_season(all_gms)
        # END for
        
        # Determine playoff seedings
        # TODO: make function that will take season results and return a playoff
        #       bracket object
        
        # Run ranking algorithm on simulated season results
        #rank_ok = True
        #for tm in sim_wins:
        #	if sim_wins[tm] == 0 or sim_wins[tm] == 16:
        #		rank_ok = False
        #		i_sn -= 1
        ## END for
        #if rank_ok:
        #	tm_vals = rank_NFL.bt_rank(sim_gms)
        #	#X = []
        #	#Y = []
        #	for tm in tm_vals:
        #		real_betas.append(np.log(tm_strength[tm]))
        #		calc_betas.append(np.log(tm_vals[tm]))
        #	#plt.loglog(X,Y,'.')
        #	#plt.show()
        #	#plt.close()
        ## END if
        
        win_dist.extend(sim_wins.values())
        sig_ = np.std(win_dist)
        i_sn += 1
        if i_sn >= 1000: # or np.abs(sig - sig_) < 0.0005:
            print 'season ' + str(i_sn)
            print 'mean = {0:0.2f} wins'.format(np.mean(win_dist))
            print 'sigma = {0:0.5f} ± {1:0.2e}'.format(
                sig_, np.abs(sig - sig_)
            )
            print '    sigma/bino = {0:0.3f}'.format(sig_/2.0)
            print 'skewness = {0:0.5e}'.format(skew(win_dist))
            krt = kurtosis(win_dist)
            print 'kurtosis = {0:0.5e}'.format(krt)
            print '    kurt/bino = {0:+0.3f}'.format(krt-(-1/8.0)/(-1/8.0))
            plt.hist(win_dist, bins=17)
            plt.show()
            plt.close()
            break
        else:
            sig = sig_
        # END if
    # END while
    
    #apparent_hm_edge = float(apparent_hm_edge)/len(sim_gms)
    #print "Home edge for this season: {0:0.3f}".format(apparent_hm_edge)
    
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
    
    if len(calc_betas) > 0:
        f = open('rankings of simulated seasons.csv', 'w')
        f.write('real-beta,calc-beta\n')
        for i in range(len(real_betas)):
            f.write('{0:0.5f},{1:0.5f}\n'.format(real_betas[i], calc_betas[i]))
        f.close()
        plt.plot(real_betas, calc_betas, '.')
        plt.show()
        plt.close()
    
    # Print the divisional standings
    tms.sort(key=lambda t: sim_wins[t], reverse=True)
    tms.sort(key=lambda t: t.group)
    teams_ranked = sorted(tms, key=lambda t: t.v)
    div = ''
    for tm in tms:
        if div != tm.group:
            div = tm.group
            print ''
            print div
            print 22*'-'
        # END if
        try:
            print u'{0:22s}: {1:2d}-{2:2d} ({4:2d}, π={3:0.3f})'.format(
                str(tm), sim_wins[tm], 16-sim_wins[tm],
                tm.v, 32-teams_ranked.index(tm)
            )
        except:
            print u'{0:22s}: {1:2d}-{2:2d} (π={3:0.3f})'.format(
                str(tm), sim_wins[tm], 16-sim_wins[tm], tm.v
            )
        # END try
    # END for
    print ''
    
    # Print playoff seedings
    playoff_pool = [t for t in sorted(tms, key=lambda t: sim_wins[t])]
    pprint(playoff_pool[:12])
    # Get division winners
    #for div in NFL.divisions:
    #    print '{} champ'
    #    for t in tms:
    #        champ = None
    #        if t.group == div:
    #            if champ is None:
    #                champ = t
    #            else if 
# END main

#===============================================================================
def simulate_season(all_gms):
    sim_gms = []
    sim_wins = {}
    for gm in all_gms:
        if type(gm) is dict:
            gm = Game(gm)
        if gm.hm not in sim_wins:
            sim_wins[gm.hm] = 0
        if gm.vs not in sim_wins:
            sim_wins[gm.vs] = 0
        
        x = random.random()
        p_hm = gm.predict()
        # Translate win probability into expected point difference (pnts)
        pnts = 11.87*math.sqrt(2.0)*erfinv(2.0*p_hm - 1.0)
        pnts = int( np.random.normal(pnts, 11.87, 1)[0] )
        # Do not allow ties, if simulated point diff is 0 then flip a coin
        if pnts == 0: pnts = (-1)**random.randint(0,1)
        
        # Record results
        new_gm = {}
        if pnts > 0:
            new_gm['vssc'] = 0
            new_gm['hmsc'] = pnts
            sim_wins[gm.hm] += 1
        else:
            new_gm['vssc'] = -pnts
            new_gm['hmsc'] = 0
            sim_wins[gm.vs] += 1
        # END if
        new_gm['wk'] = gm.wk
        new_gm['date'] = gm.date
        new_gm['hm'] = gm.hm
        new_gm['vs'] = gm.vs
        new_gm['hmto'] = 0
        new_gm['vsto'] = 0
        new_gm = Game(new_gm)
        
        sim_gms.append(new_gm)
    # END for
    
    return sim_gms, sim_wins
# END simulate_season

#===============================================================================
def rand_tm_strengths(tms):
    # Set up the team strength distribution
    tm_strength = {}
    i = 0.0
    random.shuffle(tms)
    # original
    #all_betas = np.random.normal(loc=0.0, scale=0.495904447431, size=32)
    # adjusted
    all_betas = np.random.normal(loc=0.0, scale=0.75, size=32)
    all_betas = sorted(list(all_betas))
    for t in tms:
        beta = all_betas.pop(0)
        tm_strength[t] = math.exp(beta)
        t.v = math.exp(beta)
        i += 1
    # END for
    
    return tm_strength
# END rand_tm_stengths

#===============================================================================
def predictGm(v_hm, v_vs, pnt_adj=0.0):
    '''
    (P[hm win], finalScore) = predictGm(v_hm, v_vs, ('H'|'N'), ('M'|'W'))
    '''
    
    # Standard deviation in score
    sig = 11.87
    sigSqrt2 = sig*math.sqrt(2.0)
    
    p_hm = v_hm/(v_hm + v_vs)
    # final score difference at a neutral location
    # score difference = home score - visitor score
    pnts = sigSqrt2*erfinv(2.0*p_hm - 1.0)
    if pnt_adj != 0:
        # final score with home edge
        pnts += pnt_adj
        p_hm = 1.0 - ( 0.5*(1.0 + erf( -pnts / sigSqrt2 )) )
    
    return (p_hm, pnts)
# END predictGm

#===============================================================================
def logit(p): return np.log(p/(1.0-p))
def invlogit(x): return 1.0 / (1.0 + np.exp(-x))

#===============================================================================
if __name__ == '__main__':
    main()
# END if