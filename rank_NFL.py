'''NFL Ranking Module
    Author: Alex Pronschinske
'''

# built-in modules
import re
import random
from pprint import pprint

# third party modules
import numpy as np
from math import sqrt, exp, pi, log
from scipy.special import erf, erfinv, binom
from numpy.linalg import norm

# personal modules
try:
    from NFL_class import *
except:
    import sys
    sys.path.append('../../../ampPy/')
    
    from NFL_class import *
    
    print '*** added "../../../ampPy/" to sys.path ***'
# END try

#===============================================================================
def rank_process(games, debug=False):
    '''This funciton is depreciated!  Use "rw_rank" instead.'''
    print (
        'Warning: "rank_process" is a depreciated function, use ' +
        'rw_rank instead'
    )
    return rw_rank(games, debug)
# END rank_process

def rw_rank(games, debug=False):
    '''Random Walker Rankings for NFL
    
    Args:
        games (dict):
        debug = False: Debugging switch
    Returns:
        (dict) {"team name": [eq_val, eq_pnts], ...}
    '''
    
    # make team legend
    team_lkup = {}
    n_tms = 0
    for gm in games:
        if gm.hm not in team_lkup:
            team_lkup[gm.hm] = n_tms
            team_lkup[n_tms] = gm.hm
            n_tms += 1
        if gm.vs not in team_lkup:
            team_lkup[gm.vs] = n_tms
            team_lkup[n_tms] = gm.vs
            n_tms += 1
    # END for
    
    if debug: print 'Number of Vertices (Teams) on Graph = {0}'.format(n_tms)
    
    # Set up adjacency matrix
    D = [[0.0 for j in range(n_tms)] for i in range(n_tms)]
    for gm in games:
        i_hm = team_lkup[gm.hm]
        i_vs = team_lkup[gm.vs]
        try:
            p = gm.value()
        except ValueError:
            continue
        except AttributeError as err:
            print '{:l}'.format(gm)
            raise err
        # END try
        
        D[i_hm][i_hm] -= 1-p
        D[i_hm][i_vs] += p
        
        D[i_vs][i_vs] -= p
        D[i_vs][i_hm] += 1-p
    # END for
    
    # Solve for equilibrium values by using row elimination on D matrix
    Dreduc = rowElim(D)
    V = [0.0 for i in range(n_tms)]
    A = 1.0
    nGroups = 0
    for i in range(n_tms)[::-1]:
        for j in range(i+1,n_tms):
            V[i] += -Dreduc[i][j]*V[j]
        if V[i] == 0.0:
            V[i] = 1.0
            nGroups += 1
        # END if
        A += V[i]
    # END for
    
    # Normalize team values
    #A /= n_tms
    tm_vals = {}
    for i in range(n_tms):
        V[i] /= A
        tm_vals[team_lkup[i]] = V[i]
    
    # Status alert
    if nGroups == 1:
        if debug: print 'Graph Connected'
    else:
        if debug: print 'Graph Unconnected, with {0} subgroups'.format(nGroups)
    
    if debug:
        # Check equilibrium values
        Dmatrix = np.array(D)
        dV = np.dot(Dmatrix,np.array(V))
        dVnorm = norm(dV)
        print '|dV| = {0:0.8e}'.format(dVnorm)
    # END if
    
    return tm_vals
# END rw_rank

#===============================================================================
def bt_rank(games, adjusted=True, debug=False):
    '''Bradley-Terry Rankings for NFL
    
    Args:
        games (dict):
        debug = False: Debugging switch
    Returns:
        (dict) {"team name": tm_strength, ...}
    '''
    
    # Home field advantage
    p_hfa = 0.56
    g_hfa = p_hfa / (1-p_hfa)
    
    # Make team legend and wins dict
    tm_vals = {}
    tm_wins = {}
    team_lkup = {}
    n_tms = 0
    for gm in games:
        if gm.hm not in team_lkup:
            tm_vals[gm.hm] = 1.0
            tm_wins[gm.hm] = 0
            team_lkup[gm.hm] = n_tms
            team_lkup[n_tms] = gm.hm
            n_tms += 1
        # END if
        if gm.vs not in team_lkup:
            tm_vals[gm.vs] = 1.0
            tm_wins[gm.vs] = 0
            team_lkup[gm.vs] = n_tms
            team_lkup[n_tms] = gm.vs
            n_tms += 1
        # END if
        
        cutoff = 0
        if adjusted:
            pnt_diff = (
                gm.hmsc - 4.579*gm.vsto - 1.9272
                - ( gm.vssc - 4.579*gm.hmto )
            )
            cutoff = 11
        else:
            pnt_diff = gm.hmsc - gm.vssc
        # END if
        
        if cutoff < pnt_diff:
            tm_wins[gm.hm] += 1
        elif pnt_diff < -1*cutoff:
            tm_wins[gm.vs] += 1
        else:
            tm_wins[gm.hm] += 0.5
            tm_wins[gm.vs] += 0.5
        # END if
    # END for
    
    if debug:
        # Print out adjusted records
        print 'Team wins:'
        for tm in sorted( tm_wins.keys(), key=lambda t: tm_wins[t],
                          reverse=True ):
            print '    {0:22s} | {1:0.1f}'.format(str(tm), tm_wins[tm])
    # END if
    
    # Set up adjacency matrix
    D = [[0 for j in range(n_tms)] for i in range(n_tms)]
    for gm in games:
        i_hm = team_lkup[gm.hm]
        i_vs = team_lkup[gm.vs]
        
        D[i_hm][i_vs] += 1
    # END for
    
    # Solve for team strength values
    dv = 1.0
    i = 0
    while dv > 0.0 and i < 12**4:
        dv = 0.0
        for tm in tm_vals:
            i_tm = team_lkup[tm]
            # Calculate the demoninator D
            deno = 0.0
            for j in range(n_tms):
                # home games
                deno += (
                    g_hfa * D[i_tm][j]
                    / (g_hfa*tm_vals[tm] + tm_vals[team_lkup[j]])
                )
                # road games
                deno += (
                    g_hfa * D[j][i_tm]
                    / (g_hfa*tm_vals[team_lkup[j]] + tm_vals[tm])
                )
            # END for
            
            tm_v_ = tm_vals[tm]
            tm_vals[tm] = tm_wins[tm] / deno
            dv += (tm_v_ - tm_vals[tm])**2
        # END for
        
        dv = sqrt(dv)
        i += 1
    # END while
    
    #print 'Exited on round {0}, with |dv| = {1:0.2e}'.format(i, dv)
    # END for
    
    # Normalize team values
    A = np.median(tm_vals.values())
    for tm in tm_vals: tm_vals[tm] /= A
    
    return tm_vals
# END bt_rank

#===============================================================================
def bt_int_rank(games, debug=False):
    '''Bradley-Terry Rankings for NFL
    
    Args:
        games (dict):
        debug = False: Debugging switch
    Returns:
        (dict) {"team name": tm_strength, ...}
    '''
    
    # Home field advantage
    o_hfa = 1.0#0.976
    
    # Make team legend and wins dict
    # Set up list of teams
    teams = set()
    for gm in games:
        if str(gm.hm)+'_off' not in teams:
            teams.add(str(gm.hm)+'_off')
            teams.add(str(gm.hm)+'_def')
        if str(gm.vs)+'_off' not in teams:
            teams.add(str(gm.vs)+'_off')
            teams.add(str(gm.vs)+'_def')
    # END for
    teams = sorted(list(teams))
    #teams += ['Average Joes Off', 'Average Joes Def']
    i_lkup = {}
    for i in range(len(teams)):
        i_lkup[teams[i]] = i
    # END for
    #pprint(teams)
    print 'len(bt_mcmc_solve.teams) = {}'.format(len(teams))
    
    # make K and N matricies
    Ntms = len(teams)
    N = [[0 for j in range(Ntms)] for i in range(Ntms)]
    Y = np.zeros(len(teams))
    R = np.zeros(len(teams))
    for gm in games:
        N[i_lkup[str(gm.hm)+'_off']][i_lkup[str(gm.vs)+'_def']] += gm.hm_att
        N[i_lkup[str(gm.hm)+'_def']][i_lkup[str(gm.vs)+'_off']] += gm.vs_att
        Y[i_lkup[str(gm.hm)+'_off']] += gm.hm_att - gm.hm_int
        Y[i_lkup[str(gm.hm)+'_def']] += gm.vs_int
        Y[i_lkup[str(gm.vs)+'_off']] += gm.vs_att - gm.vs_int
        Y[i_lkup[str(gm.vs)+'_def']] += gm.hm_int
        R[i_lkup[str(gm.hm)+'_off']] += gm.hm_att
        R[i_lkup[str(gm.hm)+'_def']] += gm.vs_att
        R[i_lkup[str(gm.vs)+'_off']] += gm.vs_att
        R[i_lkup[str(gm.vs)+'_def']] += gm.hm_att
    # END for
    
    # Setup prior
    #for i in range(0, len(teams)-2, 2):
    #    N[i][i_lkup['Average Joes Def']] += 1
    #    N[i+1][i_lkup['Average Joes Off']] += 1
    #    N[i_lkup['Average Joes Def']][i] += 1
    #    N[i_lkup['Average Joes Off']][i+1] += 1
    #    Y[i_lkup['Average Joes Off']] += 0.0288043892403 + 0.0302514060954
    #    Y[i_lkup['Average Joes Def']] += 0.0288043892403 + 0.0302514060954
    #    Y[i] += 0.0288043892403 + 0.0302514060954
    #    Y[i+1] += 0.0288043892403 + 0.0302514060954
    #    V[i_lkup['Average Joes Off']] += 2
    #    V[i_lkup['Average Joes Def']] += 2
    # END for
    for i in range(0, len(teams), 2):
        R[i] = float(Y[i]) / R[i]
        R[i+1] = float(R[i+1] - Y[i+1]) / R[i+1]
        #print '{:.<25s}  {:0.5f} ({:02d} / {:03d})'.format(
        #    teams[i], R[i], int(Y[i]), int(n)
        #)
    # END for
    
    # Solve for team strength values
    dv = 1.0
    n = 0
    V = np.array([32.9**(1-((i+1)%2)) for i in range(len(teams))])
    while 0.0 < dv and n < 12**4:
        dv = 0.0
        for i in range(0, len(teams)):
            # Calculate the demoninator D
            deno = 0.0
            for j in range(0, len(teams)):
                # home games
                deno += o_hfa * N[i][j] / (o_hfa*V[i] + V[j])
                # road games
                deno += N[j][i] / (o_hfa*V[j] + V[i])
            # END for
            v_ = V[i]
            V[i] = Y[i] / deno
            dv += (v_ - V[i])**2
        # END for
        
        dv = sqrt(dv)
        n += 1
    # END while
    
    P = np.zeros(len(teams))
    for i in range(0, len(teams), 2):
        P[i] = V[i] / (32.9 + V[i])
        P[i+1] = 1.0 / (V[i+1] + 1.0)
    # END for
    
    print 'Exited on round {0}, with |dv| = {1:0.2e}'.format(i, dv)
    n = 1
    print ''
    print 'Rk Team Defense              | p      | W:L  | Actual '
    print '-----------------------------------------------------------'
    for tm in sorted(teams[::2], key=lambda tm: P[i_lkup[tm]], reverse=True):
        i = i_lkup[tm]
        print (
            '{:>2d} {:.<25s} | {:0.4f} | {:>02.2f} | {:0.4f}'.format(
                n, tm, P[i], V[i], R[i]
            )
        )
        n += 1
    # END for
    n = 1
    print ''
    print 'Rk Team Offense              | p      | W:L  | Actual '
    print '-----------------------------------------------------------'
    for tm in sorted(teams[1::2], key=lambda tm: P[i_lkup[tm]]):
        i = i_lkup[tm]
        print (
            '{:>2d} {:.<25s} | {:0.4f} | {:>02.2f} | {:0.4f}'.format(
                n, tm, P[i], V[i]/32.9, R[i]
            )
        )
        n += 1
    # END for
    print ''
    
    # Normalize team values
    #A = 0.0
    #for tm in tm_vals: A += tm_vals[tm]
    #for tm in tm_vals: tm_vals[tm] /= A
    
    return V
# END bt_int_rank

#==============================================================================
def bt_mcmc_solve(games):
    # Collect teams
    teams = set()
    for gm in games:
        if gm.hm not in teams:
            teams.add(gm.hm)
        if gm.vs not in teams:
            teams.add(gm.vs)
    # END for
    teams = sorted(list(teams), key=lambda tm: str(tm))
    #print 'bt_mcmc_solve.teams = ['
    #for t in teams: print '    '+str(t)
    #print ']'
    #print 'len(bt_mcmc_solve.teams) = {}'.format(len(teams))
    i = 0
    for tm in teams:
        tm._bt_mcmc_i = i
        i += 1
    # END for
    
    N_steps = int(1e5)
    markov_chain = [None for i in range(N_steps)]
    markov_chain[0] = tuple( [0.0 for i in range(len(teams))] )
    lo_last_step = bt_likelihood(markov_chain[0], games)
    for i in range(N_steps-1):
        # print progress update
        #if N_steps > 60 and i%(N_steps/60) == 0:
            #print 'i = {:<6d} ({:>2d}/60)'.format(i, 60*i/N_steps)
        # propose next step in state space
        x_proposal = bt_mcmc_transition(markov_chain[i])
        # determine whether to change state or not
        lo_proposal = bt_likelihood(x_proposal, games)
        p_acceptance = np.exp( min( (0.0, lo_proposal - lo_last_step) ) )
        if p_acceptance == 1.0:
            markov_chain[i+1] = x_proposal
            lo_last_step = lo_proposal
        elif np.random.rand() <= p_acceptance:
            markov_chain[i+1] = x_proposal
            lo_last_step = lo_proposal
        else:
            markov_chain[i+1] = markov_chain[i]
        # END if
    # END for
    
    return markov_chain
# END bt_mcmc_solve

#==============================================================================
def bt_int_mcmc_solve(games):
    # Set up list of free parameter labels
    # They will alternate tm_off, tm_def, tm_off, tm_def, ...
    free_params = set()
    for gm in games:
        if str(gm.hm)+'_off' not in free_params:
            free_params.add(str(gm.hm)+'_off')
            free_params.add(str(gm.hm)+'_def')
        if str(gm.vs)+'_off' not in free_params:
            free_params.add(str(gm.vs)+'_off')
            free_params.add(str(gm.vs)+'_def')
    # END for
    free_params = sorted(list(free_params)) + ['hfo',]
    i_lkup = {}
    for i in range(len(free_params)):
        i_lkup[free_params[i]] = i
    # END for
    print 'len(bt_mcmc_solve.free_params) = {}'.format(len(free_params))
    
    # make K and N matricies
    N = len(free_params) - 1
    ATT = [[0.0 for j in range(N)] for i in range(N)]
    INT = [[0.0 for j in range(N)] for i in range(N)]
    intrate = [[0.0, 0.0] for i in range(N)]
    for gm in games:
        ATT[i_lkup[str(gm.hm)+'_off']][i_lkup[str(gm.vs)+'_def']] += gm.hm_att
        ATT[i_lkup[str(gm.hm)+'_def']][i_lkup[str(gm.vs)+'_off']] += gm.vs_att
        INT[i_lkup[str(gm.hm)+'_off']][i_lkup[str(gm.vs)+'_def']] += gm.hm_int
        INT[i_lkup[str(gm.hm)+'_def']][i_lkup[str(gm.vs)+'_off']] += gm.vs_int
        intrate[i_lkup[str(gm.hm)+'_off']][0] += gm.hm_int
        intrate[i_lkup[str(gm.hm)+'_off']][1] += gm.hm_att
        intrate[i_lkup[str(gm.vs)+'_def']][0] += gm.hm_int
        intrate[i_lkup[str(gm.vs)+'_def']][1] += gm.hm_att
        intrate[i_lkup[str(gm.vs)+'_off']][0] += gm.vs_int
        intrate[i_lkup[str(gm.vs)+'_off']][1] += gm.vs_att
        intrate[i_lkup[str(gm.hm)+'_def']][0] += gm.vs_int
        intrate[i_lkup[str(gm.hm)+'_def']][1] += gm.vs_att
    # END for
    
    N_steps = int(1e4)
    markov_chain = [None for i in range(N_steps)]
    all_L = np.zeros(N_steps)
    markov_chain[0] = np.array( 
        [(i%2)*3.572 for i in range(len(free_params)-1)] + [-0.024,] 
    )
    for i in range(0,N,2):
        print '{:.<30s} {:>2.0f}/{:<3.0f} ({:0.3f})'.format(
            free_params[i], intrate[i][0], intrate[i][1], markov_chain[0][i]
        )
        print '{:.<30s} {:>2.0f}/{:<3.0f} ({:0.3f})'.format(
            free_params[i+1], intrate[i+1][0], intrate[i+1][1],
            markov_chain[0][i+1]
        )
    # END for
    lo_last_step = ( np.sum(bt_int_def_prior(markov_chain[0][0:-1:2]))
                     + np.sum(bt_int_off_prior(markov_chain[0][1:-1:2]))
                     + bt_int_hfa_prior(markov_chain[0][-1])
                     + bt_int_likelihood(markov_chain[0], ATT, INT)
                   )
    all_L[0] = lo_last_step
    
    for i in range(N_steps-1):
        # print progress update
        if N_steps > 60 and i%(N_steps/60) == 0:
            print 'i = {:<6d} ({:>2d}/60) {:>10.0f}'.format(
                i, 60*i/N_steps, 10*lo_last_step/np.log(10)
            )
        # END if
        
        # propose next step in state space
        #x_proposal = bt_int_mcmc_transition(markov_chain[i])
        x_proposal = np.concatenate((
            bt_int_tm_trans(markov_chain[i][:-1]),
            [bt_int_hfa_trans(markov_chain[i][-1])]
        ))
        
        # determine whether to change state or not
        lo_proposal = ( np.sum(bt_int_def_prior(x_proposal[0:-1:2]))
                         + np.sum(bt_int_off_prior(x_proposal[1:-1:2]))
                         + bt_int_hfa_prior(x_proposal[-1])
                         + bt_int_likelihood(x_proposal, ATT, INT)
                       )
        
        # decide whether to take the proposed step
        p_acceptance = np.exp( min( (0.0, lo_proposal - lo_last_step) ) )
        #p_acceptance = 1 / (1 + np.exp(lo_proposal - lo_last_step))
        if p_acceptance == 1.0:
            markov_chain[i+1] = x_proposal
            lo_last_step = lo_proposal
        elif np.random.rand() <= p_acceptance:
            markov_chain[i+1] = x_proposal
            lo_last_step = lo_proposal
        else:
            markov_chain[i+1] = markov_chain[i]
        # END if
        all_L[i+1] = lo_last_step
    # END for
    
    return markov_chain, all_L, INT, ATT
# END bt_int_mcmc_solve

#==============================================================================
def bt_likelihood(values, games):
    lp = 0.0
    # calculate prior likelihood of proposal
    for v in values:
        lp += (v/0.75)**2
    # END for
    lp *= -0.5
    # calculate marginal likelihood of proposal
    o_tie = 0.002
    o_hfa = 1.27
    for gm in games:
        o_hm = np.exp(values[gm.hm._bt_mcmc_i])
        o_vs = np.exp(values[gm.vs._bt_mcmc_i])
        if gm.tie:
            p_outcome = ( o_tie*np.sqrt(o_hm*o_vs) /
                          (o_hfa*o_hm + o_vs + o_tie*np.sqrt(o_hm*o_vs))
                        )
        elif gm.winner == gm.hm:
            p_outcome = ( o_hfa*o_hm /
                          (o_hfa*o_hm + o_vs + o_tie*np.sqrt(o_hm*o_vs))
                        )
        else:
            p_outcome = ( o_vs /
                          (o_hfa*o_hm + o_vs + o_tie*np.sqrt(o_hm*o_vs))
                        )
        # END if
        lp += np.log(p_outcome)
        # END tie
    # END for
    
    return lp
# END bt_likelihood

#==============================================================================
def bt_int_def_prior(x):
    return -( x/0.244516 )**2 / 2
# END bt_int_prior

def bt_int_off_prior(x):
    return -( (x-3.572)/0.244516 )**2 / 2
# END bt_int_prior

def bt_int_hfa_prior(x):
    # Re-check the mean and stdev of this
    return -( (x - log(0.976))/ 0.70 )**2 / 2
# END bt_int_prior

def bt_int_likelihood(values, ATT, INT):
    # calculate marginal likelihood of proposal
    lp = 0.0
    o_hfa = np.exp(values[-1])
    N = len(ATT)
    for i in range(0,N,2):
        for j in range(0,N,2):
            o_hm_def = np.exp(values[i])
            o_hm_off = np.exp(values[i+1])
            o_vs_def = np.exp(values[j])
            o_vs_off = np.exp(values[j+1])
            p_hm_int = o_vs_def / (o_hfa*o_hm_off + o_vs_def)
            p_outcome = ( p_hm_int**INT[i+1][j]
                          * (1-p_hm_int)**(ATT[i+1][j]-INT[i+1][j])
                        )
            lp += np.log(p_outcome)
            #lp_outcome = ( INT[i+1][j] * values[j+1]
            #               - (ATT[i+1][j]-INT[i+1][j]) * values[i+1]
            #               - (ATT[i+1][j]-INT[i+1][j]) * values[0]
            #               - ATT[i+1] * np.log(o_hfa*o_hm_off + o_vs_def)
            #             )
            #lp += np.log(lp_outcome)
            p_vs_int = o_hfa*o_hm_def / (o_hfa*o_hm_def + o_vs_off)
            p_outcome = ( p_vs_int**INT[i][j+1]
                          * (1-p_vs_int)**(ATT[i][j+1]-INT[i][j+1])
                        )
            #try:
            #    intrate = INT[i][j+1]/ATT[i][j+1]
            #    print '{} ({}/{} = {})'.format(
            #        p_vs_int, INT[i][j+1], ATT[i][j+1], intrate
            #    )
            #    print '{:0.6f} -- > {:+0.3e}'.format( p_outcome,
            #                                          np.log(p_outcome)
            #                                        )
            #except:
            #    pass
            lp += np.log(p_outcome)
    # END for
    
    return lp
# END bt_int_likelihood

def bt_int_hfa_trans(x):
    if 0.5 < np.random.rand():
        return np.random.normal(loc=x, scale=(0.50 * 0.618))
    else:
        return x
    # END if
# END bt_int_hfa_trans

def bt_int_tm_trans(values):
    if 0.618 < np.random.rand():
        # randomly select parameter to vary
        i = np.random.randint(1, len(values))
        # randomly select a new value
        # scale (stdev) is set to the golden ratio times the stdev of the prior
        x = np.random.normal(loc=values[i], scale=(0.244516 * 0.618))
        proposed_values = list(values)
        proposed_values[i] = x
    else:
        # randomly vary all parameters a small amount
        dX = np.random.normal(size=len(values))
        proposed_values = values + (0.244516 * 0.618**2)*dX
    # END if
    return proposed_values
# END bt_mcmc_transition

def bt_int_mcmc_transition(values):
    proposed_values = list(values)
    if 0.5 < np.random.rand():
        x = np.random.normal(loc=values[-1], scale=(0.50 * 0.618))
        proposed_values[-1] = x
    # END if
    if 0.618 < np.random.rand():
        # randomly select parameter to vary
        i = np.random.randint(1, len(values))
        # randomly select a new value
        # scale (stdev) is set to the golden ratio times the stdev of the prior
        x = np.random.normal(loc=values[i], scale=(0.12 * 0.618))
        proposed_values[i] = x
    else:
        for i in range(1, len(values)):
            proposed_values[i] = np.random.normal( loc=values[i],
                                                   scale=(0.12 * 0.618**2)
                                                 )
        # END for
    # END if
    return tuple(proposed_values)
# END bt_mcmc_transition

#==============================================================================
class IntM1(object):
    '''
    '''
    
    def __init__(self, games):
        tms = set()
        for gm in games:
            tms.add(str(gm.hm)+'_off')
            tms.add(str(gm.hm)+'_def')
            tms.add(str(gm.vs)+'_off')
            tms.add(str(gm.vs)+'_def')
        # END for
        tms = sorted(list(tms))
        k = {}
        for i in range(len(tms)):
            k[tms[i]] = i
        # END for
        N = len(tms)
        self.ATT = np.zeros((N, N))
        self.INT = np.zeros((N, N))
        for gm in games:
            khmoff = k[str(gm.hm)+'_off']
            khmdef = k[str(gm.hm)+'_def']
            kvsoff = k[str(gm.vs)+'_off']
            kvsdef = k[str(gm.vs)+'_def']
            self.ATT[khmoff][kvsdef] += gm.hm_att
            self.ATT[khmdef][kvsoff] += gm.vs_att
            self.INT[khmoff][kvsdef] += gm.hm_int
            self.INT[khmdef][kvsoff] += gm.vs_int
        # END for
        self.CMP = self.ATT - self.INT
        
        self.x = 3.572
        # Mean of the prior distribution for x
        self.mx = 3.572
        # Standard deviation of the prior distribution for x
        self.sx = 0.244516
        # Log of the home-field advantage parameter
        self.lhfa = -0.024
    # END __init__
    
    def X_trans(self):
        # randomly vary by a small amount
        self.x = np.random.normal(self.x, self.sx * 0.05)
        return self.x
    # END X_trans
    
    def L_prior(self, x):
        return -( (x-self.mx)/self.sx )**2 / 4
    # END L_prior
    
    def L_marginal(self, x):
        # calculate marginal likelihood of proposal
        lp = 0.0
        o_hfa = np.exp( self.lhfa )
        o_off = np.exp(x)
        p_hm_int = 1.0       / (o_hfa*o_off + 1.0)
        p_vs_int = o_hfa*1.0 / (o_hfa*1.0 + o_off)
        N = self.ATT.shape[0]
        for i in range(0,N,2):
            for j in range(0,N,2):
                lp += self.INT[i+1][j] * np.log(p_hm_int)
                lp += self.CMP[i+1][j] * np.log(1-p_hm_int)
                lp += self.INT[i][j+1] * np.log(p_vs_int)
                lp += self.CMP[i][j+1] * np.log(1-p_vs_int)
        # END for
        return lp
    # END L_marginal
    
    def solve(self, N_steps=int(1e4)):
        x_chain = [None for i in range(N_steps)]
        x_chain[0] = self.x
        L_chain = np.zeros(N_steps)
        L_chain[0] = self.L_marginal(self.x) + self.L_prior(self.x)
        
        for i in range(N_steps-1):
            # print progress update
            if N_steps > 60 and i%(N_steps/60) == 0:
                print '({:>2d}/60th complete) {:>10.0f}'.format(
                    60*(i+1)/N_steps, 10*L_chain[i]/np.log(10)
                )
            # END if
            
            # propose next step in parameter space
            x_chain[i+1] = self.X_trans()
            
            # calculate the likelihood of model parameters
            L_chain[i+1] = self.L_marginal(self.x) + self.L_prior(self.x)
            #print L_chain[i+1]
            
            # decide whether to take the proposed step
            p_accept = np.exp( min( (0.0, L_chain[i+1] - L_chain[i]) ) )
            #print p_accept
            if p_accept == 1.0:
                pass
            elif np.random.rand() <= p_accept:
                pass
            else:
                x_chain[i+1] = x_chain[i]
                L_chain[i+1] = L_chain[i]
            # END if
            #print ''
        # END for
        
        return x_chain, L_chain
    # END solve
    
    @classmethod
    def test(cls):
        year = 2013
        if year is None: year = int(time.strftime('%Y')) - 1
        print 'Testing IntM1 on {} season data'.format(year)
        
        # Load in the season schedule
        import pickle
        pickle_fn = 'NFL_games_{}.pypickle'.format(year)
        print 'loading {}'.format(pickle_fn)
        with open(pickle_fn, 'rb') as f:
            games = pickle.load(f)
        print '{} games'.format(len(games))
        
        # Run the solver
        M = cls(games)
        N = 10000
        x_chain, L_chain = M.solve(N)
        print 'mean parameter: {:0.3f}'.format(np.mean(x_chain))
        
        # Output the results
        import matplotlib as mpl
        mpl.use('TkAgg')
        from matplotlib import pyplot as plt
        # plot the parameter chain
        plt.plot(x_chain)
        # plot the prior distribution
        X = np.linspace(M.mx-4*M.sx, M.mx+4*M.sx, 12**3+1)
        Y = np.exp(M.L_prior(X))
        plt.plot(N*Y/np.max(Y), X)
        # plot the marginal distribution
        #X = np.linspace(M.mx-4*M.sx, M.mx+4*M.sx, 60)
        #Y = np.array( [M.L_marginal(x) for x in X] )
        #Y = np.exp(Y - np.max(Y))
        #pprint(Y)
        # plot the prior distribution
        plt.plot(N*Y, X)
        plt.show()
        plt.close()
        
        plt.hist(x_chain, bins=60, normed=True)
        #Y = np.exp(M.L_prior(X)) / M.sx / np.sqrt(2*np.pi)
        plt.show()
        plt.close()
        
    # END test
# END IntM1

#==============================================================================
class IntMFull(object):
    '''
    '''
    
    def __init__(self, games):
        tms = set()
        for gm in games:
            tms.add(str(gm.hm)+'_off')
            tms.add(str(gm.hm)+'_def')
            tms.add(str(gm.vs)+'_off')
            tms.add(str(gm.vs)+'_def')
        # END for
        tms = sorted(list(tms))
        k = {}
        for i in range(len(tms)):
            k[tms[i]] = i
        # END for
        N = len(tms)
        self.ATT = np.zeros((N, N))
        self.INT = np.zeros((N, N))
        for gm in games:
            khmoff = k[str(gm.hm)+'_off']
            khmdef = k[str(gm.hm)+'_def']
            kvsoff = k[str(gm.vs)+'_off']
            kvsdef = k[str(gm.vs)+'_def']
            self.ATT[khmoff][kvsdef] += gm.hm_att
            self.ATT[khmdef][kvsoff] += gm.vs_att
            self.INT[khmoff][kvsdef] += gm.hm_int
            self.INT[khmdef][kvsoff] += gm.vs_int
        # END for
        self.k = k
        self.CMP = self.ATT - self.INT
        
        # Parameter vector prior distribution mean
        self.mx = np.array(
            [(i%2)*3.572 for i in range(len(tms))] + [-0.024,]
        )
        # Parameter vector prior distribution standard deviation
        self.sx = np.array(
            [0.244516 for i in range(len(tms))] + [0.70,]
        )
        # inital parameter vector state
        self.x = np.array(
            [(i%2)*3.572 for i in range(len(tms))] + [-0.024,]
        )
    # END __init__
    
    def X_trans(self, t=1e100):
        g = 1 - np.exp(-t-1)
        g = 1 - t**(-1.618)
        if 0.5 < np.random.rand():
            self.x[-1] += 0.618**2 * self.sx[-1] * g * np.random.normal()
        # END if
        if 0.618 < np.random.rand():
            # randomly select parameter to vary
            i = np.random.randint(1, len(self.x)-1)
            # randomly select a new value
            # scale (stdev) is set to the golden ratio times the stdev of
            # the prior
            self.x[i] += 0.618 * self.sx[i] * g * np.random.normal()
        else:
            for i in range(1, len(self.x)-1):
                self.x[i] += 0.618**2 * self.sx[i] * g * np.random.normal()
            # END for
        # END if
        return self.x
    # END X_trans
    
    def L_prior(self):
        return np.sum( -( (self.x-self.mx)/self.sx )**2 / 2 )
    # END L_prior
    
    def L_marginal(self):
        # calculate marginal likelihood of proposal
        lp = 0.0
        w = np.exp(self.x)
        N = self.ATT.shape[0]
        for i in range(0,N,2):
            for j in range(0,N,2):
                p_hm_int = w[j] / (w[-1]*w[i+1] + w[j])
                lp += self.INT[i+1][j] * np.log(p_hm_int)
                lp += self.CMP[i+1][j] * np.log(1-p_hm_int)
                p_vs_int = w[-1]*w[i] / (w[-1]*w[i] + w[j+1])
                lp += self.INT[i][j+1] * np.log(p_vs_int)
                lp += self.CMP[i][j+1] * np.log(1-p_vs_int)
        # END for
        return lp
    # END L_marginal
    
    def solve(self, N_steps=int(1e4)):
        x_chain = [None for i in range(N_steps)]
        x_chain[0] = tuple( self.x )
        L_chain = np.zeros(N_steps)
        L_chain[0] = self.L_marginal() + self.L_prior()
        print L_chain[0]
        
        for i in range(N_steps-1):
            t = 5.0 - 4.0*(i/N_steps)
            # print progress update
            if N_steps > 60 and (i-1)%(N_steps/60) == 0:
                print '({:>2d}/60th complete) {:>10.0f}'.format(
                    60*(i+1)/N_steps, L_chain[i]
                )
            # END if
            
            # propose next step in parameter space
            #print self.x
            x_chain[i+1] = tuple( self.X_trans(t) )
            #print self.x
            
            # calculate the likelihood of model parameters
            L_chain[i+1] = self.L_marginal() + self.L_prior()
            #print L_chain[i+1]
            
            # decide whether to take the proposed step
            p_accept = np.exp( min( (0.0, (L_chain[i+1] - L_chain[i]) / t) ) )
            #print p_accept
            if p_accept == 1.0:
                pass
            elif np.random.rand() <= p_accept**(1/t):
                pass
            else:
                self.x = np.array( x_chain[i] )
                x_chain[i+1] = x_chain[i]
                L_chain[i+1] = L_chain[i]
            # END if
            #print ''
            
            if i%(N_steps/20) == 0:
                j = np.argmax(L_chain[:i+1])
                L_chain[i+1] = L_chain[j]
                x_chain[i+1] = x_chain[j]
                self.x = np.array( x_chain[j] )
            # END for
        # END for
        
        return x_chain, L_chain
    # END solve
    
    @classmethod
    def test(cls):
        year = 2013
        if year is None: year = int(time.strftime('%Y')) - 1
        print 'Testing IntM1 on {} season data'.format(year)
        
        # Load in the season schedule
        import pickle
        pickle_fn = 'NFL_games_{}.pypickle'.format(year)
        print 'loading {}'.format(pickle_fn)
        with open(pickle_fn, 'rb') as f:
            games = pickle.load(f)
        print '{} games'.format(len(games))
        
        # Run the solver
        M = cls(games)
        N = 1000
        x_chain, L_chain = M.solve(N)
        x_chain = np.array(x_chain)
        
        # Output results
        mx = np.mean(x_chain, axis=0)
        print 'defenses'
        pprint(mx[0:-1:2])
        print 'offenses'
        pprint(mx[1:-1:2])
        print 'hfa: {}'.format(mx[-1])
        
        import matplotlib as mpl
        mpl.use('TkAgg')
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(L_chain, '-k')
        plt.show()
        plt.close()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(0, x_chain.shape[1]-1, 2):
            ax.plot(x_chain[:,i], '-r', alpha=0.2)
            ax.plot(x_chain[:,i+1], '-b', alpha=0.2)
        ax.plot(x_chain[:,-1], '-k', linewidth=2)
        plt.title('All Markov Chains (off in blue, def in red)')
        plt.show()
    # END test
# END IntMFull

#==============================================================================
def bt_mcmc_transition(values):
    proposed_values = list(values)
    if 0.618 < np.random.rand():
        # randomly select parameter to vary
        i = np.random.randint(len(values))
        # randomly select a new value
        # scale (stdev) is set to the golden ratio times the stdev of the prior
        x = np.random.normal(loc=values[i], scale=(0.75 * 0.618))
        proposed_values[i] = x
    else:
        for i in range(len(values)):
            proposed_values[i] = np.random.normal( loc=values[i],
                                                   scale=(0.75 * 0.618**2)
                                                 )
        # END for
    # END if
    return tuple(proposed_values)
# END bt_mcmc_transition

#==============================================================================
class ProbDenFunc(object):
    '''Probability Density Function Class
    
    Class Methods:
        make_func
        make_normal
    Implemented Operations:
        ==, !=, +, -, *, /, **, +=, -=, *=, /=, **=, abs, neg, pos,
        call, iter, len, getitem
    Instance Attributes:
        X (ndarray): Independent variable
        Y (ndarray): Probability
    Instance Methods:
        copy
        interp_lin
    '''
    
    def __init__(self, X, Y):
        if not isinstance(X, np.ndarray): X = np.array(X)
        if not isinstance(Y, np.ndarray): Y = np.array(Y)
        self.__X = X
        self.__Y = Y
    # END __init__
    
    # Constructor Methods
    #--------------------
    def copy(self): return type(self)(self.__X, self.__Y)
    
    @classmethod
    def make_func(cls, fn, x_min, x_max, N):
        X = np.linspace(x_min, x_max, N)
        Y = np.zeros(N)
        for i in range(N): Y[i] = fn(X[i])
        return cls(X, Y)
    # END make_func
    
    @classmethod
    def make_lognormal(cls, x_min, x_max, N, mean=0.0, std=1.0):
        out = cls.make_normal(x_min, x_max, N, mean, std)
        for i in range(len(out)):
            out.__X[i] = exp(out.__X[i])
        # END for
        return out
    # END make_normal
    
    @classmethod
    def make_normal(cls, x_min, x_max, N, mean=0.0, std=1.0):
        return cls.make_func(
            lambda x: (1.0/std/sqrt(2*pi))*exp(-((x-mean)/std)**2/2.0),
            x_min, x_max, N
        )
    # END make_normal
    
    # Property access methods
    #------------------------
    @property
    def X (self): return self.__X
    @property
    def Y (self): return self.__Y

    # Magic methods
    #----------------
    def __abs__(self): return type(self)(self.__X, abs(self.__Y))
    
    def __add__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return type(self)(self.__X, self.__Y+other)
        if ProbDenFunc.array_comp(self.X, other.X):
            return type(self)(self.__X, self.__Y+other.Y)
    # END __add__
    
    def __radd__(self, other): return self + other
    # END __add__
    
    def __iadd__(self, other):
        if ProbDenFunc.array_comp(self.X, other.X): self.__Y += other.Y
    # END __iadd__
    
    def __div__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return type(self)(self.__X, self.__Y/other)
        if ProbDenFunc.array_comp(self.X, other.X):
            return type(self)(self.__X, self.__Y/other.Y)
    # END __div__
    
    def __rdiv__(self, other): return self**(-1) * other
    
    def __idiv__(self, other):
        if ProbDenFunc.array_comp(self.X, other.X): self.__Y /= other.Y
    # END __idiv__
    
    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return type(self)(self.__X, self.__Y*other)
        if ProbDenFunc.array_comp(self.X, other.X):
            return type(self)(self.__X, self.__Y*other.Y)
    # END __mul__
    
    def __rmul__(self, other): return self * other
    
    def __imul__(self, other):
        if ProbDenFunc.array_comp(self.X, other.X): self.__Y *= other.Y
    # END __imul__
    
    def __neg__(self): return type(self)(self.__X, -self.__Y)
    
    def __pos__(self): return self
    
    def __pow__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return type(self)(self.__X, self.__Y**other)
        if ProbDenFunc.array_comp(self.X, other.X):
            return type(self)(self.__X, self.__Y**other.Y)
    # END __pow__
    
    def __ipow__(self, other):
        if ProbDenFunc.array_comp(self.X, other.X): self.__Y **= other.Y
    # END __ipow__
    
    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return type(self)(self.__X, self.__Y-other)
        if ProbDenFunc.array_comp(self.X, other.X):
            return type(self)(self.__X, self.__Y-other.Y)
    # END __sub__
    
    def __rsub__(self, other): return (-self) + other
    
    def __isub__(self, other):
        if ProbDenFunc.array_comp(self.X, other.X): self.__Y -= other.Y
    # END __isub__
    
    def __call__(self, *args):
        '''A rediculously recurrsive way to call interp_lin! (or just get X & Y)
        '''
        
        if len(args) == 0: return self.__X, self.__Y
        
        if len(args) > 1: return [self(x) for x in args]
        
        arg = args[0]
        if isinstance(arg, list) or isinstance(arg, np.ndarray):
            return [self(x) for x in arg]
        
        return self.interp_lin(arg)
    # END __call__
    
    def __eq__(self, other):
        if type(other) is not type(self):
            raise TypeError(
                'Cannot equate a {} and a {}'.format(type(self), type(other))
            )
        return (ProbDenFunc.array_comp(self.X, other.X) and self.__Y == other.Y)
    # END __eq__
    
    def __iter__(self):
        for i in range(len(self)): yield self[i][0], self[i][1]

    def __len__(self): return len(self.__X)
    
    def __getitem__(self, key): return self.__X[key], self.__Y[key]
    
    def __ne__(self, other): return not self.__eq__(other)
    
    def __setitem__(self, key, value):
        if isinstance(value, int):
            value = float(value)
        elif not isinstance(value, float):
            raise TypeError("ProbDenFunc y's may only contain numeric types")
        # END if
        self.__Y[key] = value
    # END __setitem__

    # Object-specific methods
    #------------------------
    def interp_lin(self, x):
        '''Estimate y(x) using a linear interpolation
        '''
        if x < self.__X[0] or self.__X[-1] < x:
            raise ValueError(
                'Cannot interpolate a value ({})'.format(x) +
                'outside domain ([{},{}])'.format(self.__X[0], self.__X[-1])
            )
        
        dx = self.__X[1] - self.__X[0]
        ia = int((x-self.__X[0])/dx)
        if self.__X[ia] == x:
            return self.__Y[ia]
        elif self.__X[ia+1] == x:
            return self.__Y[ia+1]
        else:
            return (
                self.__Y[ia] +
                ((self.__Y[ia+1]-self.__Y[ia])/dx)*(x-self.__X[ia])
            )
        # END if
    # END interp_lin
    
    #def quick_show(self):
    #	plt.plot(self.__X, self.__Y, '.-')
    #	plt.show()
    #	plt.close()
    # END quick_show
    
    # Utility methods
    #----------------
    @staticmethod
    def array_comp(X1, X2):
        if len(X1) != len(X2): return False
        for i in range(len(X1)):
            if X1[i] != X2[i]: return False
        return True
    # END ProbDenFunc.array_comp
# END ProbDenFunc

#===============================================================================
def calc_score_diff(v1, v2=None):
    '''Calculate Expected Score Difference
    '''
    
    if isinstance(v1, float) and isinstance(v2, float):
        p1 = v1/(v1 + v2)
        
        # Standard deviation in score (*Note: this is sport specific)
        sig = 11.87
        
        # final score difference at a neutral location
        # score difference = home score - visitor score
        pnts = sig*sqrt(2.0)*erfinv(2.0*p1 - 1.0)
        
        return pnts
    elif isinstance(v1, dict):
        # Calculate expected score over median team for a set of teams
        v_out = {}
        v_median = np.median(v1.values())
        for tm in v1:
            v_out[tm] = calc_score_diff(v1[tm], v_median)
        
        return v_out
    else:
        try:
            # Assume v is an iterable that returns individual v's, e.g. a list
            v_out = []
            v_median = np.median(v1)
            for v_ in v1:
                v_out.append( calc_score_diff(v_, v_median) )
            
            return v_out
        except TypeError:
            raise TypeError(
                'calc_score_diff does not accept input of type '
                + type(v1).__name__
            )
        # END try
    # END if
# END calc_score_diff

#===============================================================================
def normalize_ranks(tm_vals):
    '''Normalize Team Rankings
    '''
    
    x_median = np.median(tm_vals.values())
    for tm in tm_vals:
        tm_vals[tm] /= x_median
    return tm_vals
# END normalize_ranks

#===============================================================================
def calc_gwp(tm_vals):
    mdn = np.median(tm_vals.values())
    for tm in tm_vals:
        tm_vals[tm] = tm_vals[tm] / (tm_vals[tm]+mdn)
    return tm_vals
# END calc_gwp

#===============================================================================
def write_schedule_details(games, tm_vals, year):
    max_gms_played = 0
    tm_schds = {}
    for tm in sorted(tm_vals.keys()):
        tm_schds[tm] = []
        for gm in games:
            if gm.hm == tm or gm.vs == tm:
                tm_schds[tm].append(gm)
                if len(tm_schds[tm]) > max_gms_played:
                    max_gms_played += 1
            # END if
        # END for
    # END for
    
    f_txt = 'Team Name,Vertex Value'
    for i in range(1, max_gms_played+1):
        f_txt += ',game ' + str(i)
    f_txt += r'\\n'
    
    for tm in tm_schds:
        f_txt += '{0},{1:0.3f}'.format(tm, tm_vals[tm])
        for gm in sorted(tm_schds[tm], key=lambda g: g.wk):
            PoEx = (
                gm.hmsc + 4.579*gm.hmto
                - gm.vssc - 4.579*gm.vsto - 1.9272
            )
            try:
                # TODO: make sure all games have hosted attribute, this
                # assumption that the game is hosted is not ideal
                if not gm.hosted:
                    PoEx += 1.9272
            except KeyError:
                pass
            # END try
            scdiff = gm.hmsc - gm.vssc
            if tm == gm.hm:
                opp = gm.vs
                w_in = gm.value() * tm_vals[opp]
                w_out = (1-gm.value()) * tm_vals[tm]
            elif tm == gm.vs:
                opp = gm.hm
                w_in = (1-gm.value()) * tm_vals[opp]
                w_out = gm.value() * tm_vals[tm]
                PoEx *= -1
                scdiff *= -1
            # END if
            f_txt += ',{0:0.3f} < {1} ({3:+d}/{2:+0.1f})'.format(
                10*(w_in-w_out), NFL.normalize_name(opp), PoEx, int(scdiff)
            )
        # END for
        f_txt += r'\\n'
    # END for
    
    # save a csv/html file
    f = open('../../../ampPy/example_chimera_table.csv.html')
    template = f.read()
    f.close()
    
    f_txt = re.sub(r"(?<=var csv_data = ')[^']*?(?=')", f_txt, template)
    f = open('NFL_schedule_details_{}.csv.html'.format(year), 'w')
    f.write(f_txt)
    f.close()
    
    # save a csv file
    f = open('up-to-date_ranks.csv', 'w')
    for tm in sorted(tm_vals.keys()):
        if tm == 'Average Joes':
            continue
        f.write('{},{}\n'.format(NFL.normalize_name(tm), log(tm_vals[tm])))
    f.close()
# write_schedule_details

#===============================================================================
def predictGm(v_hm, v_vs, pnt_adj=0.0):
    '''
    (P[hm win], finalScore) = predictGm(v_hm, v_vs, ('H'|'N'), ('M'|'W'))
    '''
    
    # Standard deviation in score
    sig = 11.87
    sigSqrt2 = sig*sqrt(2.0)
    
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
def value_game(gm_dict):
    '''Game valuing function
    
    Here's the idea.  If, for example, a team wins by 16 at home then this
    performance is better than 68.5% (or 1 stdev) of all other game outcomes.
    Therefore, the random walker voter will judge this team as
    the better team 68.5% of the time.  Or, 68.5% of RW voters will pick that
    team as better.
    '''
    
    # Raise error if game is not final
    if ('hmsc' not in gm_dict
        or 'vssc' not in gm_dict
        or 'hmto' not in gm_dict
        or 'vsto' not in gm_dict):
        raise ValueError('Unfinished game cannot be valued')
    # END if
    
    # Expected points the home team will win by
    #z = 4.579*(gm_dict['vsto']-gm_dict['hmto']) + 1.9272
    # Point difference the home team created beyond what was expected (PoEx)
    #z = (gm_dict['hmsc'] - gm_dict['vssc']) - z
    # Normalize PoEx, dividing by historical standard deviation
    #z /= 11.87
    
    #p = {'V2H': 0.0, 'H2V': 0.0}
    #p['V2H'] = 0.5*(1.0 + erf( z/sqrt(2.0) ))
    #p['H2V'] = 1.0 - p['V2H']
    
    # dev code
    #x = gm['hmsc'] - 4.579*gm['vsto'] - 1.927
    #y = gm['vssc'] - 4.579*gm['hmto'] - 0.290*(gm['hmsc']-4.579*gm['vsto'])
    #z = x - y
    z = -4.579*(gm_dict['hmto']-gm_dict['vsto']) + 1.9272
    z = gm_dict['hmsc'] - gm_dict['vssc'] - z
    p_ = invlogit(0.0698*z + 0.0038)
    p = {'V2H': p_, 'H2V': 1-p_}
    
    return p
# END valueGm

#===============================================================================
def rowElim(X):
    '''Row-Elimination
    
    Args:
        X (list): Square matrix represented as a list of lists
    Returns:
        (list) Row-eliminated version of X, represented as a list of lists
    '''
    
    L = len(X)
    # This may not be necessary to make a copy of the input here
    Y = [[X[i][j] for j in range(0,L)] for i in range(0,L)]
    tol = 1E-12
    
    for i in range(0,L):
        a = Y[i][i]
        if abs(a) <= tol:
            Y[i][i] = 0.0
            continue
        # END if
        
        # Divide the entire row by the leading entry so that the leading term
        # is 1
        for j in range(0,L):
            Y[i][j] = Y[i][j] / a
        
        # Reduce all rows below the current row
        for k in range(i+1,L):
            b = Y[k][i]
            for j in range(0,L):
                Y[k][j] -= b*Y[i][j]
        # END for
    # END for
    
    return Y
# END rowElim

#===============================================================================
def logit(p): return np.log(p/(1.0-p))
def invlogit(x): return 1.0 / (1.0 + np.exp(-x))

#===============================================================================
if __name__ == '__main__':
    IntMFull.test()
# END if
