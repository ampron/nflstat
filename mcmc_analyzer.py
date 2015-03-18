#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''Script or Module Title

    This section should be a summary of important information to help the
    editor understand the purpose and/or operation of the included code.

    Module dependencies:
        sys
    List of classes: -none-
    List of functions:
        main
'''

# built-in modules
import pickle

# third-party modules
import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
from NFL_class import *
import rank_NFL

# self package

#==============================================================================
def main(pickle_fn):
    print 'loading pickle'
    f = open(pickle_fn, 'rb')
    mc = pickle.load(f)
    f.close()
    print '    finished'
    
    print 'ndarray-ing markov chain'
    mc = np.array(mc)
    N_steps = mc.shape[0]
    print 'mc.shape = {}'.format(repr(mc.shape))
    print '    finished'
    
    #NFL.team_names.append('Green Bay Packers*')
    #NFL._tm_aliases['GB*'] = 'Green Bay Packers*'
    #NFL._tm_aliases['Green Bay Packers*'] = 'GB*'
    #NFL._tm_abreviations['Green Bay Packers*'] = 'GB*'
    #NFL._tm_aliases['Packers*'] = 'GB*'
    #NFL._tm_groups['NFC'].append('Green Bay Packers*')
    #NFL._tm_groups['NFC North'].append('Green Bay Packers*')
    #NFL._tm_divisions['Green Bay Packers*'] = 'NFC North'
    teams = sorted(NFL.team_names)
    teams = [Team(nm) for nm in teams]
    #print 'mcmc_analyzer.main.teams = ['
    #for t in teams: print '    '+str(t)
    #print ']'
    #print 'len(mcmc_analyzer.main.teams) = {}'.format(len(teams))
    
    all_means = np.mean(mc, axis=0)
    all_stdevs = np.std(mc, axis=0)
    v_lkup = {}
    for i in range(len(teams)):
        v_lkup[teams[i]] = (all_means[i], all_stdevs[i])
    n = 1
    print 'Rk Team                   | W:L  | mlnw  +/- err     | slnw'
    for tm in sorted(teams, key=lambda tm: v_lkup[tm], reverse=True):
        print (
            '{:>2d} {:.<22s} | {:0.2f} |'.format(
                n, str(tm), np.exp(v_lkup[tm][0])
            )
            + ' {:+0.2f} +/- {:0.1e} | {:0.2f}'.format(
                v_lkup[tm][0], v_lkup[tm][1]/np.sqrt(N_steps), v_lkup[tm][1]
            )
        )
        n += 1
    # END for
    
    nm = 'Denver Broncos'
    for i in range(len(teams)):
        if nm == str(teams[i]):
            break
    # END for
    print '{}:'.format(teams[i])
    tm1_mc = list(mc[:,i])
    tm1_mc_sorted = sorted(tm1_mc)
    mean = np.mean(tm1_mc)
    stdev = np.std(tm1_mc)
    err_low = tm1_mc_sorted[int( N_steps*(0.5*(1-0.68)) )]
    err_upp = tm1_mc_sorted[-int( N_steps*(0.5*(1-0.68)) )]
    print '    mean value: {:0.3f} +/- {:0.6f}'.format(
        mean, stdev/np.sqrt(N_steps)
    )
    print '    standard deviation of value: {:0.3f}'.format(stdev)
    print '    CI = [{:0.3f}, {:0.3f}]'.format(err_low, err_upp)
    
    nm = 'Seattle Seahawks'
    for i in range(len(teams)):
        if nm == str(teams[i]):
            break
    # END for
    print '{}:'.format(teams[i])
    tm2_mc = list(mc[:,i])
    tm2_mc_sorted = sorted(tm2_mc)
    mean = np.mean(tm2_mc)
    stdev = np.std(tm2_mc)
    err_low = tm2_mc_sorted[int( N_steps*(0.5*(1-0.68)) )]
    err_upp = tm2_mc_sorted[-int( N_steps*(0.5*(1-0.68)) )]
    print '    mean value: {:0.3f} +/- {:0.6f}'.format(
        mean, stdev/np.sqrt(N_steps)
    )
    print '    standard deviation of value: {:0.3f}'.format(stdev)
    print '    CI = [{:0.3f}, {:0.3f}]'.format(err_low, err_upp)
    
    gm_mc = [
        tm1_mc[i] - tm2_mc[i] for i in range(N_steps)
    ]
    print 'mean ln(odds) for DEN over SEA: {:0.3f} +/- {:0.6f}'.format(
        np.mean(gm_mc), np.std(gm_mc)/np.sqrt(N_steps)
    )
    print 'mean odds for DEN over SEA: {:0.3f}'.format(np.exp(np.mean(gm_mc)))
    #plt.hist(gm_mc, bins=60)
    #plt.show()
    #plt.hist(np.exp(gm_mc), bins=60)
    #plt.show()
    #plt.hist(1.0 / (1.0 + np.exp(-np.array(gm_mc))), bins=60)
    #plt.show()
    #plt.plot(range(N_steps), tm1_mc, 'o-', range(N_steps), tm2_mc, '*-')
    #plt.plot(tm1_mc, tm2_mc, 'o-')
    #plt.show()
    
    plt.plot(tm1_mc, '*-', tm2_mc, '*-')
    plt.title('Broncos, Seahawks markov chains')
    plt.show()
    
    plt.hist(tm1_mc, bins=60)
    plt.title('Broncos distribution')
    plt.show()
    
    plt.hist(tm2_mc, bins=60)
    plt.title('Seahawks distribution')
    plt.show()
# END main

#==============================================================================
def int_mcmc_analyzer(pickle_fn):
    print 'loading pickle "{}"'.format(pickle_fn)
    f = open(pickle_fn, 'rb')
    mc, L, mc_m1, L_m1, INT, ATT = pickle.load(f)
    f.close()
    print '    finished'
    
    print 'ndarray-ing markov chain'
    mc = np.array(mc)
    N_steps = mc.shape[0]
    print 'mc.shape = {}'.format(repr(mc.shape))
    print '    finished'
    
    teams = []
    for t in NFL.team_names:
        teams.append(t+'_def')
        teams.append(t+'_off')
    # END for
    teams = sorted(list(teams))
    i_lkup = {}
    for i in range(len(teams)):
        i_lkup[teams[i]] = i
    # END for
    
    all_means = np.mean(mc, axis=0)
    all_stdevs = np.std(mc, axis=0)
    v_lkup = {}
    for i in range(len(teams)):
        v_lkup[teams[i]] = (all_means[i], all_stdevs[i])
    i += 1
    hfo = (all_means[i], all_stdevs[i])
    teams_unsrt = list(teams)
    teams.sort(key=lambda tm: v_lkup[tm], reverse=True)
    teams.sort(key=lambda tm: tm[-3:])
    
    mm = np.mean(mc_m1)
    ms = np.stdev(mc_m1)
    y = tuple( 
        [(i%2)*mm for i in range(len(free_params)-1)] + [-0.024,] 
    )
    mL_m1 = ( rank_NFL.bt_int_1tm_prior(mm)
              +rank_NFL.bt_int_likelihood(y, ATT, INT)
            )
    mL_full = rank_NFL.bt_int_likelihood(all_means, ATT, INT)
    mL_full = ( np.sum(bt_int_def_prior(all_means[0:-1:2]))
                + np.sum(bt_int_off_prior(all_means[1:-1:2]))
                + bt_int_hfa_prior(all_means[-1])
                + bt_int_likelihood(all_means, ATT, INT)
              )
    print 'L_mean : L_prior = {:0.1f} dB'.format(
        10*(L_mean-L_m1)/np.log(10)
    )
    
    n = 1
    mdef = np.median(all_means[::2])
    moff = np.median(all_means[1::2])
    print 'Home-Field advantage = {:0.2f} |'.format(hfo[0])
    print 'Median Off: {:+0.2f}           |'.format(moff)
    print 'Median Def: {:+0.2f}           |'.format(mdef)
    print ''
    print 'Rk Team                     | AI%   | mlnw  +/- err     | slnw'
    print '----------------------------+-------+-------------------+-----'
    for tm in teams:
        if n < 33:
            p = np.exp(v_lkup[tm][0]) / (np.exp(v_lkup[tm][0]) + np.exp(moff))
        else:
            p = np.exp(mdef) / ( np.exp(mdef) + np.exp(v_lkup[tm][0]) )
        if n == 33:
            print 62*'-'
        print (
            '{:>2d} {:.<24s} | {:>05.3f} |'.format((n-1)%32+1, str(tm), p)
            + ' {:+0.2f} +/- {:0.1e} | {:0.2f}'.format(
                v_lkup[tm][0], v_lkup[tm][1]/np.sqrt(N_steps), v_lkup[tm][1]
            )
        )
        n += 1
    # END for
    
    fig, ax = plt.subplots()
    ax.hist(mc_m1, bins=60, normed=True)
    pm = 3.572
    ps = 0.244516*np.sqrt(2)
    xdisp = np.linspace(m-4*s, m+4*s, 12**3+1)
    ax.plot(
        xdisp, np.exp(-0.5*((xdisp-m)/s)**2)/s/np.sqrt(2*np.pi), '-r'
    )
    xdisp = np.linspace(mm-4*ms, mm+4*ms, 12**3+1)
    ax[0].plot(
        xdisp, np.exp(-0.5*((xdisp-mm)/ms)**2)/ms/np.sqrt(2*np.pi),
        '-k', linewidth=2
    )
    ax.set_title('Single parameter model PDF')
    plt.show()
    plt.close()
    
    nm = 'Green Bay Packers_off'
    for i in range(len(teams_unsrt)):
        if nm == str(teams_unsrt[i]):
            break
    # END for
    print '{}:'.format(teams_unsrt[i])
    tm1_mc = list(mc[:,i])
    tm1_mc_sorted = sorted(tm1_mc)
    mean1 = np.mean(tm1_mc)
    stdev1 = np.std(tm1_mc)
    err_low = tm1_mc_sorted[int( N_steps*(0.5*(1-0.68)) )]
    err_upp = tm1_mc_sorted[-int( N_steps*(0.5*(1-0.68)) )]
    print '    mean value: {:0.3f} +/- {:0.6f}'.format(
        mean1, stdev1/np.sqrt(N_steps)
    )
    print '    standard deviation of value: {:0.3f}'.format(stdev1)
    print '    CI = [{:0.3f}, {:0.3f}]'.format(err_low, err_upp)
    
    nm = 'Green Bay Packers_def'
    for i in range(len(teams_unsrt)):
        if nm == str(teams_unsrt[i]):
            break
    # END for
    print '{}:'.format(teams_unsrt[i])
    tm2_mc = list(mc[:,i])
    tm2_mc_sorted = sorted(tm2_mc)
    mean2 = np.mean(tm2_mc)
    stdev2 = np.std(tm2_mc)
    err_low = tm2_mc_sorted[int( N_steps*(0.5*(1-0.68)) )]
    err_upp = tm2_mc_sorted[-int( N_steps*(0.5*(1-0.68)) )]
    print '    mean value: {:0.3f} +/- {:0.6f}'.format(
        mean2, stdev2/np.sqrt(N_steps)
    )
    print '    standard deviation of value: {:0.3f}'.format(stdev2)
    print '    CI = [{:0.3f}, {:0.3f}]'.format(err_low, err_upp)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].hist(tm1_mc, bins=60, normed=True)
    m = 3.572
    s = 0.244516
    xdisp = np.linspace(m-4*s, m+4*s, 12**3+1)
    ax[0].plot(
        xdisp, np.exp(-0.5*((xdisp-m)/s)**2)/s/np.sqrt(2*np.pi), '-r'
    )
    xdisp = np.linspace(mean1-4*stdev1, mean1+4*stdev1, 12**3+1)
    ax[0].plot(
        xdisp, np.exp(-0.5*((xdisp-mean1)/stdev1)**2)/stdev1/np.sqrt(2*np.pi),
        '-k', linewidth=2
    )
    ax[0].set_title('Green Bay Packers_off')
    ax[1].hist(tm2_mc, bins=60, normed=True)
    m = 0.0
    s = 0.244516
    xdisp = np.linspace(m-4*s, m+4*s, 12**3+1)
    axtwin1 = ax[1].twinx()
    axtwin1.plot(
        xdisp, np.exp(-0.5*((xdisp-m)/s)**2)/s/np.sqrt(2*np.pi), '-r'
    )
    xdisp = np.linspace(mean2-4*stdev2, mean2+4*stdev2, 12**3+1)
    ax[1].plot(
        xdisp, np.exp(-0.5*((xdisp-mean2)/stdev2)**2)/stdev2/np.sqrt(2*np.pi),
        '-k', linewidth=2
    )
    ax[1].set_title('Green Bay Packers_def')
    plt.show()
    
    #
    #gm_mc = [
    #    tm1_mc[i] - tm2_mc[i] for i in range(N_steps)
    #]
    #print 'mean prob. for GRB INT to CHI: {:0.3f}'.format(
    #    1 / (1 + np.exp(-np.mean(gm_mc)))
    #)
    #plt.hist(gm_mc, bins=60)
    #plt.show()
    #plt.hist(np.exp(gm_mc), bins=60)
    #plt.show()
    #plt.hist(1.0 / (1.0 + np.exp(-np.array(gm_mc))), bins=60)
    #plt.show()
    
    # Plot all Markov chains
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #for i in range(0, mc.shape[1]-2, 2):
    #    ax.plot(mc[:,i], '-r', alpha=0.2)
    #    ax.plot(mc[:,i+1], '-b', alpha=0.2)
    #ax.plot(mc[:-1], '-k', linewidth=2, alpha=0.2)
    #plt.title('All Markov Chains (off in blue, def in red)')
    #plt.show()
# END int_mcmc_analyzer

#==============================================================================
if __name__ == '__main__':
    int_mcmc_analyzer("NFL_int_mcmc_results_2014.pypickle")
# END if