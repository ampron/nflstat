#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''NFL Studies
    
    This section should be a summary of important infomation to help the editor
    understand the purpose and/or operation of the included code.
    
    List of classes: -none-
    List of functions:
        main
'''

# built-in modules
import sys
import traceback
import pickle
from pprint import pprint
from StringIO import StringIO

# third party modules
import numpy as np
from scipy import polyval, polyfit
from scipy.stats import pearsonr, skew, kurtosis, linregress, probplot
from scipy.stats import t as Tdist
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

# personal modules
try:
    import collectors_NFL as clct
    import rank_NFL as rank
    from NFL_class import NFL, WLTRecord
except:
    sys.path.append('../../../ampPy/')
    
    import collectors_NFL as clct
    #import rank_NFL as rank
    from NFL_class import NFL, WLTRecord
    
    print '*** added "../../../ampPy/" to sys.path ***'
# END try

#===============================================================================
def main(*args):
    #t = LabeledTable(['a', 'b'], ['x', 'y'])
    #t['a',:]
    #season_compile()
    get_wins_dist()
    
    #get_PoEx_dist()
    #scoring_corr()
    #interceptions()
    
    #x = np.array(x)
    #plt.hist(np.log(x))
    #plt.show()
    #plt.close()
    #print np.std(x)
    #quit()
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #for y in range(2002, 2012):
    #   v = rank_process(str(y))
    #   v = np.log(np.array(sorted(v, reverse=True)))
    #   ax.plot(v)
    #   print 'm= {0:0.5f}, s= {1:0.5f}'.format(np.mean(v), np.std(v))
    #plt.show()
    #plt.close()
# END main

#==============================================================================
def interceptions():
    '''League-wide interception rate over the last 9 years
    ints/att = 4683 / 158569
    This gives a rate of 0.0295
    The 0.997 binomial CI is [0.0283, 0.0308]
    or ~ ±0.0013
    '''
    games = []
    for year in range(2005, 2014):
        pickle_fn = 'NFL_games_{}.pypickle'.format(year)
        print 'loading pickle, ' + pickle_fn
        f = open(pickle_fn, 'rb')
        games.extend(pickle.load(f))
        f.close()
    # END for
    
    hm_ints = 0
    hm_atts = 0
    vs_ints = 0
    vs_atts = 0
    for g in games:
        try:
            hm_ints += g.hm_int
            hm_atts += g.hm_att
            vs_ints += g.vs_int
            vs_atts += g.vs_att
        except AttributeError as err:
            print g['date']
            hm_ints += g['hm_int']
            hm_atts += g['hm_att']
            vs_ints += g['vs_int']
            vs_atts += g['vs_att']
    # END for
    
    print 'Total'
    print hm_ints + vs_ints
    print hm_atts + vs_atts
    p = float(hm_ints + vs_ints)/float(hm_atts + vs_atts)
    print '{} +/- {:0.4f} or 1:{:0.1f}'.format( p,
                                               1.0/np.sqrt(hm_atts+vs_atts),
                                               (1-p)/p
                                             )
    print 'Home teams'
    print hm_ints
    print hm_atts
    p = float(hm_ints)/hm_atts
    print '{} or 1:{:0.1f}'.format(p, (1-p)/p)
    o_hm = p/(1-p)
    print 'Visiting teams'
    print vs_ints
    print vs_atts
    p = float(vs_ints)/vs_atts
    print '{} or 1:{:0.1f}'.format(p, (1-p)/p)
    o_vs = p/(1-p)
    print 'HFO coefficent estimate'
    print 1/o_hm
    print 'Typical participant value'
    print np.sqrt(o_hm*o_vs)
# END interceptions

#==============================================================================
def get_TmStrentgh_dist():
    year = '2012'
    x = []
    for y in range(2003, 2013):
        #y = 2012
        games = clct.download_schedule_espn(y, 'final')
        tm_vals = rank.rw_rank(games, year, debug=True)
        x.extend(tm_vals.values())
    # END for
    
    print np.mean(x)
    print np.std(x)
    
    plt.hist(x)
    plt.show()
    plt.close()
# END get_TmStrength_dist

#==============================================================================
def get_PoEx_dist():
    all_games = []
    for y in range(2003, 2013):
        all_games.extend( clct.download_schedule_espn(y, 'final') )
    
    all_PoEx = []
    i = 0
    print all_games[0].keys()
    for gm in all_games:
        try:
            z = -4.579*(gm['hmto']-gm['vsto']) + 1.927
            z = gm['hmsc'] - gm['vssc'] - z
            all_games[i]['PoEx'] = z
            all_PoEx.append(z)
        except KeyError:
            print 'excluding game: {0[date]}, {0[vs]} at {0[hm]}'.format(gm)
        # END try
        i += 1
    # END for
    
    plt.hist(all_PoEx)
    plt.show()
    
    print 'mean = {0:0.2f} wins'.format(np.mean(all_PoEx))
    print 'sigma = {0:0.5f}'.format(np.std(all_PoEx))
    print 'skewness = {0:0.5e}'.format(skew(all_PoEx))
    print 'kurtosis = {0:0.5e}'.format(kurtosis(all_PoEx))
# END get_PoEx_dist

#==============================================================================
def scoring_corr():
    # Organize data
    all_games = []
    for y in range(2005, 2014):
        #all_games.extend( clct.download_schedule_espn(y, 'final') )
        pickle_fn = 'NFL_games_{}.pypickle'.format(y)
        f = open(pickle_fn, 'rb')
        all_games.extend( pickle.load(f) )
        f.close()
    # END for
    print '{} games'.format(len(all_games))
    
    X = []
    Y = []
    for gmobj in all_games:
        gm = gmobj.__dict__
        if 'hmto' not in gm or 'hmdefTD' not in gm: continue
        #X.append( gm['hmsc'] )
        #Y.append( gm['vssc'] )
        x = gm['hmsc'] - 4.579*gm['vsto'] - 1.927
        X.append(x)
        y = ( gm['vssc'] - 4.579*gm['hmto'] - 
              0.290*(gm['hmsc']-4.579*gm['vsto'])
            )
        Y.append(y)
    # END for
    print len(X)
    
    # Calculate correlation
    (r, p) = pearsonr(X, Y)
    print 'N = ' + str(len(X))
    print 'r = {0:0.3f} (p < {1:0.8f})'.format(r, p)
    
    # Fit data with linear regression
    (ar, br) = polyfit(X, Y, 1)
    print 'y = {0:0.8f}x + {1:0.8f}'.format(ar, br)
    X_ = sorted(X)
    xr = np.linspace(X_[0], X_[-1], 50)
    yr = polyval([ar,br], xr)
    
    # Visualize data
    plt.plot(X, Y, '.', xr, yr, '-r')
    plt.show()
    plt.close()
# END get_PoEx_dist

#==============================================================================
def get_wins_dist():
    Nties = 0
    Ngm = 0
    # Examine over range of seasons
    #for y in range(2009, 2014):
    #    for gm in games:
    #        if 17 < gm.wk:
    #            continue
    #        if gm.tie:
    #            Nties += 1
    #        # END if
    #        Ngm += 1
    # END for
    
    #print '{} ties over {} games ({:0.5f})'.format(
    #    Nties, Ngm, Nties / float(Ngm)
    #)
    
    ofS, dfS = season_compile()
    
    # Analysis of interception rates
    ointrates = logit( ofS[:,'int'] / ofS[:,'att'] )
    print 'atts/ints have {} keys'.format(len(ointrates))
    print 'should have {}'.format(32*5)
    lirm = np.mean(ointrates)
    lirs = np.std(ointrates)
    print 'mean logit(intrate): {:0.3f}'.format(lirm)
    print 'stdev logit(intrate): {:0.3f}'.format(lirs)
    print 'team lw stdev (/sqrt2): {:0.6f}'.format(lirs/np.sqrt(2))
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(ointrates, bins=24)
    ax[0].set_title('logit(int rates) for 2005 - 2013')
    probXY, fitstats = probplot(ointrates)
    ax[1].plot(probXY[0], probXY[1], '.k')
    xfit = np.linspace(min(probXY[0]), max(probXY[0]), 144)
    ax[1].plot(xfit, fitstats[0]*xfit + fitstats[1], '-r')
    ax[1].set_title('Probability plot, R^2= {:0.6f}'.format(fitstats[2]))
    fig.tight_layout()
    
    print ''
    X = (ofS[:,'sc'] - 7*ofS[:,'defTD']) / (ofS[:,'drives'] - ofS[:,'to'])
    Y = (-dfS[:,'sc'] + 7*dfS[:,'defTD']) / (dfS[:,'drives'] - dfS[:,'to'])
    oInt = logit( ofS[:,'int'] / ofS[:,'att'] )
    dInt = logit( dfS[:,'int'] / dfS[:,'att'] )
    Z = logit( ofS[:,'wpct'] )
    
    #print 'HFA for pnts per drive'
    #print 'Home Teams: {} / {} ({:0.2f})'.format( hpnts, hdrv,
    #                                              float(hpnts)/hdrv
    #                                            )
    #print 'Away Temas: {} / {} ({:0.2f})'.format( vpnts, vdrv,
    #                                              float(vpnts)/vdrv
    #                                            )
    #hfa = float(hpnts)/hdrv - float(vpnts)/vdrv
    #print 'Home - Away: {}'.format(hfa)
    #print ''
    
    m, b, r, p, std_err = linregress(X, Z)
    print 'off pnts only'
    print 'm = {}, b = {}'.format(m, b)
    print 'se = {}'.format(std_err)
    print 'R**2 = {:0.6f}, p = {:0.3f}\n'.format(r**2, p)
    
    m, b, r, p, std_err = linregress(Y, Z)
    print 'def pnts only'
    print 'm = {}, b = {}'.format(m, b)
    print 'se = {}'.format(std_err)
    print 'R**2 = {:0.6f}, p = {:0.3f}\n'.format(r**2, p)
    
    m, b, r, p, std_err = linregress(oInt, Z)
    print 'oInt rate'
    #r_self, p_self = pearsonr(tosplits[:,0], tosplits[:,1])
    #print 'self corr = {:0.3f} (p < {:0.3e})'.format(r_self, p_self)
    print 'm = {}, b = {}'.format(m, b)
    print 'se = {}'.format(std_err)
    print 'R**2 = {:0.6f}, p = {:0.3f}'.format(r**2, p)
    #print 'self_corr * win_corr = {:0.3f}'.format(r*r_self)
    print ''
    
    m, b, r, p, std_err = linregress(dInt, Z)
    print 'dInt rate'
    #r_self, p_self = pearsonr(tosplits[:,0], tosplits[:,1])
    #print 'self corr = {:0.3f} (p < {:0.3e})'.format(r_self, p_self)
    print 'm = {}, b = {}'.format(m, b)
    print 'se = {}'.format(std_err)
    print 'R**2 = {:0.6f}, p = {:0.3f}'.format(r**2, p)
    #print 'self_corr * win_corr = {:0.3f}'.format(r*r_self)
    print ''
    
    oFum = logit( (ofS[:,'to']-ofS[:,'int']) /
                   (ofS[:,'att']+ofS[:,'sk']+ofS[:,'runs'])
                )
    m, b, r, p, std_err = linregress(oFum, Z)
    print 'oFum'
    print 'm = {}, b = {}'.format(m, b)
    print 'se = {}'.format(std_err)
    print 'R**2 = {:0.6f}, p = {:0.3f}'.format(r**2, p)
    print ''
    
    dFum = logit( (dfS[:,'to']-dfS[:,'int']) /
                   (dfS[:,'att']+dfS[:,'sk']+dfS[:,'runs'])
                )
    m, b, r, p, std_err = linregress(dFum, Z)
    print 'dFum'
    print 'm = {}, b = {}'.format(m, b)
    print 'se = {}'.format(std_err)
    print 'R**2 = {:0.6f}, p = {:0.3f}'.format(r**2, p)
    print ''
    
    oPass = ofS[:,'pass'] / (ofS[:,'att'] + ofS[:,'sk'])
    m, b, r, p, std_err = linregress(oPass, Z)
    print 'oPass'
    print 'm = {}, b = {}'.format(m, b)
    print 'se = {}'.format(std_err)
    print 'R**2 = {:0.6f}, p = {:0.3f}'.format(r**2, p)
    print ''
    
    dPass = dfS[:,'pass'] / (dfS[:,'att'] + dfS[:,'sk'])
    m, b, r, p, std_err = linregress(dPass, Z)
    print 'dPass'
    print 'm = {}, b = {}'.format(m, b)
    print 'se = {}'.format(std_err)
    print 'R**2 = {:0.6f}, p = {:0.3f}'.format(r**2, p)
    print ''
    
    oR = ofS[:,'rush'] / ofS[:,'runs']
    m, b, r, p, std_err = linregress(oR, Z)
    print 'oR'
    print 'm = {}, b = {}'.format(m, b)
    print 'se = {}'.format(std_err)
    print 'R**2 = {:0.6f}, p = {:0.3f}'.format(r**2, p)
    print ''
    
    dR = dfS[:,'rush'] / dfS[:,'runs']
    m, b, r, p, std_err = linregress(dR, Z)
    print 'dR'
    print 'm = {}, b = {}'.format(m, b)
    print 'se = {}'.format(std_err)
    print 'R**2 = {:0.6f}, p = {:0.3f}'.format(r**2, p)
    print ''
    
    m, b, r, p, std_err = linregress(X+Y, Z)
    print 'off-def pnts'
    print 'm = {}, b = {}'.format(m, b)
    print 'se = {}'.format(std_err)
    print 'R**2 = {:0.6f}, p = {:0.3f}'.format(r**2, p)
    print ''
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(X, Z, '.')
    ax[0,0].set_title('off scoring')
    ax[0,1].plot(Y, Z, '.')
    ax[0,1].set_title('def scoring')
    ax[1,0].plot(oInt, Z, '.')
    ax[1,0].set_title('off int')
    ax[1,1].plot(dInt, Z, '.')
    ax[1,1].set_title('def int')
    
    from numpy.linalg import lstsq
    A = np.array( [X, Y] ).T
    c, chisq, _, _ = lstsq(A, Z)
    Rsqr = 1 - chisq[0] / np.sum((Z-np.mean(Z))**2)
    print 'opnts, dpnts'
    print 'logit(w/g) = {} * opnts + {} * dpnts'.format(*c)
    print 'cross-corr = {} (p < {:0.5f})'.format(*pearsonr(X, Y))
    print 'R**2 = {}'.format(Rsqr)
    print ''
    
    from numpy.linalg import lstsq
    B = np.ones((len(X),1))
    A = np.array( [oPass, oInt, dPass, dInt] ).T
    c, chisq, _, _ = lstsq(A, Z)
    Rsqr = 1 - chisq[0] / np.sum((Z-np.mean(Z))**2)
    print 'oPass, oInt, dPass, dInt'
    print c
    #print 'cross-corr = {} (p < {:0.5f})'.format(*pearsonr(X-Y, T))
    print 'R**2 = {}'.format(Rsqr)
    print ''
    
    pprint(type(ofS[0,'rush']))
    A = np.array( [Z, X, oPass, oInt, oR, oFum, Y, dPass, dInt, dR, dFum] )
    P = np.corrcoef(A)
    pprint(P.shape)
    tmin = Tdist.ppf(1-(0.05/2), len(X)-2)
    print tmin / np.sqrt(len(X)-2+tmin**2)
    tmin = Tdist.ppf(1-(0.01/2), len(X)-2)
    print tmin / np.sqrt(len(X)-2+tmin**2)
    labels = [
        'lw', 'opnts', 'oPass', 'oInt', 'oR', 'oFum', 'dpnts', 'dPass', 'dInt', 'dR', 'dFum'
    ]
    ptbl = LabeledTable(labels, labels, P)
    ptbl.write_csv('p table.csv')
    
    #Xdisp = np.linspace(min(X-Y), max(X-Y), 145)
    #plt.plot(X-Y, Z, 'o', Xdisp, m*Xdisp+b, '--r')
    #plt.show()
    #plt.close()
    
    #plt.hist(all_win_totals, bins=17)
    #plt.show()
    #plt.close()
    
    #print 'raw counts:'
    #print '    wins | count'
    #for n in sorted(win_dist.keys()):
    #    print '    {0:>4.1f} | {1:d}'.format(float(n), win_dist[n])
    #print 'mean = {0:0.2f} wins'.format(np.mean(all_win_totals))
    #sig = np.std(all_win_totals)
    #print 'sigma = {0:0.5f}'.format(sig)
    #print '    sigma/bino = {0:0.3f}'.format(sig/2.0)
    #print 'skewness = {0:0.5e}'.format(skew(all_win_totals))
    #krt = kurtosis(all_win_totals)
    #print 'kurtosis = {0:0.5e}'.format(krt)
    #print '    kurt/bino = {0:+0.3f}'.format(krt-(-1/8.0)/(-1/8.0))
    
    #plt.show()
    plt.close()
# END get_wins_dist

#==============================================================================
def season_compile():
    tms = []
    for tm in NFL.team_names:
        for y in range(2009, 2014):
            tms.append(tm + ' ' + str(y))
        # END for
    # END for
    
    ofS = LabeledTable(
        tms,
        [ 'sc', 'drives', 'to', 'defTD', 'pass', 'cmp', 'att', 'int',
          'sk', 'skyd', 'runs', 'rush', 'wpct'
        ]
    )
    dfS = LabeledTable(
        tms,
        [ 'sc', 'drives', 'to', 'defTD', 'pass', 'cmp', 'att', 'int',
          'sk', 'skyd', 'runs', 'rush', 'wpct'
        ]
    )
    # Examine over range of seasons
    for y in range(2009, 2014):
        ystr = ' '+str(y)
        pickle_fn = 'NFL_games_{}.pypickle'.format(y)
        f = open(pickle_fn, 'rb')
        games = pickle.load(f)
        f.close()
        print 'loaded {}'.format(pickle_fn)
        
        # Tally up W-L-T records (as len=3 lists)
        records = {}
        for gm in sorted(games, key=lambda g: g.date):
            hmyr = gm.hm + ystr
            vsyr = gm.vs + ystr
            
            if 17 < gm.wk:
                continue
            if hmyr not in records:
                records[hmyr] = WLTRecord()
            if vsyr not in records:
                records[vsyr] = WLTRecord()
            
            if not gm.tie:
                records[gm.winner + ystr].addW()
                records[gm.loser  + ystr].addL()
            else:
                records[gm.hm + ystr].addT()
                records[gm.vs + ystr].addT()
            # END if
            
            for k in ofS.cl[:-1]:
                try:
                    ofS[hmyr,k] += gm.hm_stat(k)
                    ofS[vsyr,k] += gm.vs_stat(k)
                    dfS[hmyr,k] += gm.vs_stat(k)
                    dfS[vsyr,k] += gm.hm_stat(k)
                except KeyError as err:
                    pprint(gm.__dict__)
                    raise err
                # END try
            # END for
        # END for
        
        for tmyr in records:
            ofS[tmyr,'wpct'] = records[tmyr].Wpct
            dfS[tmyr,'wpct'] = records[tmyr].Wpct
        # END for
    # END for
        
    ofS.write_csv('ofS.csv')
    dfS.write_csv('dfS.csv')
    
    return ofS, dfS
# END season_compile

#==============================================================================
class LabeledTable(object):
    '''
    '''
    
    def __init__(self, rows, cols, M=None):
        if M is None:
            self.M = np.zeros( (len(rows), len(cols)) )
        else:
            self.M = np.array(M)
        # END if
        try:
            self.rl = list(rows)
        except TypeError:
            self.rl = [str(i) for i in range(rows)]
        # END try
        self.rlkup = self._make_lkup(self.rl)
        try:
            self.cl = list(cols)
        except TypeError:
            self.cl = [str(i) for i in range(rows)]
        # END try
        self.clkup = self._make_lkup(self.cl)
    # END __init__
    
    @staticmethod
    def _make_lkup(L):
        return dict( zip(L, range(len(L))) )
    # END _make_lkup
    
    def __getitem__(self, key):
        r = key[0]
        c = key[1]
        if isinstance(r, str):
            r = self.rlkup[r]
        if isinstance(c, str):
            c = self.clkup[c]
        return self.M[r,c]
    # END __getitem__
    
    def __setitem__(self, key, value):
        r = key[0]
        c = key[1]
        if isinstance(r, str):
            r = self.rlkup[r]
        if isinstance(c, str):
            c = self.clkup[c]
        self.M[r,c] = value
    # END __setitem__
    
    def write_csv(self, save_path):
        ftxt = StringIO()
        ftxt.write(',')
        for j in range(len(self.cl)):
            ftxt.write(self.cl[j])
            if j < len(self.cl)-1:
                ftxt.write(',')
            # END if
        # END for
        ftxt.write('\n')
        for i in range(len(self.rl)):
            ftxt.write(self.rl[i])
            ftxt.write(',')
            for j in range(self.M.shape[1]):
                ftxt.write(str(self.M[i,j]))
                if j < len(self.cl)-1:
                    ftxt.write(',')
                # END if
            # END for
            ftxt.write('\n')
        # END for
        
        with open(save_path, 'w') as f:
            f.write(ftxt.getvalue())
        # END with
    # END write_table
# END labeledTable

#==============================================================================
def logit(p): return np.log(p/(1.0-p))
def invlogit(x): return 1.0 / (1.0 + np.exp(-x))

#==============================================================================
if __name__ == '__main__':
    main()
# END if