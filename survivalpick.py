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
import random
from math import log, exp
from pprint import pprint

# custom modules
from NFL_class import NFL

#==============================================================================
def main():
    solver = SuvivalPicksSolver('2014_remaining_schd.csv')
    pks = solver.get_solution()
    print pks
# END main

#==============================================================================
class Team(object):
    '''
    '''
    
    def __init__(self, name, lgwp=None, gwp=None):
        self.name = name
        if lgwp is not None:
            self.lgwp = lgwp
            self.gwp = invlogit(lgwp)
        elif gwp is not None:
            self.gwp = gwp
            self.lgwp = logit(gwp)
        else:
            self.lgwp = 0
            self.gwp = 0.5
        # END if
    # END __init__
    
    def __hash__(self):
        return hash(self.name)
    # END __hash__
    
    def __str__(self):
        return self.name
    # END __str__
# END Team

#==============================================================================
class Game(object):
    '''
    '''
    
    def __init__(self, hm, vs, wk, hosted=True):
        self.hmtm = hm
        self.vstm = vs
        self.wk = wk
        self.hm_wp = Game.predict_game(hm, vs, hosted)
    # END __init__
    
    def __str__(self):
        outstr = '<Game: wk{0:02d}, {1} ({2:0.2f}) @ {3} ({4:0.2f})>'.format(
            self.wk+1, self.vstm, 1-self.hm_wp, self.hmtm, self.hm_wp
        )
        return outstr
    # END __str__
    
    @property
    def winner_wp(self):
        if self.hm_wp >= 0.5:
            return self.hm_wp
        else:
            return 1 - self.hm_wp
        # END if
    # END get_winner_wp
    
    @property
    def exp_winner(self):
        if self.hm_wp >= 0.5:
            return self.hmtm
        else:
            return self.vstm
        # END if
    # END exp_winner
    
    @property
    def exp_looser(self):
        if self.hm_wp < 0.5:
            return self.hmtm
        else:
            return self.vstm
        # END if
    # END exp_looser
    
    @staticmethod
    def predict_game(hm, vs, hosted):
        hm_lwp = 0
        if hosted:
            hm_lwp = log(1.27)
        hm_lwp += hm.lgwp - vs.lgwp
        return invlogit(hm_lwp)
    # END predict_game
# END Game

#==============================================================================
class Schedule(object):
    '''
    '''
    
    def __init__(self, gms=[], wks=[]):
        self.all_games = tuple(gms)
        self.weeks = tuple(wks)
    # END __init__
    
    @classmethod
    def import_csv(cls, file_path):
        '''Example line:
            1,Thu,September 5,Baltimore Ravens,@,Denver Broncos,8:30 PM
        '''
        
        #f = open('up-to-date_ranks.csv')
        f = open('2014_preseason_rankings.csv')
        tm_lgwp = {}
        for ln in f:
            vals = ln.split(',')
            if len(vals) != 2:
                print ln
                print vals
            tm_abv = vals[0]
            lg = vals[1]
            tm_name = NFL.normalize_name(tm_abv)
            tm_lgwp[tm_name] = float(lg)
        # END for
        f.close()
        
        # Debug
        #print 'Using the following rankings:'
        #r = 0
        #for tm in sorted(tm_lgwp.keys(), key=lambda k: tm_lgwp[k])[::-1]:
        #    r += 1
        #    print '    {:>2d} {:.<24s} {:0.2f}'.format(r, tm, tm_lgwp[tm])
        #print ''
        
        gms = []
        wks = []
        last_iwk = 0
        with open(file_path) as f:
            for ln in f:
                vals = ln.split(',')
                iwk = int(vals[0])
                if iwk > last_iwk:
                    wks.append([])
                    last_iwk = iwk
                hmtm = Team(vals[5], lgwp=tm_lgwp[vals[5]])
                vstm = Team(vals[3], lgwp=tm_lgwp[vals[3]])
                g = Game(hmtm, vstm, iwk)
                wks[-1].append(g)
            # END for
        # END with
        
        return cls(gms, wks)
    # END import_csv
# END Schedule

#==============================================================================
class PickSet(object):
    '''
    '''
    
    used_teams = [
    ]
    
    def __init__(self, picks=None):
        if picks is None:
            self.pick_list = []
            self.selected_teams = set(self.used_teams)
            self._estrk = (hash(self), 0.0)
            self._psucc = (hash(self), 0.0)
        else:
            self.pick_list = list(picks.pick_list)
            self.selected_teams = set(picks.selected_teams)
            self._estrk = picks._estrk
            self._psucc = picks._psucc
        # END if
    # END __init__
    
    def __getitem__(self, i):
        return self.pick_list[i]
    # END __getitem__
    
    def __hash__(self):
        hash_ = []
        for g in self.pick_list:
            hash_.append(str(g.wk))
            hash_.append(str(g.hmtm))
            hash_.append(str(g.vstm))
        return hash(tuple(hash_))
    # END __hash__
    
    def __len__(self):
        return len(self.pick_list)
    # END __len__
    
    def __str__(self):
        outstr = ''
        for g in sorted(self.pick_list, key=lambda g: g.wk):
            outstr += 'wk{0:02d}: {1} over {2}, p = {3:0.2f}\n'.format(
                g.wk, g.exp_winner, g.exp_looser, g.winner_wp
            )
        # END for
        try:
            outstr += 'E[strikes] = {:0.3f}\n'.format(self.exp_strikes)
            outstr += 'P[strikes < 3] = {:0.3f}\n'.format(self.p_success)
        except IndexError:
            pass
        # END try
        return outstr
    # END __str__
    
    @property
    def exp_strikes(self):
        if self._estrk[0] == hash(self):
            return self._estrk[1]
        
        s = 0
        for gm in self.pick_list:
            s += 1-gm.winner_wp
        self._estrk = (hash(self), s)
        return s
    # END exp_strikes
    
    @property
    def p_success(self):
        if self._psucc[0] == hash(self):
            return self._psucc[1]
        
        p = 0
        for Nfails in range(3):
            combos = self._p_success_get_combos(Nfails)
            for c in combos:
                p_ = 1.0
                for g in self.pick_list:
                    if g in c:
                        p_ *= 1 - g.winner_wp
                    else:
                        p_ *= g.winner_wp
                    # END if
                # END for
                p += p_
            # END for
        # END for
        
        self._psucc = (hash(self), p)
        return p
    # END p_success
    
    @classmethod
    def set_used_teams(cls, tms):
        cls.used_teams = list(tms)
    # END set_used_teams
    
    def _p_success_get_combos(self, N=0, combos=[set(),]):
        if N == 0:
            return combos
        elif N > 0:
            c = combos.pop(0)
            for g in self.pick_list:
                if g not in c:
                    combos.append(c | set([g]))
            # END for
            return self._p_success_get_combos(N-1, combos)
        else:
            raise ValueError('N must be > 0')
        # END if    
    # END _p_success_get_combos
    
    def add(self, gm):
        if gm.exp_winner.name in self.selected_teams:
            return False
        
        self.selected_teams.add(gm.exp_winner.name)
        if len(self.pick_list) == 0 or self.pick_list[-1].wk < gm.wk:
            self.pick_list.append(gm)
        elif gm.wk < self.pick_list[0].wk:
            self.pick_list.insert(0, gm)
        else:
            a = 0
            b = len(self.pick_list)
            while a+1 != b:
                i = (a+b) / 2
                if gm.wk < b:
                    b = i
                else:
                    a = i
            # END while
            self.pick_list.insert(b, gm)
        # END if
        return True
    # END add
# END PickSet

#==============================================================================
class SuvivalPicksSolver(object):
    '''
    '''
    
    def __init__(self, schd_file):
        self.schd = Schedule.import_csv(schd_file)
        print 'schedule loaded'
    # END __init__
    
    def get_solution(self):
        # arbitrily high number
        fewest_strikes = 18
        
        # initialize some random solutions
        Ncans = 18
        curr_gen = []
        W = range(len(self.schd.weeks))
        for i in range(Ncans):
            random.shuffle(W)
            pks = PickSet()
            for i in W:
                wk = self.schd.weeks[i]
                wk.sort(key=lambda g: g.winner_wp, reverse=True)
                while not pks.add(wk[0]):
                    wk = wk[1:]
            # END for
            if len(pks) != len(self.schd.weeks):
                print 'ping s1'
            curr_gen.append(pks)
            
            curr_gen.append( self.rand_PickSet() )
        # END for
        
        for pks in curr_gen:
            if len(pks) != len(self.schd.weeks):
                print 'ping q1'
        # END for
        
        N_rnds = 10000
        N_gen_limit = 6
        last_rnd_best = fewest_strikes+1
        for i_rnd in range(N_rnds):
            curr_gen.sort(key=lambda ps: ps.exp_strikes)
            fewest_strikes = curr_gen[0].exp_strikes
            if fewest_strikes < last_rnd_best:
                print 'round {:d}: {:0.3f} strikes'.format(
                    i_rnd, fewest_strikes
                )
            # END if
            #if abs(fewest_strikes - last_rnd_best) <= 1.0E-3:
            #    break
            # END if
            
            # fittest continue
            next_gen = list(curr_gen[:N_gen_limit])
            random.shuffle(next_gen)
            # add variants on fittest and random mergers of fittest
            n = len(next_gen)
            for j in range(n):
                next_gen.append( self.rand_variation(next_gen[j]) )
                next_gen.append( 
                    self.rand_merge(next_gen[j], next_gen[(j+1)%n])
                )
                # END if
                next_gen.append( self.rand_PickSet() )
            # END for
            curr_gen = next_gen
            last_rnd_best = fewest_strikes
        # END for
        print 'finished on round {}'.format(i_rnd)
        
        return curr_gen[0]
    # END get_solution
    
    def rand_PickSet(self):
        pks = PickSet()
        for wk in self.schd.weeks:
            while not pks.add( random.choice(wk) ):
                pass
        # END for
        if len(pks) != len(self.schd.weeks):
            print 'ping r1'
        return pks
    # END rand_PickSet
    
    def rand_variation(self, pks):
        if len(pks) != len(self.schd.weeks): raise RuntimeError()
        new_pks = PickSet()
        rand_wk = random.randint(0, len(pks)-1)
        err_str = str(len(pks)) + '\n'
        err_str += str(len(new_pks)) + '\n'
        for i in range(len(pks)):
            if i == rand_wk:
                continue
            new_pks.add( pks[i] )
            err_str += str(len(new_pks)) + '\n'
        # END for
        while not new_pks.add( random.choice(self.schd.weeks[rand_wk]) ):
            continue
        err_str += str(len(new_pks)) + '\n'
        if len(new_pks) != len(self.schd.weeks):
            print err_str
            print 'ping1'
        return new_pks
    # END rand_variation
    
    def rand_merge(self, pks1, pks2):
        if ( len(pks1) != len(self.schd.weeks)
             or len(pks2) != len(self.schd.weeks) ):
            raise RuntimeError()
        # END if
        new_pks = PickSet()
        err_str = str(len(pks1)) + ', ' + str(len(pks2)) + '\n'
        err_str += str(len(new_pks)) + '\n'
        for i in range(len(pks1)):
            if random.randint(0, 1) == 0:
                if not new_pks.add( pks1.pick_list[i] ):
                    if not new_pks.add( pks2.pick_list[i] ):
                        rand_gm = random.choice(self.schd.weeks[i])
                        while not new_pks.add(rand_gm):
                            rand_gm = random.choice(self.schd.weeks[i])
                    # END if
                # END if
            # END if
            else:
                if not new_pks.add( pks2.pick_list[i] ):
                    if not new_pks.add( pks1.pick_list[i] ):
                        rand_gm = random.choice(self.schd.weeks[i])
                        while not new_pks.add(rand_gm):
                            rand_gm = random.choice(self.schd.weeks[i])
                    # END if
                # END if
            # END if
            err_str += str(len(new_pks)) + '\n'
        # END for
        if len(new_pks) != len(self.schd.weeks):
            print err_str
            print 'ping3'
        return new_pks
    # END rand_merge
# END SurvivalPicksSolver

#==============================================================================
def logit(p):
    return log(p / (1-p))
def invlogit(w):
    return 1.0 / (1.0 + exp(-w))

#==============================================================================
if __name__ == '__main__':
    main()
# END if