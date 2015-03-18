#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''NFL Class Module
    
    This module only contains code for the NFL class.  This class will be used
    to store all of the universal NFL data
    
    List of classes:
        NFL
    List of functions: -none-
'''

# built-in modules
from pprint import pprint

# third-party modules
import numpy as np

# self package

#===============================================================================
class NFL(object):
    '''National Football League (NFL)
    
    Class Attributes:
        teams (list): All normalized full-length team names
        _tm_aliases (dict)
        conferences (list)
        divisions (list)
        _tm_groups (dict)
        home_field_odds
    Class Methods:
        as_seasonyear
        get_tmgroup
        normalize_name
        shorten_name
    Instantiation Args: -none-
    '''
    
    # Class attributes
    #-----------------
    team_names = [
        "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens",
        "Buffalo Bills", "Carolina Panthers", "Chicago Bears",
        "Cincinnati Bengals", "Cleveland Browns", "Dallas Cowboys",
        "Denver Broncos", "Detroit Lions", "Green Bay Packers",
        "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
        "Kansas City Chiefs", "Miami Dolphins", "Minnesota Vikings",
        "New England Patriots", "New Orleans Saints", "New York Giants",
        "New York Jets", "Oakland Raiders", "Philadelphia Eagles",
        "Pittsburgh Steelers", "San Diego Chargers", "San Francisco 49ers",
        "Seattle Seahawks", "St. Louis Rams", "Tampa Bay Buccaneers",
        "Tennessee Titans", "Washington Redskins"
    ]
    _tm_aliases = {
        "AVG": "Average Joes",
        "ARI": "Arizona Cardinals",
        "ATL": "Atlanta Falcons",
        "BAL": "Baltimore Ravens",
        "BUF": "Buffalo Bills",
        "CAR": "Carolina Panthers",
        "CHI": "Chicago Bears",
        "CIN": "Cincinnati Bengals",
        "CLE": "Cleveland Browns",
        "DAL": "Dallas Cowboys",
        "DEN": "Denver Broncos",
        "DET": "Detroit Lions",
        "GNB": "Green Bay Packers",
        "GB": "Green Bay Packers",
        "GRB": "Green Bay Packers",
        "HOU": "Houston Texans",
        "IND": "Indianapolis Colts",
        "JAC": "Jacksonville Jaguars",
        "JAX": "Jacksonville Jaguars",
        "KAN": "Kansas City Chiefs", "KC": "Kansas City Chiefs",
        "MIA": "Miami Dolphins",
        "MIN": "Minnesota Vikings",
        "NWE": "New England Patriots", "NE": "New England Patriots",
        "NOR": "New Orleans Saints", "NO": "New Orleans Saints",
        "NYG": "New York Giants",
        "NYJ": "New York Jets",
        "OAK": "Oakland Raiders",
        "PHI": "Philadelphia Eagles",
        "PIT": "Pittsburgh Steelers",
        "SDG": "San Diego Chargers", "SD": "San Diego Chargers",
        "SFO": "San Francisco 49ers", "SF": "San Francisco 49ers",
        "SEA": "Seattle Seahawks",
        "STL": "St. Louis Rams",
        "TAM": "Tampa Bay Buccaneers",
        "TPA": "Tampa Bay Buccaneers",
        "TB": "Tampa Bay Buccaneers",
        "TEN": "Tennessee Titans",
        "WAS": "Washington Redskins",
        
        "Cardinals": "Arizona Cardinals",
        "Falcons": "Atlanta Falcons",
        "Ravens": "Baltimore Ravens",
        "Bills": "Buffalo Bills",
        "Panthers": "Carolina Panthers",
        "Bears": "Chicago Bears",
        "Bengals": "Cincinnati Bengals",
        "Browns": "Cleveland Browns",
        "Cowboys": "Dallas Cowboys",
        "Broncos": "Denver Broncos",
        "Lions": "Detroit Lions",
        "Packers": "Green Bay Packers",
        "Texans": "Houston Texans",
        "Colts": "Indianapolis Colts",
        "Jaguars": "Jacksonville Jaguars",
        "Chiefs": "Kansas City Chiefs",
        "Dolphins": "Miami Dolphins",
        "Vikings": "Minnesota Vikings",
        "Patriots": "New England Patriots",
        "Saints": "New Orleans Saints",
        "Giants": "New York Giants",
        "Jets": "New York Jets",
        "Raiders": "Oakland Raiders",
        "Eagles": "Philadelphia Eagles",
        "Steelers": "Pittsburgh Steelers",
        "Chargers": "San Diego Chargers",
        "49ers": "San Francisco 49ers",
        "Seahawks": "Seattle Seahawks",
        "Rams": "St. Louis Rams",
        "Buccaneers": "Tampa Bay Buccaneers",
        "Titans": "Tennessee Titans",
        "Redskins": "Washington Redskins",
        
        "Average Joes": "AVG",
        "Arizona Cardinals": "ARI",
        "Atlanta Falcons": "ATL",
        "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR",
        "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN",
        "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN",
        "Detroit Lions": "DET",
        "Green Bay Packers": "GNB",
        "Houston Texans": "HOU",
        "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAC",
        "Kansas City Chiefs": "KAN",
        "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN",
        "New England Patriots": "NWE",
        "New Orleans Saints": "NOR",
        "New York Giants": "NYG",
        "New York Jets": "NYJ",
        "Oakland Raiders": "OAK",
        "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT",
        "San Diego Chargers": "SDG",
        "San Francisco 49ers": "SFO",
        "Seattle Seahawks": "SEA",
        "St. Louis Rams": "STL",
        "Tampa Bay Buccaneers": "TAM",
        "Tennessee Titans": "TEN",
        "Washington Redskins": "WAS"
    }
    _tm_abreviations = {
        "Average Joes": "AVG",
        "Arizona Cardinals": "ARI",
        "Atlanta Falcons": "ATL",
        "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR",
        "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN",
        "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN",
        "Detroit Lions": "DET",
        "Green Bay Packers": "GRB", #alternate is "GNB"
        "Houston Texans": "HOU",
        "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX", #alternate is "JAX"
        "Kansas City Chiefs": "KAN",
        "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN",
        "New England Patriots": "NWE",
        "New Orleans Saints": "NOR",
        "New York Giants": "NYG",
        "New York Jets": "NYJ",
        "Oakland Raiders": "OAK",
        "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT",
        "San Diego Chargers": "SDG",
        "San Francisco 49ers": "SFO",
        "Seattle Seahawks": "SEA",
        "St. Louis Rams": "STL",
        "Tampa Bay Buccaneers": "TPA", #alternate is "TAM"
        "Tennessee Titans": "TEN",
        "Washington Redskins": "WAS"
    }
    conferences = ['AFC', 'NFC']
    divisions = [
        'AFC North', 'AFC East', 'AFC South', 'AFC West',
        'NFC North', 'NFC East', 'NFC South', 'NFC West'
    ]
    _tm_groups = {
        'AFC': [ "Baltimore Ravens", "Buffalo Bills", "Cincinnati Bengals",
                 "Cleveland Browns", "Denver Broncos", "Houston Texans",
                 "Indianapolis Colts", "Jacksonville Jaguars",
                 "Kansas City Chiefs", "Miami Dolphins",
                 "New England Patriots", "New York Jets", "Oakland Raiders",
                 "Pittsburgh Steelers", "San Diego Chargers",
                 "Tennessee Titans"
               ],
        'NFC': [ "Arizona Cardinals", "Atlanta Falcons", "Carolina Panthers",
                 "Chicago Bears", "Dallas Cowboys", "Detroit Lions",
                 "Green Bay Packers", "Minnesota Vikings",
                 "New Orleans Saints", "New York Giants",
                 "Philadelphia Eagles", "San Francisco 49ers",
                 "Seattle Seahawks", "St. Louis Rams", "Tampa Bay Buccaneers",
                 "Washington Redskins"
               ],
        'AFC North': [ "Baltimore Ravens", "Cincinnati Bengals", 
                       "Cleveland Browns", "Pittsburgh Steelers"
                     ],
        'AFC East': [ "Buffalo Bills", "Miami Dolphins",
                      "New England Patriots", "New York Jets"
                    ],
        'AFC South': [ "Houston Texans", "Indianapolis Colts",
                       "Jacksonville Jaguars", "Tennessee Titans"
                     ],
        'AFC West': [ "Denver Broncos", "Kansas City Chiefs",
                      "Oakland Raiders", "San Diego Chargers"
                    ],
        'NFC North': [ "Chicago Bears", "Detroit Lions",
                       "Green Bay Packers", "Minnesota Vikings"
                     ],
        'NFC East': [ "Dallas Cowboys", "New York Giants",
                      "Philadelphia Eagles", "Washington Redskins"
                    ],
        'NFC South': [ "Atlanta Falcons", "Carolina Panthers",
                       "New Orleans Saints", "Tampa Bay Buccaneers"
                     ],
        'NFC West': [ "Arizona Cardinals", "San Francisco 49ers",
                      "Seattle Seahawks", "St. Louis Rams"
                    ],
    }
    _tm_divisions = {
        "Arizona Cardinals": "NFC West",
        "Atlanta Falcons": "NFC South",
        "Baltimore Ravens": "AFC North",
        "Buffalo Bills": "AFC East",
        "Carolina Panthers": "NFC South",
        "Chicago Bears": "NFC North",
        "Cincinnati Bengals": "AFC North",
        "Cleveland Browns": "AFC North",
        "Dallas Cowboys": "NFC East",
        "Denver Broncos": "AFC West",
        "Detroit Lions": "NFC North",
        "Green Bay Packers": "NFC North",
        "Houston Texans": "AFC South",
        "Indianapolis Colts": "AFC South",
        "Jacksonville Jaguars": "AFC South",
        "Kansas City Chiefs": "AFC West",
        "Miami Dolphins": "AFC East",
        "Minnesota Vikings": "NFC North",
        "New England Patriots": "AFC East",
        "New Orleans Saints": "NFC South",
        "New York Giants": "NFC East",
        "New York Jets": "AFC East",
        "Oakland Raiders": "AFC West",
        "Philadelphia Eagles": "NFC East",
        "Pittsburgh Steelers": "AFC North",
        "San Diego Chargers": "AFC West",
        "San Francisco 49ers": "NFC West",
        "Seattle Seahawks": "NFC West",
        "St. Louis Rams": "NFC West",
        "Tampa Bay Buccaneers": "NFC South",
        "Tennessee Titans": "AFC South",
        "Washington Redskins": "NFC East"
    }
    home_field_odds = 1.27
    
    def _init_(self):
        pass
    # END _init_
    
    # Class-specific methods
    #-----------------------
    @classmethod
    def as_seasonyear(cls, d):
        '''Convert Date to Season Formal Year
        
        Given a date this method will return the year that the season containing
        the given date began.
        
        Args:
            d (str|int): A date formatted as "yyyymmdd" or "yyyymm"
        Returns:
            (int) Formal season year.  Will return None if the date is out-of-
            season.
        '''
        
        # normalize type and value of d
        if isinstance(d, int): d = str(d)
        if len(d) < 6:
            raise ValueError('"{}" is not a valid date'.format(d))
        elif len(d) > 6:
            d = d[:6]
        # END if
        
        seasonyear = int(d[:4])
        month = int(d[4:6])
        
        if month > 12:
            raise ValueError('"{}" is not a valid date'.format(d))
        elif month >= 9:
            return seasonyear
        elif month <= 2:
            return seasonyear - 1
        else:
            return None
        # END if
    # END as_seasonyear
    
    @classmethod
    def get_tmgroup(cls, name):
        '''Get Team Grouping
        
        This methods will return a list of teams names for a given division
        or conference in the NFL
        
        Args:
            group_name (str): Acceptible names are AFC, NFC, AFC North, etc.
        Returns:
            (list)
        '''
        if name in cls._tm_groups:
            return cls._tm_groups[name]
        elif name in cls._tm_divisions:
            return cls._tm_divisions[name]
        else:
            return ''
            raise ValueError(
                'Invald input name, should be team or group name; ' +
                'got "{}"'.format(name)
            )
        # END if
    # END get_tmgroup
    
    @classmethod
    def normalize_name(cls, name):
        '''Team Name Normalizer
        
        This method will return a normalized full team name given an
        abbreviation or alias.
        
        Args:
            name (str)
        Returns:
            (str)
        Example:
            >>> NFL.normalize_name("Packers")
            >>> "Green Bay Packers"
            >>> NFL.normalize_name("GB")
            >>> "Green Bay Packers"
            >>> NFL.normalize_name("Green Bay Packers")
            >>> "GNB"
        '''
        name = str(name)
        if name in cls.team_names:
            return name
        else:
            return cls._tm_aliases[name]
        # END if
    # END normalize_name
    
    @classmethod
    def shorten_name(cls, name):
        '''Team Name Shortener
        
        This method will return a 3-letter team abbreviation given a
        normalized full team name.
        '''
        
        if len(name) == 3 and name in cls._tm_aliases:
            return name
        else:
            return cls._tm_abreviations[name]
        # END if
    # END shorten_name
    
    @classmethod
    def get_league_instance(cls):
        return [Team(tn) for tn in cls.team_names]
    # END get_league_instance
# END NFL

#==============================================================================
class Schedule(object):
    '''
    '''
    
    def __init__(self):
        self.all_gms #?
        self.gms_rounds = []
        self.tm_schd_lkup = {}
    # END __init__
    
    def __iter__(self):
        pass
    # END __iter__
    
    def __len__(self):
        return len(self.all_gms)
    # END __len__
# END Schedule

#==============================================================================
class Season(object):
    '''Single Season Class
    
    Instance Attributes:
        schd (?): schedule
    '''
    
    def __init__(self):
        pass
    # END __init__
    
    
# END Season

#==============================================================================
class Game(object):
    '''Single Game Class
    
    Instance Attributes:
        "final": (bool)
        "date": (int) in yyyymmdd format
        "wk": (int) NFL week number
        "hosted": (bool)
        "vs": (str) normalized name of visiting team
        "hm": (str) normalized name of home team
        "vssc": (int) final score of the visiting team
        "hmsc": (int) final score of the home team
        "vsdrives": (int) number of drives by the visiting team
        "hmdrives": (int) number of drives by the home team
        "vsto": (int) number of turnovers by the visiting team
        "hmto": (int) number of turnovers by the home team
        "vsdefTD": (int) number of TD scored by visitor defense or special
                         teams
        "hmdefTD": (int) number of TD scored by visitor defense or special
                         teams
        "vs_pass": (int) number of passing yards by the visiting team
        "hm_pass": (int) number of passing yards by the home team
        "vs_cmp": (int) number of passing comp. by the visiting team
        "vs_att": (int) number of passing att. by the visiting team
        "hm_cmp": (int) number of passing comp. by the home team
        "hm_att": (int) number of passing att. by the home team
        "hm_int": (int) number of interception thrown by the home offense
        "vs_int": (int) ...
        "hm_sk": (int) number of sacks taken by home team
        "hm_skyd": (int) number of yards lost to sacks taken by home team
        "vs_sk": (int) number of sacks taken by visiting team
        "vs_skyd": (int) number of yards lost to sacks taken by visiting team
    '''
    
    valid_atts = ['final', 'date', 'wk', 'hosted', 'vs', 'hm']
    
    valid_stats = [
        "vssc", "hmsc", "vsdrives", "hmdrives", "vsto", "hmto", "vsdefTD",
        "hmdefTD", "vs_pass", "hm_pass", "vs_cmp", "vs_att", "hm_cmp",
        "hm_att", "hm_rush", "vs_rush", "hm_runs", "vs_runs", "hm_int", "vs_int", "hm_sk", "hm_skyd", "vs_sk", "vs_skyd"
    ]
    
    def __init__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) > 0:
            # Process keyword arguments as key-value pairs for game dict
            for k in kwargs:
                if k in Game.valid_stats or k in Game.valid_atts:
                    super(Game, self).__setattr__(k, kwargs[k])
            # END for
        elif len(args) == 1 and isinstance(args[0], dict):
            # Input argument is a single dictionary, make this the game dict
            for k in args[0]:
                if k in Game.valid_stats or k in Game.valid_atts:
                    super(Game, self).__setattr__(k, args[0][k])
            # END for
        else:
            raise TypeError('Invalid input for Game class')
        # END if
        
        if 'final' not in self.__dict__: self.final = False
        if 'hosted' not in self.__dict__: self.hosted = True
        self.__winner = None
        self.__loser = None
        self.__tie = None
    # END __init__
    
    # Property methods
    #-----------------
    @property
    def winner(self):
        if not self.final:
            raise RuntimeError(
                'Cannot determine winner of game that is not final'
            )
        # END if
        if self.__winner is None and self.__tie is None:
            if self.vssc < self.hmsc:
                self.__winner = self.hm
                self.__loser = self.vs
                self.__tie = False
                return self.winner
            elif self.hmsc < self.vssc:
                self.__winner = self.vs
                self.__loser = self.hm
                self.__tie = False
            else:
                self.__tie = True
            # END if
            return self.winner
        else:
            return self.__winner
        # END if
    # END winner
    
    @property
    def loser(self):
        if not self.final:
            raise RuntimeError(
                'Cannot determine winner of game that is not final'
            )
        # END if
        if self.__loser is None and self.__tie is None:
            if self.vssc < self.hmsc:
                self.__winner = self.hm
                self.__loser = self.vs
                self.__tie = False
                return self.winner
            elif self.hmsc < self.vssc:
                self.__winner = self.vs
                self.__loser = self.hm
                self.__tie = False
            else:
                self.__tie = True
            # END if
            return self.loser
        else:
            return self.__loser
        # END if
    # END loser
    
    @property
    def tie(self):
        if not self.final:
            raise RuntimeError(
                'Cannot determine winner of game that is not final'
            )
        # END if
        if self.__tie is None:
            if self.vssc < self.hmsc:
                self.__winner = self.hm
                self.__loser = self.vs
                self.__tie = False
            elif self.hmsc < self.vssc:
                self.__winner = self.vs
                self.__loser = self.hm
                self.__tie = False
            else:
                self.__tie = True
            # END if
            return self.tie
        else:
            return self.__tie
        # END if
    # END tie
    
    # Magic Methods
    #--------------
    def __contains__(self, tm):
        if tm == self.hm or tm == self.vs:
            return True
        else:
            return False
        # END if
    # END __contains__
    
    def __format__(self, fmtstr):
        if fmtstr == '' or fmtstr == 's':
            return self.__to_str()
        elif fmtstr == 'l':
            return self.__to_str(short=False)
        else:
            raise ValueError(
                'Unknown format code {} for object of type {}'.format(
                    fmtstr, type(self).__name__
                )
            )
        # END if
    # END __format__
    
    def __to_str(self, short=True):
        at_vs = 'vs.'
        if self.hosted: at_vs = '@'
        final_flag = ''
        if self.final and short: final_flag = '-F'
        vs_name = self.vs
        hm_name = self.hm
        if short:
            vs_name = NFL.shorten_name(str(self.vs))
            hm_name = NFL.shorten_name(str(self.hm))
        # END if
        vs_sc = ''
        hm_sc = ''
        if not short:
            vs_sc = ' {:>2d}'.format(self.vssc)
            hm_sc = ' {:>2d}'.format(self.hmsc)
        # END if
        
        out = '{}{}: {}{} {} {}{}'.format(
            self.date, final_flag,
            vs_name, vs_sc, at_vs, hm_name, hm_sc
        )
        return out
    # END __to_str
    def __str__(self): return self.__to_str()
    
    # Game-specific methods
    #----------------------
    def predict(self):
        '''Predict outcome for the game
        
        Returns:
            (float) Probability of the home team winning
        '''
        
        hfo = NFL.home_field_odds
        if not self.hosted: hfo = 1.0
        return hfo*self.hm.v / (hfo*self.hm.v + self.vs.v)
    # END predict
    
    def value(self):
        '''Game valuing method
        
        This version is based of a logistic regression of win% vs TO-adj
        points
        ***Model Changed***
        Based on correlation of TO-free & DefTD-free pnt diff per driv
        to logit(wins/games)
        m = 1.4445786913, b = ?
        R**2 = 0.780067, (p < 0.000)
        
        Returns:
            (float) Outcome weighting relative to home team (i.e. home
            incoming edge weight, or "V2H")
        '''
        
        #z = -4.579*(self.hmto-self.vsto) + 1.9272
        #z = self.hmsc - self.vssc - z
        #return invlogit(0.0698*z + 0.0038)
        hmpnts = float(self.hmsc-self.hmdefTD) / (self.hmdrives-self.hmto)
        vspnts = float(self.vssc-self.vsdefTD) / (self.vsdrives-self.vsto)
        return invlogit( 1.4445786913*(hmpnts-vspnts) - 0.006108530416 )
    # END value
    
    def stat(self, k):
        return self.__dict__[k]
    # END stat
    
    def hm_stat(self, k):
        try:
            return self.__dict__[k]
        except KeyError:
            pass
        # END try
        try:
            return self.__dict__['hm'+k]
        except KeyError:
            pass
        # END try
        return self.__dict__['hm_'+k]
    # END hm_stat
    
    def vs_stat(self, k):
        try:
            return self.__dict__[k]
        except KeyError:
            pass
        # END try
        try:
            return self.__dict__['vs'+k]
        except KeyError:
            pass
        # END try
        return self.__dict__['vs_'+k]
    # END vs_stat
    
    # Test method(s)
    #---------------
    @classmethod
    def full_test(cls):
        print 'Testing {} class...'.format(cls)
        print 'creating example instance from dict'
        d = {
            "final": True,
            "date": 20130101, "wk": 12,
            "hosted": True,
            "vs": 'Green Bay Packers', "hm": 'Chicago Bears',
            "vssc": 22, "hmsc": 14,
            "vsto": 0, "hmto": 1
        }
        print 'game_dict = '
        pprint(d, indent=4)
        g1 = cls(d)
        print ''
        print 'type(g1) = ' + repr(type(g1))
        print 'dir(g1) = '
        pprint(dir(g1), indent=4)
        print 'print g1 --> "{0}"'.format(g1)
        print 'print long g1 --> "{0:l}"'.format(g1)
        print 'g1.winner = ' + g1.winner
        print 'g1.loser = ' + g1.loser
        print 'g1.tie = ' + str(g1.tie)
        try:
            print 'g1.predict() = {}'.format(g1.predict())
        except AttributeError:
            print 'predict() failed from team being a str instead of a Team'
        # END try
        print 'g1.value() = {}'.format(g1.value())
        
        print 'creating example instance from keyword args'
        d = {
            "final": True,
            "date": 20130101, "wk": 12,
            "hosted": False,
            "vs": 'Arizona Cardinals', "hm": 'Seattle Seahawks',
            "vssc": 22, "hmsc": 14,
            "vsto": 0, "hmto": 1
        }
        g1 = cls(**d)
        print 'print g1 --> "{0:l}"'.format(g1)
        
        print 'Game testing complete.'
    # END full_test
# END Game

#===============================================================================
class Team(object):
    '''Team Class
    
    Instantiation Args:
        name (str): team will be looked up in NFL class using this name
    Instance Attributes:
        group (str): name of the team's subgroup (i.e. "NFC North")
    Operations:
        a + b: string addition using team name
        a == b: string equivalence test
        hash(a): returns hash of team name string
        str(a): returns team name as a string
    Notes:
        Consider abstracting to a general Team class for a data structure
        containing alias, affiliation, members, etc...
    Example:
        tm1 = Team('Green Bay Packers')
        tm2 = Team('GNB')
        str(tm1)
            "Green Bay Packers"
        str(tm2)
            "Green Bay Packers"
        tm1 == tm2
            True
    '''
    
    def __init__(self, name):
        self._name = NFL.normalize_name(name)
        self._group = NFL.get_tmgroup(self._name)
    # END __init__
    
    # Property methods
    #-----------------
    @property
    def group(self): return self._group
    
    # Magic methods
    #--------------
    def __add__(self, other):
        return str(self) + other
    # END __add__
    
    def __radd__(self, other):
        return str(self) + other
    # END __radd__
    
    def __eq__(self, other):
        return str(self) == str(other)
    # END __eq__
    
    def __format__(self, fmtcode):
        if fmtcode == '' or fmtcode == 's':
            return NFL.shorten_name(str(self))
        elif fmtcode == 'l':
            return str(self)
        # END if
    # END __format__
    
    def __hash__(self): return hash(self._name)
    
    def __str__(self): return self._name
    
    # Team-specific methods
    #----------------------
    def set_rand_strength(self, loc=0.0, scale=0.75):
        self.v = np.exp( np.random.normal(loc=loc, scale=scale) )
    # END set_rand_strength
# END Team

#===============================================================================
class WLTRecord(object):
    '''Win-Loss-Tie Record
    
    Class Attributes:
        W (int)
        L (int)
        T (int)
        Wpct (float)
        ngms (int)
    Class Methods:
        addW
        addL
        addT
    Instantiation Args: -none-
    '''
    
    def __init__(self, *args):
        if len(args) == 0:
            self._wlt = [0, 0, 0]
        elif len(args) == 1:
            self._wlt = [int(args[0][0]), int(args[0][1]), int(args[0][2])]
        else:
            self._wlt = [int(args[0]), int(args[1]), int(args[2])]
        # END if
    # END __init__
    
    # Property Methods
    #-----------------
    @property
    def W(self): return self._wlt[0]
    
    @property
    def L(self): return self._wlt[1]
    
    @property
    def T(self): return self._wlt[2]
    
    @property
    def Wpct(self): return (self.W + 0.5*self.T)/self.ngms
    
    @property
    def ngms(self): return self.W + self.L + self.T
    
    # Magic Methods
    #--------------
    def __eq__(self, other): return self.Wpct == other.Wpct
    
    def __le__(self, other): return self.Wpct <= other.Wpct
    
    def __lt__(self, other): return self.Wpct < other.Wpct
    
    def __ge__(self, other): return self.Wpct >= other.Wpct
    
    def __gt__(self, other): return self.Wpct > other.Wpct
    
    def __ne__(self, other): return self.Wpct != other.Wpct
    
    def __float__(self): return self.Wpct
    
    def __format__(self, fmtstr):
        out = '{0:'+fmtstr+'}-'+'{1:'+fmtstr+'}-'+'{2:'+fmtstr+'}'
        return out.format(*self._wlt)
    # END __format__
    
    def __str__(self):
        return str(self.W) + '-' + str(self.L) + '-' + str(self.T)
    # END __str__
    
    # WLTRecord-Specific Instance Methods
    #------------------------------------
    def addW(self, wins=1):
        self._wlt[0] += int(wins)
    # END addW
    
    def addL(self, losses=1):
        self._wlt[1] += int(losses)
    # END addL
    
    def addT(self, ties=1):
        self._wlt[2] += int(ties)
    # END addT
    
    # Test method(s)
    #---------------
    @classmethod
    def full_test(cls):
        print 'Testing WLTRecord class...'
        print 'creating empty instance'
        r = cls()
        print 'type(r) = ' + repr(type(r))
        print 'r = ' + str(r)
        print 'Adding 5 wins'
        r.addW(4)
        r.addW()
        print 'r = ' + str(r)
        print 'Adding 2 losses'
        r.addL(1)
        r.addL()
        print 'r = ' + str(r)
        print 'Adding 3 ties'
        r.addT(2)
        r.addT()
        print 'r = {0:02d} ({1:02d} games)'.format(r, r.ngms)
        print 'r = {0:2d} ({1:0.3f})'.format(r, r.Wpct)
        print 'creating second record of 6-1-4'
        r2 = cls(6,1,4)
        print 'r2 = {0:02d} ({1:0.3f})'.format(r2, r2.Wpct)
        print 'creating third record of 4-2-4'
        r3 = cls((4,2,4))
        print 'r3 = {0:02d} ({1:0.3f})'.format(r3, r3.Wpct)
        print 'r > r2 is ' + str(r>r2)
        print 'r2 < r3 is ' + str(r2<r3)
        print 'r >= r is ' + str(r>=r)
        print 'r <= r3 is ' + str(r<=r3)
        print 'r == r is' + str(r==r)
        print 'r == r2 is ' + str(r==r2)
        print 'r3 != r is ' + str(r3!=r)
        print 'float(r) = {0:0.3f}'.format(float(r))
        print 'WLTRecord testing complete.'
    # END full_test
# END WLTRecord

#===============================================================================
def logit(p): return np.log(p/(1.0-p))
def invlogit(x): return 1.0 / (1.0 + np.exp(-x))





