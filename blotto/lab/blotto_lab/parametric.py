"""Exact best-response envelopes for a continuous mixture of two populations."""
from fractions import Fraction
from math import lcm
from .core import ScoreTable, best_response, mix_tables


def strategic_share_map(a,b):
    if a.game!=b.game:
        raise ValueError('Mismatched games')
    game = a.game
    denominator = lcm(a.denominator,b.denominator)
    rows_a = [[k*(denominator//a.denominator) for k in row] for row in a.numerators]
    rows_b = [[k*(denominator//b.denominator) for k in row] for row in b.numerators]
    for i,v in enumerate(game.values):
        if any(not 0<=k<=v*denominator for k in rows_a[i]+rows_b[i]):
            raise ValueError('Expected legal bounded local score tables')
    # Each total slope is in [-V*D,V*D]. This integer factor makes the primary
    # objective dominate any difference in the directional secondary objective.
    dominance = 2*game.total_value*denominator+1
    cache,queries = {},[]
    def oracle(share,direction=0):
        key = share,direction
        if key in cache: return cache[key]
        primary = [[(share.denominator-share.numerator)*x+share.numerator*y
                    for x,y in zip(ar,br)] for ar,br in zip(rows_a,rows_b)]
        if direction:
            rows = tuple(tuple(p*dominance+direction*(y-x) for p,x,y in zip(pr,ar,br))
                         for pr,ar,br in zip(primary,rows_a,rows_b))
            # This auxiliary integer table is a lexicographic optimization device,
            # not a population or an expected-score table.
            table = ScoreTable(game,rows,1,'primary then directional slope','auxiliary integer objective')
        else:
            table = ScoreTable(game,tuple(tuple(row) for row in primary),denominator*share.denominator)
        response = best_response(table)
        x = response.allocation
        record = {'share':share,'direction':direction,'allocation':x,
            'behavior_score':a.expected_score(x),'strategic_score':b.expected_score(x),
            'score':(1-share)*a.expected_score(x)+share*b.expected_score(x),
            'optimum_count':response.optimum_count}
        cache[key]=record; queries.append(record)
        return record
    def same_line(left,right):
        return left['behavior_score']==right['behavior_score'] and left['strategic_score']==right['strategic_score']
    def between(lo,hi,left,right):
        if same_line(left,right):
            return [(lo,hi,left)]
        numerator = right['behavior_score']-left['behavior_score']
        denominator_cross = ((left['strategic_score']-left['behavior_score'])-
                             (right['strategic_score']-right['behavior_score']))
        crossing = numerator/denominator_cross
        if not lo<crossing<hi:
            raise AssertionError('Directional endpoint responses must bracket an interior crossing')
        exact = oracle(crossing)
        current = (1-crossing)*left['behavior_score']+crossing*left['strategic_score']
        if exact['score']==current:
            return [(lo,crossing,left),(crossing,hi,right)]
        if exact['score']<current:
            raise AssertionError('Oracle cannot be worse than a legal endpoint response')
        left_at_cross,right_at_cross = oracle(crossing,-1),oracle(crossing,1)
        return between(lo,crossing,left,left_at_cross)+between(crossing,hi,right_at_cross,right)
    raw = between(Fraction(0),Fraction(1),oracle(Fraction(0),1),oracle(Fraction(1),-1))
    merged=[]
    for lo,hi,line in raw:
        if merged and same_line(merged[-1][2],line): merged[-1]=(merged[-1][0],hi,line)
        else: merged.append((lo,hi,line))
    segments=[]
    for lo,hi,line in merged:
        ordinary = oracle((lo+hi)/2)
        assert ordinary['score']==(1-(lo+hi)/2)*line['behavior_score']+(lo+hi)/2*line['strategic_score']
        assert same_line(ordinary,line)
        segments.append({'lower_share':lo,'upper_share':hi,'allocation':ordinary['allocation'],
            'behavior_score':ordinary['behavior_score'],'strategic_score':ordinary['strategic_score'],
            'open_interval_optimum_count':ordinary['optimum_count']})
    boundaries=sorted({s['lower_share'] for s in segments}|{Fraction(1)})
    return {'segments':segments,'boundaries':[oracle(s) for s in boundaries],
            'queries':queries,'oracle_calls':len(queries)}
