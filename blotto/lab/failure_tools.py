"""Exact diagnostic tools for expected-score mixture response maps."""
from fractions import Fraction
from math import lcm
from blotto_lab.core import BestResponse, ScoreTable, best_response, validate_allocation


def capped_response(table, first_cap=None):
    """Use a dominating forbidden-bid penalty; return values in original units."""
    if first_cap is None:
        return best_response(table)
    if type(first_cap) is not int or not 0 <= first_cap <= table.game.budget:
        raise ValueError('Invalid first-castle cap')
    span = sum(max(row)-min(row) for row in table.numerators)+1
    rows = list(table.numerators)
    floor = min(rows[0])-span
    rows[0] = tuple(v if k<=first_cap else floor for k,v in enumerate(rows[0]))
    auxiliary = ScoreTable(table.game,tuple(rows),table.denominator,
                           'forbidden-bid penalty','auxiliary objective')
    br = best_response(auxiliary)
    assert br.allocation[0]<=first_cap
    return BestResponse(br.allocation,table.expected_score(br.allocation),
                        br.optimum_count,br.evaluations,'exact DP with first-bid cap')


def share_map(a,b,first_cap=None):
    """Complete rational envelope with directional endpoint ties."""
    if a.game!=b.game:
        raise ValueError('Mismatched games')
    denominator=lcm(a.denominator,b.denominator)
    ra=[[v*(denominator//a.denominator) for v in row] for row in a.numerators]
    rb=[[v*(denominator//b.denominator) for v in row] for row in b.numerators]
    slope_span=sum(max(y-x for x,y in zip(ar,br))-min(y-x for x,y in zip(ar,br)) for ar,br in zip(ra,rb))
    dominance=slope_span+1
    cache={};queries=[]
    def oracle(alpha,direction=0):
        key=(alpha,direction)
        if key in cache:return cache[key]
        primary=[[(alpha.denominator-alpha.numerator)*x+alpha.numerator*y for x,y in zip(ar,br)] for ar,br in zip(ra,rb)]
        rows=tuple(tuple(p*dominance+direction*(y-x) for p,x,y in zip(pr,ar,br)) for pr,ar,br in zip(primary,ra,rb))
        response=capped_response(ScoreTable(a.game,rows,1,'lexicographic objective','auxiliary'),first_cap)
        x=response.allocation;av=a.expected_score(x);bv=b.expected_score(x)
        record={'share':alpha,'direction':direction,'allocation':x,'baseline_score':av,
                'scenario_score':bv,'score':(1-alpha)*av+alpha*bv,
                'optimum_count':response.optimum_count}
        cache[key]=record;queries.append(record);return record
    def same(l,r):return l['baseline_score']==r['baseline_score'] and l['scenario_score']==r['scenario_score']
    def between(lo,hi,left,right):
        if same(left,right):return [(lo,hi,left)]
        cross=(right['baseline_score']-left['baseline_score'])/((left['scenario_score']-left['baseline_score'])-(right['scenario_score']-right['baseline_score']))
        assert lo<cross<hi
        point=oracle(cross)
        value=(1-cross)*left['baseline_score']+cross*left['scenario_score']
        if point['score']==value:return [(lo,cross,left),(cross,hi,right)]
        assert point['score']>value
        return between(lo,cross,left,oracle(cross,-1))+between(cross,hi,oracle(cross,1),right)
    raw=between(Fraction(0),Fraction(1),oracle(Fraction(0),1),oracle(Fraction(1),-1))
    merged=[]
    for lo,hi,line in raw:
        if merged and same(merged[-1][2],line):merged[-1]=(merged[-1][0],hi,line)
        else:merged.append((lo,hi,line))
    segments=[]
    for lo,hi,line in merged:
        point=oracle((lo+hi)/2)
        assert same(point,line)
        segments.append({'lower_share':lo,'upper_share':hi,'allocation':point['allocation'],
                         'baseline_score':point['baseline_score'],'scenario_score':point['scenario_score'],
                         'open_interval_optimum_count':point['optimum_count']})
    boundaries=sorted({s['lower_share'] for s in segments}|{Fraction(1)})
    return {'segments':segments,'boundaries':[oracle(x) for x in boundaries],
            'queries':queries,'first_cap':first_cap,'oracle_calls':len(queries)}


def first_loss_share(mapped,a,b,x,threshold):
    """Earliest replacement mass reaching a given opportunity cost, or None."""
    threshold=Fraction(threshold);ax=a.expected_score(x);bx=b.expected_score(x)
    if threshold<0:raise ValueError('Negative threshold')
    for s in mapped['segments']:
        lo,hi=s['lower_share'],s['upper_share']
        intercept=s['baseline_score']-ax
        slope=(s['scenario_score']-s['baseline_score'])-(bx-ax)
        vlo=intercept+slope*lo;vhi=intercept+slope*hi
        if vlo>=threshold:return lo
        if vhi>=threshold:
            assert slope>0
            return (threshold-intercept)/slope
    return None


def marginal_probabilities(table):
    result=[]
    for value,row in zip(table.game.values,table.numerators):
        previous=Fraction(0);prob=[]
        for number in row:
            current=Fraction(2*number,value*table.denominator)-2*previous
            if current<0:raise ValueError('Not a valid marginal score table')
            prob.append(current);previous+=current
        if previous!=1:raise ValueError('Marginal mass does not sum to one')
        result.append(prob)
    return result


def attribute(table,x,response):
    x=validate_allocation(x,table.game);response=validate_allocation(response,table.game)
    probabilities=marginal_probabilities(table);castles=[]
    for i,(v,bid,rbid,prob) in enumerate(zip(table.game.values,x,response,probabilities)):
        win=sum(prob[:bid]);tie=prob[bid];lose=1-win-tie
        actual=Fraction(table.numerators[i][bid],table.denominator)
        counter=Fraction(table.numerators[i][rbid],table.denominator)
        assert actual==v*(win+tie/2)
        castles.append({'castle':i+1,'bid':bid,'response_bid':rbid,'win':win,'tie':tie,'lose':lose,
                        'score':actual,'response_score':counter,'regret_contribution':counter-actual})
    moves=[];old=table.expected_score(x)
    for i in range(table.game.castles):
        if not x[i]:continue
        for j in range(table.game.castles):
            if i==j:continue
            y=list(x);y[i]-=1;y[j]+=1
            moves.append({'from_castle':i+1,'to_castle':j+1,'allocation':tuple(y),'gain':table.expected_score(y)-old})
    best=max((p['gain'] for p in moves),default=Fraction(0))
    return {'allocation':x,'best_response':response,'score':old,'best_response_score':table.expected_score(response),
            'regret':table.expected_score(response)-old,'castles':castles,
            'best_single_troop_gain':best,'best_single_troop_moves':[p for p in moves if p['gain']==best]}


def frozen_envelope(a,b,candidates):
    """Exact upper envelope of a declared finite candidate set."""
    lines={name:(a.expected_score(x),b.expected_score(x)) for name,x in candidates.items()}
    points={Fraction(0),Fraction(1)};names=sorted(lines)
    for i,name in enumerate(names):
        av,bv=lines[name]
        for other in names[i+1:]:
            cv,dv=lines[other];den=(bv-av)-(dv-cv)
            if den:
                crossing=(cv-av)/den
                if 0<crossing<1:points.add(crossing)
    def winners(t):
        values={k:(1-t)*v[0]+t*v[1] for k,v in lines.items()};best=max(values.values())
        return [k for k in names if values[k]==best]
    segments=[]
    for lo,hi in zip(sorted(points),sorted(points)[1:]):
        win=winners((lo+hi)/2)
        if segments and segments[-1]['winners']==win:segments[-1]['upper_share']=hi
        else:segments.append({'lower_share':lo,'upper_share':hi,'winners':win})
    boundaries=sorted({s['lower_share'] for s in segments}|{Fraction(1)})
    return {'segments':segments,'boundaries':[{'share':p,'winners':winners(p)} for p in boundaries],
            'scores':{k:{'baseline':v[0],'scenario':v[1]} for k,v in lines.items()}}
