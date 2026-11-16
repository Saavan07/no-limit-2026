from fractions import Fraction
import random
import unittest
from blotto_lab.core import Game,Population,ScoreTable,iter_allocations,mix_tables
from failure_tools import attribute,capped_response,first_loss_share,marginal_probabilities,share_map
from verify_failure_analysis import forward_response


class FailureAnalysisTests(unittest.TestCase):
    def brute(self,table,cap):
        values=[(table.expected_score(x),x)for x in iter_allocations(table.game)if cap is None or x[0]<=cap]
        top=max(v for v,x in values);wins=[x for v,x in values if v==top]
        return top,min(wins),len(wins)

    def test_capped_and_forward_oracles_against_complete_enumeration(self):
        game=Game(6,(1,2,3,4,5));actions=list(iter_allocations(game));rng=random.Random(882)
        for repeat in range(5):
            table=Population.from_people(rng.sample(actions,8),game).table()
            for cap in [None,0,1,3,6]:
                expected=self.brute(table,cap)
                for response in [capped_response(table,cap),forward_response(table,cap)]:
                    self.assertEqual((response.expected_score,response.allocation,response.optimum_count),expected)
        # Auxiliary directional objectives may contain negative local values.
        table=ScoreTable(game,tuple(tuple(rng.randrange(-50,50)for _ in range(7))for _ in range(5)),7)
        for cap in [0,2,6]:
            br=capped_response(table,cap);self.assertEqual((br.expected_score,br.allocation,br.optimum_count),self.brute(table,cap))

    def test_complete_maps_and_boundary_ties_against_enumeration(self):
        game=Game(5,(1,2,3));actions=list(iter_allocations(game));rng=random.Random(333)
        for repeat in range(5):
            a,b=[Population.from_people(rng.sample(actions,6),game).table()for _ in range(2)]
            for cap in [None,0,2]:
                mapped=share_map(a,b,cap)
                for seg in mapped['segments']:
                    lo,hi=seg['lower_share'],seg['upper_share'];self.assertLess(lo,hi)
                    for t in [lo,(lo+hi)/2,hi]:
                        table=mix_tables([a,b],[1-t,t]);top,x,ties=self.brute(table,cap)
                        self.assertEqual(table.expected_score(seg['allocation']),top)
                        if lo<t<hi:self.assertEqual((seg['allocation'],seg['open_interval_optimum_count']),(x,ties))
                for p in mapped['boundaries']:
                    expected=self.brute(mix_tables([a,b],[1-p['share'],p['share']]),cap)
                    self.assertEqual((p['score'],p['allocation'],p['optimum_count']),expected)

    def test_attribution_marginals_and_local_plateau(self):
        game=Game(6,(1,2,3));pop=Population.from_people([(0,3,3),(0,3,3),(2,2,2),(4,0,2)],game);table=pop.table()
        probabilities=marginal_probabilities(table)
        for i,row in enumerate(probabilities):
            for bid,p in enumerate(row):self.assertEqual(p,Fraction(sum(m for x,m in pop.entries if x[i]==bid),pop.total_mass))
        br=capped_response(table);x=(2,2,2);record=attribute(table,x,br.allocation)
        self.assertEqual(sum(v['regret_contribution']for v in record['castles']),br.expected_score-table.expected_score(x))
        self.assertEqual(sum(v['score']for v in record['castles']),pop.reference_mean(x))

    def test_loss_threshold_at_boundary_and_not_reached(self):
        game=Game(2,(1,2));a=Population.from_people([(0,2)],game).table();b=Population.from_people([(2,0)],game).table()
        mapped=share_map(a,b);initial=capped_response(a).allocation
        for target in [Fraction(0),Fraction(1,10),Fraction(1,2),Fraction(10)]:
            found=first_loss_share(mapped,a,b,initial,target)
            if found is not None:
                optimum=capped_response(mix_tables([a,b],[1-found,found])).expected_score
                self.assertEqual(optimum-((1-found)*a.expected_score(initial)+found*b.expected_score(initial)),target)
            else:self.assertLess(capped_response(b).expected_score-b.expected_score(initial),target)


if __name__=='__main__':unittest.main()
