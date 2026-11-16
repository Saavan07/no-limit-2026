import unittest
from blotto_lab.core import Game,iter_allocations
from game2_lab.core import reference_group
from game2_lab.weight_counters import worst_clone_population


class ClonePopulationTests(unittest.TestCase):
    def test_exact_inverse_oracle_against_every_small_game_allocation(self):
        for game in [Game(4,(1,2,3)),Game(2,(1,2,3,4,5))]:
            actions=list(iter_allocations(game))
            for x in actions:
                scored=[]
                for y in actions:
                    result=reference_group([x,y,y,y],game);scored.append((result.payouts[0],result.castle_points[0],y))
                expected=min(scored);actual=worst_clone_population(x,game)
                self.assertEqual((actual['payout'],actual['castle_points'],actual['opponent']),expected)
        self.assertEqual(worst_clone_population((0,0,0,0,100))['payout'],3.75)
        self.assertEqual(worst_clone_population((0,1,10,86,3))['payout'],0)


if __name__=='__main__':unittest.main()
