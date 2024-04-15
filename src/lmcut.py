import sys
import numpy as np

from instance import STRIPS
from heuristic import Heuristic
from hmax import Hmax


class LMCut(Heuristic):
    def __call__(self, s: set[int]) -> int:
        # transform the strips task with the lm cut start and goal
        s_down, s_up = self.strips.lm_transform(s)
        hlmcut = 0

        # construct hmax heuristic for the transformed task
        hmax_fun = Hmax(self.strips)

        # while True:
        # hmax, s_star = hmax_fun(s_up)

        # compute fixed point of sdown of
        hmax, s_star = hmax_fun(s_down)
        print(hmax)
        print(s_star)

        if hmax == np.iinfo(np.int32).max:
            return np.iinfo(np.int32).max

        while hmax != 0:
            # construct justificiation graph G w.r.t pdf for
            break

        return hlmcut


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Usage: python lmcut.py <path>')

    strips = STRIPS.from_SAS(sys.argv[1])
    heuristic = LMCut(strips)

    print(heuristic(strips.s0))
