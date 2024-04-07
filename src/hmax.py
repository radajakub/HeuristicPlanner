import sys
import numpy as np

from instance import Instance
from heuristic import Heuristic


class Hmax(Heuristic):
    def __call__(self, s: set[int]) -> tuple[int, np.ndarray]:
        # prepare enriched state
        sigma = np.full(len(self.strips.F), np.iinfo(np.int32).max, dtype=int)
        for p in s:
            sigma[p] = 0

        # apply actions without preconditions
        # setup counter for remaining actions
        U = np.zeros(len(self.strips.A))
        for ai, a in enumerate(self.strips.A):
            pre_size = len(a.pre)
            U[ai] = pre_size
            if pre_size == 0:
                for p in a.add:
                    sigma[p] = min(sigma[p], a.cost)

        C = set()
        while not self.strips.g.issubset(C):
            q = None
            q_min = np.inf
            for r in self.strips.F.difference(C):
                if sigma[r] < q_min:
                    q_min = sigma[r]
                    q = r
            C.add(q)
            for ai, a in enumerate(self.strips.A):
                if q in a.pre:
                    U[ai] -= 1
                    if U[ai] == 0:
                        for p in a.add:
                            v = a.cost + sigma[q]
                            if v < sigma[p]:
                                sigma[p] = v

        return max(sigma[p] for p in self.strips.g), sigma


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Usage: python hmax.py <path>')

    instance = Instance(sys.argv[1])
    heuristic = Hmax(instance.strips)

    print(heuristic(instance.strips.s0)[0])
