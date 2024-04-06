import sys
import numpy as np

from strips import STRIPS


def hmax(strips: STRIPS, s: set[int]):
    # prepare enriched state
    sigma = np.full(len(strips.F), np.iinfo(np.int32).max, dtype=int)
    for p in s:
        sigma[p] = 0

    # apply actions without preconditions
    # setup counter for remaining actions
    U = np.zeros(len(strips.A))
    for ai, a in enumerate(strips.A):
        pre_size = len(a.pre)
        U[ai] = pre_size
        if pre_size == 0:
            for p in a.add:
                sigma[p] = min(sigma[p], a.cost)

    C = set()
    while not strips.g.issubset(C):
        q = None
        q_min = np.inf
        for r in strips.F.difference(C):
            if sigma[r] < q_min:
                q_min = sigma[r]
                q = r
        C.add(q)
        for ai, a in enumerate(strips.A):
            if q in a.pre:
                U[ai] -= 1
                if U[ai] == 0:
                    for p in a.add:
                        v = a.cost + sigma[q]
                        if v < sigma[p]:
                            sigma[p] = v

    return max(sigma[p] for p in strips.g)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Usage: python hmax.py <path>')

    strips = STRIPS.from_SAS(sys.argv[1])
    value = hmax(strips, strips.s0)
    print(value)
