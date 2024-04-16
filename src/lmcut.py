import sys
import numpy as np

from instance import Instance, STRIPS, STRIPSAction
from heuristic import Heuristic
from hmax import Hmax, INF
from id_machine import IdMachine


class JustificationGraph:

    # a multigraph where:
    # - nodes are strips facts
    # - edges are {<p, a, q> in F x A x F | q in add_a and p = pcf(a)}
    def __init__(self, strips: STRIPS, pcf: np.ndarray):
        self.strips = strips
        self.V = strips.F
        self.num_vertices = len(strips.F)
        # do edges as adjacency list where items are tuples (A, q)
        self.E = [[] for _ in range(self.num_vertices)]
        # save also reversed edges for faster traversal
        self.E_rev = [[] for _ in range(self.num_vertices)]
        for ai, a in enumerate(strips.A):
            p = pcf[ai]
            for q in a.add:
                self.E[p].append((a, q))
                self.E_rev[q].append((a, p))

    def cut(self) -> int:
        # find states reachable with cost 0 from up state
        visited = np.zeros(len(self.V), dtype=bool)
        q = [(g, 0) for g in self.strips.g]

        up_set = set()

        while q:
            s, cost = q.pop(0)
            up_set.add(s)
            for a, s_ in self.E_rev[s]:
                new_cost = cost + a.cost
                if new_cost == 0 and not visited[s_]:
                    visited[s_] = True
                    q.append((s_, new_cost))

        visited = np.zeros(len(self.V), dtype=bool)
        q = [s0 for s0 in self.strips.s0]

        down_set = set()

        while q:
            s = q.pop(0)
            down_set.add(s)
            for a, s_ in self.E[s]:
                if not visited[s_] and s_ not in up_set:
                    visited[s_] = True
                    q.append(s_)

        landmark = set()
        for p, edges in enumerate(self.E):
            for a, q in edges:
                if p in down_set and q in up_set:
                    landmark.add(a)

        return min(l.cost for l in landmark), landmark

    def __str__(self) -> str:
        res = f'V: {self.V}\n'
        res += 'E:\n'
        for u, edges in enumerate(self.E):
            for a, v in edges:
                res += f'{u} -> {v} by {a.name}\n'
        res += 'E_rev:\n'
        for u, edges in enumerate(self.E_rev):
            for a, v in edges:
                res += f'{u} <- {v} by {a.name}\n'
        return res


class LMCut(Heuristic):
    @staticmethod
    def compute_pcf(strips: STRIPS, s_star: np.ndarray) -> np.ndarray:
        # compute pcf(a) = argmax_{p in pre_a} s_star(p) for a in A
        pcf = np.zeros(len(strips.A), dtype=int)
        for ai, a in enumerate(strips.A):
            p_max = -INF
            for p in a.pre:
                if s_star[p] > p_max:
                    p_max = s_star[p]
                    pcf[ai] = p
                elif s_star[p] == p_max:
                    v1, d1 = strips.id_machine.get_value(p)
                    v2, d2 = strips.id_machine.get_value(pcf[ai])
                    if v1 > v2 or (v1 == v2 and d1 > d2):
                        pcf[ai] = p
        return pcf

    def compute(self, s: set[int]) -> int:
        # transform the strips task with the lm cut start and goal
        strips = self.strips.lm_transform(s)

        hmax_fun = Hmax(strips)

        hlmcut = 0

        # compute fixed point
        hmax, s_star = hmax_fun.compute(list(strips.s0))

        if hmax == INF:
            return INF

        while hmax != 0:
            # compute pcf
            pcf = self.compute_pcf(strips, s_star)

            # build justification graph
            jg = JustificationGraph(strips, pcf)

            # compute cut
            cut_cost, cut = jg.cut()

            # adjust heuristic value
            hlmcut += cut_cost

            # adjust costs of actions
            for a in cut:
                a.cost -= cut_cost

            # recompute hmax and fixed point
            hmax, s_star = hmax_fun.compute(list(strips.s0))

        return hlmcut

    def __call__(self, s: list[int]) -> int:
        return self.compute(self.strips.vars_to_facts(s))


def test() -> None:
    id_machine = IdMachine()
    for i in range(5):
        id_machine.get_id((i, 0))
    strips = STRIPS(
        F={0, 1, 2, 3, 4},
        A=[
            STRIPSAction(pre={0}, add={1, 2}, cost=3, name='o1'),
            STRIPSAction(pre={0}, add={3}, cost=5, name='o2'),
            STRIPSAction(pre={1}, add={2, 3}, cost=1, name='o3'),
            STRIPSAction(pre={0, 1}, add={4}, cost=4, name='o4'),
        ],
        s0={0},
        g={2, 3, 4},
        id_machine=id_machine
    )
    print('original strips')
    print(strips)

    # extend strips
    strips = strips.lm_transform(s={0})
    print('extended strips')
    print(strips)

    # hmax
    for i in range(3):
        print(f'iteration {i + 1}')
        hmax_fun = Hmax(strips)
        hmax, s_star = hmax_fun.compute(list(strips.s0))
        print(f'[hmax {hmax}] s_star: {s_star}')
        print()

        # pcf
        pcf = LMCut.compute_pcf(strips, s_star)
        print(f'pcf {pcf}')
        print()

        # justification graph
        jg = JustificationGraph(strips, pcf)
        print('justification graph')
        print(jg)

        cut_cost, cut = jg.cut()
        for a in cut:
            a.cost -= cut_cost

        print(strips)

    hmax_fun = Hmax(strips)
    hmax, s_star = hmax_fun.compute(list(strips.s0))
    print(f'final hmax: {hmax}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Usage: python lmcut.py <path>')

    instance = Instance(sys.argv[1])
    heuristic = LMCut(instance.strips)

    print(heuristic.compute(instance.strips.s0))
