from __future__ import annotations

import sys
from queue import PriorityQueue

import numpy as np

from instance import Instance, FDROperator
from heuristic import Heuristic
from hmax import Hmax


class Plan:
    def __init__(self, actions=[], value=0):
        self.actions = actions
        self.value = value

    def add_action(self, op: FDROperator) -> Plan:
        return Plan(actions=self.actions + [op.name], value=self.value + op.cost)

    def __str__(self) -> str:
        res = ''
        for a in self.actions:
            res += f'{a}\n'
        res += f'Plan cost: {self.value}'
        return res


class State:
    @staticmethod
    def initial(instance: Instance) -> State:
        vs = [0 for _ in range(len(instance.fdr.variables))]
        for (v, d) in instance.fdr.init_state:
            vs[v] = d
        return State(vs)

    def __init__(self, vs: list[int], plan: Plan = Plan()):
        self.vs = vs
        self.plan = plan

    def get_applicable(self, instance: Instance) -> list[FDROperator]:
        applicable = []
        for op in instance.fdr.operators:
            add = True
            for var, val in op.preconditions:
                if self.vs[var] != val:
                    add = False
                    break
            if add:
                applicable.append(op)
        return applicable

    def apply(self, op: FDROperator) -> State:
        new_vs = self.vs.copy()
        for var, val in op.effects:
            new_vs[var] = val
        return State(new_vs, self.plan.add_action(op))

    def is_goal(self, instance: Instance) -> State:
        for var, val in instance.fdr.goal_state:
            if self.vs[var] != val:
                return False
        return True

    def __str__(self) -> str:
        res = ''
        for i, v in enumerate(self.vs):
            res += f'({i}: {v}) '
        res += '\n'
        res += str(self.plan)
        return res


def search(instance: Instance, hclass: Heuristic) -> Plan:
    h = hclass(instance.strips)

    s0 = State.initial(instance)

    shape = tuple(len(v.values) for v in instance.fdr.variables)

    # try it with a matrix, we can do it with dict then
    g = np.full(shape, np.inf)
    # g = defaultdict(np.inf)
    g[*s0.vs] = 0

    open = PriorityQueue()

    it = 0
    open.put((0, 0, s0))
    while not open.empty():
        s = open.get()[2]
        if s.is_goal(instance):
            return s.plan
        for op in s.get_applicable(instance):
            it += 1
            new_s = s.apply(op)
            v = new_s.plan.value
            if (v < g[*new_s.vs]):
                g[*new_s.vs] = v
                hs = h(new_s.vs)
                open.put((v + hs, it, new_s))

    return None


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception('Usage: python hmax.py <path> <heuristic>')

    instance = Instance(sys.argv[1])

    if sys.argv[2] == 'hmax':
        h = Hmax
    else:
        raise Exception('Unknown <heuristic> - options are {hmax}')

    plan = search(instance, h)
    print(plan)
