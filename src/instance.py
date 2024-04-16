from __future__ import annotations
from copy import deepcopy
import sys

from id_machine import IdMachine


class FDRVariable:
    def __init__(self):
        self.name = ''
        self.values = []

    def set_name(self, name: str) -> None:
        self.name = name

    def add_value(self, value: str) -> None:
        self.values.append(value)

    def __str__(self) -> str:
        return f'{self.name}: {self.values}'


class FDROperator:
    def __init__(self):
        self.name = ''
        self.preconditions = []
        self.effects = []
        self.cost = 0

    def set_name(self, name: str) -> None:
        self.name = name

    def add_precondition(self, var: str, val: str) -> None:
        self.preconditions.append((var, val))

    def add_effect(self, var: str, val: str) -> None:
        self.effects.append((var, val))

    def set_cost(self, cost: int) -> None:
        self.cost = cost

    def __str__(self) -> str:
        res = f'{self.name} ({self.cost}):\n'
        res += f'- pre: {self.preconditions}\n'
        res += f'- eff: {self.effects}'
        return res


class FDR:
    def __init__(self, path: str):
        self.variables = []
        self.init_state = []
        self.goal_state = []
        self.operators = []

        self.var_machine = IdMachine[str]()

        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # skip first 6 lines for now (version and metric info)
        lines = lines[6:]

        n = 0

        # load variables
        num_vars = int(lines[n])
        n += 1
        for _ in range(num_vars):
            assert lines[n] == 'begin_variable'
            var = FDRVariable()
            n += 1

            var.set_name(lines[n])
            n += 1

            # skip -1
            n += 1

            var_id = self.var_machine.get_id(var.name)

            value_num = int(lines[n])
            n += 1
            for i in range(value_num):
                var.add_value(i)
                n += 1

            assert lines[n] == 'end_variable'
            self.variables.append(var)
            n += 1

        # skip mutex groups
        num_mutex_groups = int(lines[n])
        n += 1
        for _ in range(num_mutex_groups):
            while lines[n] != 'end_mutex_group':
                n += 1
            n += 1

        # load init state
        assert lines[n] == 'begin_state'
        n += 1
        for i in range(num_vars):
            action_idx = int(lines[n])
            var_id = self.var_machine.get_id(self.variables[i].name)
            self.init_state.append((var_id, self.variables[i].values[action_idx]))
            n += 1
        assert lines[n] == 'end_state'
        n += 1

        # load goal state
        assert lines[n] == 'begin_goal'
        n += 1
        num_goals = int(lines[n])
        n += 1
        for i in range(num_goals):
            var_idx, val_idx = [int(x) for x in lines[n].split()]
            var = self.variables[var_idx]
            var_id = self.var_machine.get_id(var.name)
            self.goal_state.append((var_id, var.values[val_idx]))
            n += 1
        assert lines[n] == 'end_goal'
        n += 1

        # load operators
        num_operators = int(lines[n])
        n += 1
        for _ in range(num_operators):
            assert lines[n] == 'begin_operator'
            operator = FDROperator()
            n += 1

            operator.set_name(lines[n])
            n += 1

            # only preconditions
            num_pre = int(lines[n])
            n += 1
            for _ in range(num_pre):
                var_idx, val_idx = [int(x) for x in lines[n].split()]
                var = self.variables[var_idx]
                var_id = self.var_machine.get_id(var.name)
                operator.add_precondition(var_id, var.values[val_idx])
                n += 1

            # preconditions and effects
            num_eff = int(lines[n])
            n += 1
            for _ in range(num_eff):
                _, var_idx, from_idx, to_idx = [int(x) for x in lines[n].split()]
                var = self.variables[var_idx]
                var_id = self.var_machine.get_id(var.name)
                if from_idx != -1:
                    operator.add_precondition(var_id, var.values[from_idx])
                operator.add_effect(var_id, var.values[to_idx])
                n += 1

            # cost
            cost = int(lines[n])
            operator.set_cost(cost)
            n += 1

            assert lines[n] == 'end_operator'
            self.operators.append(operator)
            n += 1

    def __str__(self) -> str:
        res = 'V:\n'
        for var in self.variables:
            res += f'- {var}\n'
        res += 's0:\n'
        for var, val in self.init_state:
            res += f'- {var}: {val}\n'
        res += 'g:\n'
        for var, val in self.goal_state:
            res += f'- {var}: {val}\n'
        res += 'op:\n'
        for op in self.operators:
            res += f'- {op}\n'
        return res


class STRIPSAction:
    def __init__(self, pre: set[int], add: set[int], cost: int, name: str = None):
        self.pre = pre
        self.add = add
        self.cost = cost
        self.name = name

    def __str__(self) -> str:
        return f'- {self.name} ({self.cost}): {self.pre} -> {self.add}'


class STRIPS:
    @staticmethod
    def from_FDR(fdr: FDR) -> STRIPS:
        # translate tuples (var, value) to unique ids
        id_machine = IdMachine[tuple[int, int]]()

        # transform variables
        F = set()
        for var in fdr.variables:
            var_id = fdr.var_machine.get_id(var.name)
            for val in var.values:
                varval = (var_id, val)
                F.add(id_machine.get_id(varval))

        # transform initial state
        s0 = set(id_machine.get_id((var, val)) for var, val in fdr.init_state)

        # transform goal state
        g = set(id_machine.get_id((var, val)) for var, val in fdr.goal_state)

        # transform operations into actions
        A = list()
        for op in fdr.operators:
            s_action = STRIPSAction(
                pre=set(id_machine.get_id((var, val)) for var, val in op.preconditions),
                add=set(id_machine.get_id((var, val)) for var, val in op.effects),
                cost=op.cost,
                name=op.name
            )
            A.append(s_action)

        return STRIPS(F, A, s0, g, id_machine)

    def __init__(self, F: set[int], A: list[STRIPSAction], s0: set[int], g: set[int], id_machine: IdMachine[tuple[int, int]]):
        self.F = F
        self.A = A
        self.s0 = s0
        self.g = g
        self.id_machine = id_machine

    def vars_to_facts(self, v: list[int]) -> list[int]:
        return [self.id_machine.get_id(x) for x in enumerate(v)]

    def lm_transform(self, s: list[int]) -> STRIPS:
        strips = deepcopy(self)

        down = len(strips.F)
        up = down + 1

        strips.F.add(down)
        strips.F.add(up)

        down_set = {down}
        up_set = {up}

        strips.A.append(STRIPSAction(pre=down_set, add=s, cost=0, name='a_down'))
        strips.A.append(STRIPSAction(pre=self.g, add=up_set, cost=0, name='a_up'))

        strips.s0 = down_set
        strips.g = up_set

        return strips

    def __str__(self) -> str:
        res = f'F: {self.F}\n'
        res += f's0: {self.s0}\n'
        res += f'g: {self.g}\n'
        res += 'A:\n'
        for a in self.A:
            res += f'{a}\n'
        return res


class Instance:
    def __init__(self, path: str):
        self.fdr = FDR(path)
        self.strips = STRIPS.from_FDR(self.fdr)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Usage: python strips.py <path>')

    instance = Instance(sys.argv[1])
    print(instance.fdr)
    print(instance.strips)
