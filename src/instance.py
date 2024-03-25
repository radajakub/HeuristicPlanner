import sys


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
        res += f'- eff: {self.effects}\n'
        return res


class FDR:
    def __init__(self, path: str):
        self.variables = []
        self.init_state = []
        self.goal_state = []
        self.operators = []

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

            value_num = int(lines[n])
            n += 1
            for _ in range(value_num):
                var.add_value(lines[n])
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
            self.init_state.append((self.variables[i].name, self.variables[i].values[action_idx]))
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
            self.goal_state.append((var.name, var.values[val_idx]))
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
                operator.add_precondition(var.name, var.values[val_idx])
                n += 1

            # preconditions and effects
            num_eff = int(lines[n])
            n += 1
            for _ in range(num_eff):
                _, var_idx, from_idx, to_idx = [int(x) for x in lines[n].split()]
                var = self.variables[var_idx]
                operator.add_precondition(var.name, var.values[from_idx])
                operator.add_effect(var.name, var.values[to_idx])
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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Usage: python instance.py <path>')

    fdr = FDR(sys.argv[1])
    print(fdr)
