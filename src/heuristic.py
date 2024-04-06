from strips import STRIPS


class Heuristic:
    def __init__(self, strips: STRIPS):
        self.strips = strips

    def __call__(self, s: set[int]) -> int:
        raise NotImplementedError('Heuristic must implement __call__ method')
