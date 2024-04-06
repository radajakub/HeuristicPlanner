from typing import TypeVar, Generic, Dict

T = TypeVar('T')


class IdMachine(Generic[T]):
    def __init__(self):
        self.ids: Dict[T, int] = {}
        self.values: Dict[int, T] = {}
        self.next_id = 0

    def get_id(self, value: T) -> int:
        if value not in self.ids:
            self.ids[value] = self.next_id
            self.values[self.next_id] = value
            self.next_id += 1

        return self.ids[value]

    def get_value(self, id: int) -> T:
        return self.values[id]
