from typing import Callable

dot_func = Callable[[int, list[float], int, list[float], int], float]

class BlasFunctions:
    def __init__(self, dot: dot_func):
        self.dot = dot