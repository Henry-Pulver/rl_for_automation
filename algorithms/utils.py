from functools import reduce
import operator


def prod(iterable):
    return reduce(operator.mul, iterable, 1)
