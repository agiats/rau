import collections
import random


def get_random_seed(random_seed):
    return random.getrandbits(32) if random_seed is None else random_seed


def get_random_generator_and_seed(random_seed):
    random_seed = get_random_seed(random_seed)
    return random.Random(random_seed), random_seed


def group_by(iterable, key):
    result = collections.defaultdict(list)
    for value in iterable:
        result[key(value)].append(value)
    return result


def product(iterable):
    result = 1
    for value in iterable:
        result *= value
    return result
