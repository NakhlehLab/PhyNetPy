from math import sqrt

def partials_index(n):
    return int(.5 * (n - 1) * (n + 2))


def undo_index(num):
    a = 1
    b = 1
    c = -2 - 2 * num
    d = (b ** 2) - (4 * a * c)
    sol = (-b + sqrt(d)) / (2 * a)
    n = int(sol)
    r = num - partials_index(n)

    return [n, r]

def map_nr_to_index(n, r):
    """
    Takes an (n,r) pair and maps it to a 1d vector index

    (1,0) -> 0
    (1,1) -> 1
    (2,0) -> 2
    ...
    """
    starts = int(.5 * (n - 1) * (n + 2))
    return starts + r