"""The most couples anyone could form: maximum matching over the answer key.

A rule's raw count of couples says little by itself. Forty couples is
excellent in a world where forty-five were possible and poor in one where
three hundred were. So each world's answer key is first turned into a
ceiling: treat every pair that would reach a given stage as an edge between
the two sides, and find the largest set of edges that uses nobody twice.
That number is what any rule, with any information and any amount of time,
could at most achieve, and a rule's headline score is its share of it.

The matching is found with the Hopcroft-Karp algorithm, written out here so
the package needs nothing beyond numpy. A test checks it against brute
force on small graphs.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

_FREE = -1


def maximum_matching(adjacency: Sequence[Sequence[int]], n_right: int) -> list[tuple[int, int]]:
    """A largest matching in a bipartite graph (Hopcroft-Karp).

    ``adjacency[u]`` lists the right-hand vertices that left-hand vertex
    ``u`` may be paired with. Returns ``(left, right)`` pairs.
    """
    n_left = len(adjacency)
    match_left = [_FREE] * n_left
    match_right = [_FREE] * n_right
    while True:
        # Breadth-first: layer the graph by distance from the free left vertices.
        layer = [_FREE] * n_left
        queue = deque(u for u in range(n_left) if match_left[u] == _FREE)
        for u in queue:
            layer[u] = 0
        reachable_free_right = False
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                partner = match_right[v]
                if partner == _FREE:
                    reachable_free_right = True
                elif layer[partner] == _FREE:
                    layer[partner] = layer[u] + 1
                    queue.append(partner)
        if not reachable_free_right:
            break
        # Depth-first along the layers, without recursion, to flip augmenting paths.
        position = [0] * n_left
        for root in range(n_left):
            if match_left[root] != _FREE:
                continue
            path = [root]
            while path:
                u = path[-1]
                if position[u] == len(adjacency[u]):
                    layer[u] = _FREE  # dead end; do not come back
                    path.pop()
                    continue
                v = adjacency[u][position[u]]
                position[u] += 1
                partner = match_right[v]
                if partner == _FREE:
                    for node in reversed(path):  # augment: re-pair everyone on the path
                        previous = match_left[node]
                        match_left[node], match_right[v] = v, node
                        v = previous
                    break
                if layer[partner] == layer[u] + 1:
                    path.append(partner)
    return [(u, v) for u, v in enumerate(match_left) if v != _FREE]


def most_couples(pairs: np.ndarray, majority: np.ndarray) -> int:
    """The largest number of disjoint couples among the True entries of ``pairs``.

    ``pairs`` is an ``(n, n)`` symmetric boolean matrix over people;
    ``majority`` says which side each person is on. Only pairs across the
    two sides count.
    """
    left = np.flatnonzero(majority)
    right = np.flatnonzero(~majority)
    block = pairs[np.ix_(left, right)]
    adjacency = [np.flatnonzero(row).tolist() for row in block]
    return len(maximum_matching(adjacency, right.size))
