"""Fitting ``P(accept | rank)`` to observed decisions.

A probability curve is a claim about a specific world: how often people in
it say yes to their first choice, their fifth, their tenth. The article's
curve was set by hand for one app. Whether it holds anywhere else is a
question for data, so this module fits the curve from decisions and leaves
the checking to held-out data (see ``simulate.run_simulation``).

The fit is the simplest one that can be read off a table: the observed
yes-rate at each rank, adjusted only as far as needed to make the curve
never rise as rank gets worse (isotonic regression by pooling adjacent
violators). There are N numbers and nothing else.

What the fit cannot repair is the product ``P(A accepts) x P(B accepts)``.
A fitted per-person curve can be right at every rank and the product still
wrong, if the two decisions in a pair are correlated. The calibration table
reports both so the two failures can be told apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wavelength.matching import TOP_N, article_curve


@dataclass
class RankOutcomes:
    """How many decisions were seen at each rank, and how many were a yes."""

    top_n: int = TOP_N
    seen: np.ndarray = field(init=False)
    yes: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.seen = np.zeros(self.top_n, dtype=np.int64)
        self.yes = np.zeros(self.top_n, dtype=np.int64)

    def add(self, rank: int, accepted: bool) -> None:
        """Record one person's decision about the candidate at ``rank`` in their list."""
        if not 1 <= rank <= self.top_n:
            raise ValueError(f"rank {rank} is outside 1..{self.top_n}")
        self.seen[rank - 1] += 1
        self.yes[rank - 1] += bool(accepted)


def _non_increasing(values: list[float], weights: list[float]) -> list[float]:
    """Weighted least-squares fit of a sequence that never rises (pool adjacent violators)."""
    blocks: list[list[float]] = []  # [mean, weight, length]
    for value, weight in zip(values, weights):
        blocks.append([value, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            mean, w, length = blocks.pop()
            previous = blocks[-1]
            total = previous[1] + w
            previous[0] = (previous[0] * previous[1] + mean * w) / total
            previous[1] = total
            previous[2] += length
    return [block[0] for block in blocks for _ in range(int(block[2]))]


def fit_rank_curve(outcomes: RankOutcomes) -> tuple[float, ...]:
    """A non-increasing ``P(accept | rank)`` fitted to the outcomes.

    Each rank's rate is ``(yes + 1) / (seen + 2)``, so that a rank seen a
    handful of times is not declared certain or impossible. A rank with no
    observations takes the value of the nearest rank that has some. With no
    observations at all there is nothing to fit, and the article's curve is
    returned unchanged.
    """
    observed = [r for r in range(outcomes.top_n) if outcomes.seen[r] > 0]
    if not observed:
        return article_curve(outcomes.top_n)
    rates = [(outcomes.yes[r] + 1) / (outcomes.seen[r] + 2) for r in observed]
    fitted = _non_increasing(rates, [float(outcomes.seen[r]) for r in observed])
    nearest = [min(observed, key=lambda r: (abs(r - rank), r)) for rank in range(outcomes.top_n)]
    return tuple(float(fitted[observed.index(r)]) for r in nearest)
