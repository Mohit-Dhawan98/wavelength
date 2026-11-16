"""Stage 2 of the article: rule out known dealbreakers, then each person's favourite."""

from wavelength.play import rule


@rule("dealbreakers then greedy")
def dealbreakers_then_greedy(v):
    """The article's tri-signal score with global weights, one-directional."""
    s = 0.25 * v.text + 0.15 * v.photo + 0.60 * v.features("global")
    return v.pair_best(s)  # v.ok, the default, already excludes known dealbreakers
