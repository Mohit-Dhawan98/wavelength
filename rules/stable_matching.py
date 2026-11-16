"""Gale and Shapley's deferred acceptance, run on the scores the app can compute."""

from wavelength.play import rule


@rule("stable matching")
def stable_matching(v):
    """No two people would both rather have each other, as far as the scores can tell."""
    s = 0.25 * v.text + 0.15 * v.photo + 0.60 * v.features("personal")
    return v.pair_stable(s)
