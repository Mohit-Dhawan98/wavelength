"""Each person's highest-scored candidate, with no dealbreaker check and no second opinion."""

from wavelength.play import rule


@rule("best for me")
def best_for_me(v):
    """Score everyone on all four channels and hand each person their favourite."""
    s = 0.25 * v.text + 0.15 * v.photo + 0.60 * v.features("global")
    return v.pair_best(s, among=v.open)
