"""Introduce people at random. The floor every other rule should clear."""

from wavelength.play import rule


@rule("random introductions")
def random_introductions(v):
    """Anyone with anyone, as long as they have not met."""
    luck = (v.text * 0) + ((v.age[:, None] * 7919 + v.age[None, :] * 104729 + v.round) % 97)
    return v.pair_best(luck, among=v.open)
