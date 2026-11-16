"""Introduce people at random. The floor every other rule should clear."""

from wavelength.play import rule


@rule("random introductions")
def random_introductions(v):
    """Anyone with anyone, as long as they have not met."""
    return v.pair_best(v.random(), among=v.open)
