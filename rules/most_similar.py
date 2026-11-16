"""Stage 1 of the article: whoever sounds most like you. No dealbreakers checked."""

from wavelength.play import rule


@rule("most similar")
def most_similar(v):
    """Each person gets the person whose words are most like their own."""
    return v.pair_best(v.text, among=v.open)
