"""Stage 3 of the article, approximated: per-user weights, then pairs who are in each other's top ten.

Rebuilt from the article's description. It is not the production implementation.
"""

from wavelength.play import rule


@rule("the article's rule, approximated")
def article(v):
    """0.25 text + 0.15 photo + 0.60 features by each person's own weights; mutual top 10."""
    s = 0.25 * v.text + 0.15 * v.photo + 0.60 * v.features("personal")
    return v.pair_mutual(s, top=10)
