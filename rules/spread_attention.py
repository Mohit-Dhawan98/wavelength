"""The article's rule, but give people who keep saying no or going quiet a rest."""

from wavelength.play import rule


@rule("spread attention")
def spread_attention(v):
    """Skip anyone who declined twice lately or let two introductions in a row expire."""
    s = 0.25 * v.text + 0.15 * v.photo + 0.60 * v.features("personal")
    resting = (v.history.recent_declines >= 2) | (v.history.idle_rounds >= 2)
    among = v.ok & ~resting[:, None] & ~resting[None, :]
    return v.pair_mutual(s, top=10, among=among)
