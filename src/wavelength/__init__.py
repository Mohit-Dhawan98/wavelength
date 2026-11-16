"""Wavelength: a reference implementation of mutual-match matchmaking.

This package accompanies the article "Why The Girl You Want Doesn't Want You
Back" (https://themlnerd.substack.com/p/why-the-girl-you-want-doesnt-want).
It is a clean-room implementation of the article's ideas, not the production
system the article describes.

Module map, in the order the pipeline runs:

- ``features``   categorical taxonomy and graduated distance matrices
- ``filters``    hard filters, applied before any scoring
- ``weights``    per-user importance weights with provenance
- ``scoring``    tri-signal, per-viewer (asymmetric) scores
- ``matching``   mutual top-N, success probability, greedy assignment
- ``baselines``  similarity-only and one-directional strategies
"""

__version__ = "0.1.0"
