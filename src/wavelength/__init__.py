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

- ``calibration``  fit ``P(accept | rank)`` from observed decisions

The simulation, which runs those strategies in a made-up world whose
assumptions are stated as named parameters:

- ``scenario``     the assumptions about a world, as documented fields
- ``presets``      named worlds: today, casual, serious, swipe, dystopia
- ``scenario_io``  scenarios as plain data: files and ``--set`` overrides
- ``population``   hidden truth: who the simulated people are
- ``decisions``    hidden truth: who would say yes to whom
- ``observation``  what the app has learned; all a matcher sees
- ``world``        one pool: truth and observations from one seed
- ``strategies``   the three strategies, as functions of the observations
- ``diagnostics``  facts about a world to read before any score
- ``simulate``     run, score and (optionally) calibrate
- ``report``       plain-text rendering
- ``cli``          the ``wavelength`` command
"""

__version__ = "0.1.0"
