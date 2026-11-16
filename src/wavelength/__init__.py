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

A simulated world whose assumptions are named parameters, with a hidden
answer key:

- ``scenario``     the assumptions about a world, as documented fields
- ``presets``      named worlds, and the loader for cited preset files
- ``scenario_io``  scenarios as plain data: files and ``--set`` overrides
- ``population``   hidden truth: who the simulated people are
- ``decisions``    hidden truth: who would say yes to whom
- ``key``          hidden truth: how far every pair would get
- ``observation``  what the app has learned; all a rule is built from
- ``world``        one pool: truth, key and observations from one seed

The game played in that world:

- ``best``         the most couples possible (maximum matching)
- ``view``         what a rule sees, and helpers to turn scores into pairs
- ``engine``       several rounds of play, with attention and feedback
- ``play``         the ``rule`` decorator, ``run`` and the scorecard
- ``tournament``   every rule against every world, with spread
- ``web``          the same as plain data, for a browser

A closer look at one round:

- ``strategies``   the article's three stages as functions of the observations
- ``diagnostics``  facts about a world to read before any score
- ``simulate``     run, score and (optionally) calibrate
- ``report``       plain-text rendering
- ``cli``          the ``wavelength`` command
"""

__version__ = "0.1.0"
