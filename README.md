# Wavelength

Most matching systems look for the best match for each user. This one looks
for the match where two people are each other's best, because a match only
counts when both say yes.

This repository is a small reference implementation of the ideas in
[Why The Girl You Want Doesn't Want You Back: how we 3x'd match acceptance by
switching from "best match" to "mutual match"](https://themlnerd.substack.com/p/why-the-girl-you-want-doesnt-want)
by Mohit Dhawan (2026-02-01).

## What this is and is not

- It is a clean-room reference implementation, written to make the article's
  ideas concrete and testable: a few small typed Python modules and a test
  suite.
- It has no real users, no LLM calls and no network access. The text and
  visual signals are plain vectors you pass in, not embeddings of any text
  or analysis of any photo.
- It is not the production system. That code belongs to a client and is not
  public. The article shares its architecture with permission; this
  repository shares none of its code or data.
- It makes no performance claims of its own. The results in the article
  (22%, 35% and 68% mutual acceptance across the three stages) were measured
  on a live app and are the article's. Nothing here reproduces or tests
  them; the tests check that the algorithm does what the article describes.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```
uv sync
uv run pytest
```

## How the code maps to the article

| Article section | Code |
| --- | --- |
| When Embeddings Miss Dealbreakers (Stage 1) | `baselines.similarity_scores` |
| Structured Features from Conversations: taxonomy and distance matrices | `features.py` |
| What Photos Reveal: `0.25 x text + 0.15 x visual + 0.60 x features` | `scoring.tri_signal_score`, `scoring.directed_score` |
| Hard Filters: The Circuit Breaker | `filters.HARD_FILTERS`, `filters.hard_filter_violations` |
| Why Global Weights Fail: tiers, sources, provenance | `weights.WeightProfile`, `weights.WeightSignal` |
| Asymmetric scoring, Score(A->B) != Score(B->A) | `scoring.directed_score`, `scoring.score_matrix` |
| Bidirectional Matching, steps 1 to 4 | `matching.rank_matrix`, `mutual_top_n`, `acceptance_probability`, `assign_by_probability` |
| Greedy one-directional matching (Stage 2) | `baselines.one_directional_greedy` |

Constants taken directly from the article: the conflict-resolution distance
matrix; the hard-filter thresholds (timeline 0.2; commitment, family and
trust 0.3); the signal mix (0.25 / 0.15 / 0.60); the tier bands (alpha
9.5-10, beta 5.0-7.0, gamma 1.0-3.0); the source order (onboarding, match
feedback, post-date feedback); N = 10; and the rank decay
`P(rank) = 1 - 0.5 x (rank - 1) / (N - 1)`. The tests pin each of these,
along with the article's worked examples.

Everything else is illustrative: the other eleven distance matrices (apart
from two commitment-style distances the article quotes), most of the global
default weights. The article
describes 26 features; this repository carries 12.

See [docs/PRINCIPLES.md](docs/PRINCIPLES.md) for the reasoning behind the
design, one principle at a time.

## Limitations of the approach

These are the article's own, and they apply to this implementation too.

- **Independence.** `P(mutual) = P(A accepts) x P(B accepts)` treats the two
  decisions as independent. Real decisions are correlated (shared context,
  reciprocity). The product is useful for ordering pairs and should not be
  trusted as a calibrated probability without checking against real
  outcomes.
- **Scaling.** Ranking every user's candidates and intersecting the lists is
  O(n x k). The article reports this breaking its latency budget around 200K
  users. This implementation is worse: it scores all pairs densely in memory
  (O(n squared)) because that is the clearest way to show the idea. There is
  no candidate generation or approximate nearest-neighbour stage.
- **Cold start.** A user with no behavioural signals is scored with global
  default weights until feedback arrives, and a user with an unknown
  dealbreaker is not matched at all (the hard filter fails closed).
- **Features are treated as independent,** although real priorities are
  correlated. The article names this as future work.
- **One decay curve for everyone,** hand-set rather than fitted.
- **Not Gale-Shapley.** The assignment is greedy over mutual top-N pairs and
  makes no stability guarantee.

## Layout

```
src/wavelength/
  features.py    taxonomy and graduated distance matrices
  filters.py     hard filters, applied before scoring
  weights.py     per-user weights with tiers and provenance
  scoring.py     tri-signal, per-viewer scoring
  matching.py    mutual top-N, success probability, greedy assignment
  baselines.py   similarity-only and one-directional strategies
tests/           one file per module, plus property checks on random inputs
docs/PRINCIPLES.md
```

## License

MIT. See [LICENSE](LICENSE).
