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
  ideas concrete and testable: a few small typed Python modules, a test
  suite, and a simulation whose assumptions you set.
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
  The simulation's output describes whatever made-up world you configure,
  and is not a claim about real users either.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```
uv sync
uv run pytest
uv run wavelength scenarios
uv run wavelength simulate --scenario today
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
| The three stages, run on the same people | `simulate.similarity_only`, `filtered_global`, `bidirectional` |

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

## Simulation

The simulation generates a pool of made-up people, runs the article's three
strategies on the same pool, and reports how each proposed introduction
would have gone. Its purpose is to let you ask "what happens to these
strategies in a world like this?" and to make you say, in numbers, what
"a world like this" means.

The three strategies are similarity only (Stage 1); features and hard
filters with global weights, one-directional (Stage 2); and per-user weights
with bidirectional matching (Stage 3). For each one the report gives the
share of proposed pairs accepted by both people, the share accepted by
exactly one, the share of users left without a proposal, successful matches
per 100 users, hard-filter violations measured on people's true values, and
a calibration table comparing predicted P(mutual) with what happened.

This README shows no output and draws no conclusion from any run, on
purpose. Run it and read the numbers next to the assumptions that produced
them.

### Commands

```
uv run wavelength scenarios                      # list the preset worlds
uv run wavelength describe serious               # every field, its value, its meaning
uv run wavelength simulate --scenario today      # one world
uv run wavelength simulate --scenario today --compare casual,serious,dystopia
uv run wavelength simulate --scenario today --set pickiness=0.08 --seed 1 --users 400 --trials 10
uv run wavelength simulate --scenario my_world.toml
```

`--seed` makes a run repeatable, `--users` overrides the pool size and
`--trials` sets how many independent pools are drawn and added up. With
`--compare`, the same `--set`, `--users`, `--seed` and `--trials` apply to
every scenario, and the result is one table per measure with a column per
scenario.

### Scenarios

A scenario is a `Scenario` dataclass (`src/wavelength/scenario.py`). Every
field is a quantity about people or about a dating market that you could
argue about without reading code. Each carries its meaning, a sensible
range and the basis of its default; `wavelength describe` prints all three.

| Field | What it means in the world |
| --- | --- |
| `pool_size` | People in one matching pool, such as one city's users in one age band. |
| `majority_share` | The pool has two sides and introductions only cross them. Share of the pool on the larger side; 0.5 is balanced. |
| `active_share` | Share of people who respond to an introduction at all. The rest are profiles that never answer, and the matcher cannot tell. |
| `archetypes` | The mix of what people want. Each archetype (settling down, relationship-seeking, exploring, casual) has a share of the pool and its own odds for relationship timeline, commitment style, family plans and trust boundaries. This is what makes dealbreakers correlated: people who want marriage more often want children. |
| `trait_shares` | Population share of each value of the eight features that are not hard-filtered. |
| `dealbreaker_shares` | Per core feature, the share of people for whom it is a true dealbreaker (they refuse anyone clearly different) and not just a preference. |
| `personality_dealbreaker_share`, `lifestyle_dealbreaker_share` | The same, for any one personality or lifestyle feature. |
| `pickiness` | Share of the other side a typical person would say yes to. Smaller is pickier. |
| `pickiness_spread` | How much people differ in pickiness; 0 means everyone is alike. |
| `minority_yes_ratio` | How often the scarce side says yes, relative to the larger side. |
| `desirability_consensus` | How much people agree on who is desirable. At 0 everyone has a different favourite; near 1 attention piles onto a few people. |
| `chemistry_share` | Share of attraction no system can observe. |
| `attraction_mix` | What personal fit is made of: agreement on values, how alike two people sound, how alike their lives are. |
| `self_knowledge` | How often the importance people state at onboarding is the importance they act on. |
| `onboarding_reach`, `match_feedback_reach`, `post_date_reach` | Share of features whose importance each source has said anything about. |
| `match_feedback_accuracy`, `post_date_accuracy` | How often importance learned from behaviour is the true importance. |
| `feature_coverage` | Profile completeness: share of non-dealbreaker features extracted, across all users. |
| `full_profile_share` | Share of users with every feature extracted. |
| `dealbreakers_known_share` | Share of users whose four hard-filter values are all known. The filters fail closed for everyone else. |
| `extraction_accuracy` | How often an extracted value is the true one. |
| `text_signal_quality`, `photo_signal_quality` | How well the text and photo signals reflect how a person really comes across and lives. |
| `intent_visible_in_bio` | How much the way people describe themselves gives away what they want. |

Where the defaults come from. Two are figures the article reports about its
own app: `feature_coverage` (85% coverage from conversational extraction;
the article gives 30% for surveys, which is a useful alternative setting)
and `full_profile_share` (about 34% of users with full extraction). Two
lean on Hitsch, Hortacsu and Ariely (2010), cited by the article, for
direction only: `minority_yes_ratio` and `desirability_consensus`. Every
other default is an assumption and is labelled as one in the code.

### Presets

Four named worlds. `today` is the defaults; each of the others is a short
list of overrides in `src/wavelength/presets.py`, with a reason next to
every number.

| Preset | The world it depicts |
| --- | --- |
| `today` | A mainstream app resembling now: mixed intentions, somewhat more people on one side, moderate pickiness, imperfect self-knowledge, about a third of profiles complete. |
| `casual` | Most people are exploring or want something casual. Few firm dealbreakers, people say yes more readily, and attraction is more about how someone comes across. Same market as `today`. |
| `serious` | A matrimony-style pool: marriage-minded, many firm dealbreakers, pickier, but high intent, fuller profiles and near-balanced sides. |
| `dystopia` | Four seeking for every one sought, everyone wanting the same few people, wide variation in pickiness, low self-knowledge, thin profiles, most accounts inactive. Intentions are left as in `today`. |

The inputs that differ between them (these are settings, not results):

| Input | today | casual | serious | dystopia |
| --- | --- | --- | --- | --- |
| Archetype shares: settling down / relationship-seeking / exploring / casual | 20% / 35% / 30% / 15% | 5% / 15% / 40% / 40% | 55% / 35% / 8% / 2% | 20% / 35% / 30% / 15% |
| `dealbreaker_shares`: timeline / commitment / family / trust / emotional availability | 0.55 / 0.5 / 0.45 / 0.3 / 0.3 | 0.25 / 0.2 / 0.1 / 0.15 / 0.15 | 0.85 / 0.8 / 0.75 / 0.5 / 0.4 | 0.55 / 0.5 / 0.45 / 0.3 / 0.3 |
| `pickiness` | 0.1 | 0.2 | 0.06 | 0.05 |
| `pickiness_spread` | 0.5 | 0.5 | 0.5 | 1 |
| `majority_share` | 0.6 | 0.6 | 0.55 | 0.8 |
| `minority_yes_ratio` | 0.7 | 0.7 | 0.7 | 0.25 |
| `active_share` | 0.7 | 0.7 | 0.85 | 0.35 |
| `desirability_consensus` | 0.25 | 0.4 | 0.25 | 0.7 |
| `attraction_mix`: values / vibe / lifestyle | 0.5 / 0.3 / 0.2 | 0.3 / 0.4 / 0.3 | 0.65 / 0.2 / 0.15 | 0.5 / 0.3 / 0.2 |
| `self_knowledge` | 0.55 | 0.55 | 0.65 | 0.35 |
| `onboarding_reach` | 0.7 | 0.7 | 0.9 | 0.5 |
| `match_feedback_reach` | 0.4 | 0.4 | 0.4 | 0.15 |
| `post_date_reach` | 0.15 | 0.15 | 0.15 | 0.03 |
| `feature_coverage` | 0.85 | 0.85 | 0.9 | 0.3 |
| `full_profile_share` | 0.34 | 0.34 | 0.5 | 0.05 |
| `dealbreakers_known_share` | 0.9 | 0.9 | 0.98 | 0.6 |
| `extraction_accuracy` | 0.9 | 0.9 | 0.9 | 0.75 |
| `text_signal_quality` | 0.6 | 0.6 | 0.6 | 0.3 |
| `photo_signal_quality` | 0.5 | 0.5 | 0.5 | 0.3 |

The presets were not tuned to produce any outcome. In particular they were
not tuned to reproduce the article's 22%, 35% and 68%, and there is no
reason to expect a run to land on them: those were measured on a live app
with real people.

### Setting your own

Everything can be set three ways.

In Python:

```python
from dataclasses import replace
from wavelength.presets import PRESETS
from wavelength.simulate import run_simulation

world = replace(PRESETS["serious"].scenario, pickiness=0.03, majority_share=0.7)
report = run_simulation(world, seed=0)
```

In a TOML or JSON file, which has the same shape as a preset. `base` names
the preset to start from (default `today`); everything else overrides one
field:

```toml
base = "serious"
pool_size = 300
pickiness = 0.04

[dealbreaker_shares]
family_plans = 0.9

[archetypes.settling_down]
share = 0.70

[archetypes.relationship_seeking]
share = 0.20
```

On the command line, one field at a time, with dotted paths into the same
structure:

```
--set pickiness=0.08
--set dealbreaker_shares.family_plans=0.9
--set archetypes.casual.share=0.5
--set archetypes.casual.relationship_timeline.casual=0.9
```

Shares that belong to one distribution must sum to 1. A file has to say so
explicitly; `--set` on a single share rescales its siblings proportionally.

### How the simulated world works

The truth is hidden from the matcher. Each simulated person has true
feature values, true importance weights, a true "vibe" and lifestyle, and a
true answer to every possible introduction. A strategy sees none of that. It
sees what an app could have extracted: feature values with gaps and
mistakes, text and photo signals of limited quality, and weights resolved
from simulated onboarding, match-feedback and post-date signals through the
same `WeightProfile` logic as the rest of this repository.

The modelling assumptions, all of them:

- **Two sides.** Everyone is on one of two sides and is only introduced to
  the other. This is the simplest market in which scarcity can exist; it
  does not represent everyone.
- **Archetypes.** A person's four dealbreaker values are drawn from their
  archetype's odds, independently within the archetype.
- **Style.** How people sound and live comes from one of a few "styles"
  that are independent of archetype, plus however much `intent_visible_in_bio`
  lets through. So two people can sound alike and want opposite lives. This
  is the article's account of why similarity fails, built into the world as
  a premise.
- **Attraction.** How drawn A is to B is a mix of personal fit (by A's true
  weights), B's general appeal (which the matchers here have no signal for),
  and chemistry (which nothing can observe).
- **Dealbreakers are refusals.** If a feature is a true dealbreaker for A
  and B is clearly different on it, A says no, whatever else is right.
- **A personal bar.** Each person says yes to their own top share of the
  other side, set by `pickiness`. This assumes the article's central premise:
  people accept those near the top of their own list.
- **Inactive people never answer,** and the matcher does not know who they
  are.
- **Dealbreaker values are known correctly or not at all.** In reality
  people also misstate them; that is left out so that a filter violation in
  the report is always the strategy's doing.
- **One round.** No messaging, no dates, no learning between rounds, no
  one leaving.

The comparison is not circular: nobody's answer is computed from a
matcher's score. Acceptance comes from true values and true weights the
matcher never sees, a refusal rule the scorer does not have, general appeal
and chemistry it cannot observe, and a different mix of signals from the
scorer's. Every decision is drawn once before any strategy runs, so the same
introduction has the same outcome whichever strategy proposes it.

### What it cannot prove

A simulation built on a premise cannot confirm the premise. This one
assumes that people say yes to those near the top of their own list, that
dealbreakers are real, and that behaviour reveals importance better than
stated preference. A method designed around those ideas is then run in a
world where they hold. What the simulation can show is how much each idea is
worth under stated conditions, where it stops mattering, and whether a
predicted probability is calibrated in a given world. Whether real people
behave like any of these worlds is an empirical question that only real
outcomes can answer.

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

  scenario.py    the assumptions about a world, as documented fields
  presets.py     named worlds: today, casual, serious, dystopia
  scenario_io.py scenarios as plain data: files and --set overrides
  population.py  hidden truth: who the simulated people are
  decisions.py   hidden truth: who would say yes to whom
  observation.py what the app has learned; all a matcher sees
  simulate.py    the three strategies on the same people, and the score
  report.py      plain-text rendering
  cli.py         the `wavelength` command
tests/           one file per module, plus property checks on random inputs
docs/PRINCIPLES.md
```

## License

MIT. See [LICENSE](LICENSE).
