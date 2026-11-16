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
  suite, and a simulated world whose assumptions you set. The world doubles
  as a game: write a matching rule in a few lines and score it against an
  answer key it cannot see.
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
uv run wavelength play rules/article.py
uv run wavelength tournament rules/
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
| The three stages, run on the same people | `strategies.similarity_only`, `filtered_global`, `bidirectional` |
| The three stages as plug-in rules | `rules/most_similar.py`, `rules/dealbreakers_then_greedy.py`, `rules/article.py` |
| Checking and refitting `P(rank)` | `calibration.fit_rank_curve`, and the `curve` argument in `matching.py` |

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

## Play

The simulated world is also a game with one question: can you write a rule
that introduces the right people to each other?

### Write a rule

A rule is a function of what the app can see this week, returning the pairs
to introduce. This is a complete one:

```python
from wavelength.play import rule

@rule("vibes first")
def vibes_first(v):
    s = 0.5 * v.text + 0.3 * v.voice + 0.2 * v.features()
    return v.pair_mutual(s, top=10)
```

`v.text`, `v.photo` and `v.voice` are how alike two people are in what they
say, how they live and how they talk; `v.features()` is agreement on
structured features by what the app has learned each person cares about.
`v.pair_mutual` is the article's method; `v.pair_best` and `v.pair_stable`
are the alternatives. [rules/README.md](rules/README.md) lists everything a
rule can see. The article's own three stages are three of the rules in
[rules/](rules/), a few lines each.

### Score it

```
uv run wavelength play rules/article.py --world today
uv run wavelength tournament rules/ --worlds today,casual,serious,swipe,dystopia --seeds 5 --out leaderboard.json
```

or from Python:

```python
from wavelength.play import run
card = run(vibes_first, world="today", seed=0, people=600, rounds=8)
```

The scorecard is plain data: couples formed at each stage as a share of the
best possible, what happened week by week, how many introductions were
answered, expired or wasted, how many people gave up, and an event log for
a small cast of people.

### Submit it

Add your one file to `rules/` and open a pull request. The tournament runs
on the pull request and attaches the leaderboard. Three steps, spelled out
in [rules/README.md](rules/README.md).

### What the score means

Before any rule runs, the world works out from its hidden truth how far
every possible pair would get if introduced:

| Stage | Meaning | Depends on |
| --- | --- | --- |
| 1 | Both say yes | What each feels on the day against their own bar; no dealbreaker visible in the other's profile |
| 2 | The conversation keeps going | How alike what they say is, how their ways of talking fit, closeness on the features about communicating; someone must make the first move |
| 3 | They meet | The chemistry they share, how alike their lives are; they must live within reach |
| 4 | Both want to meet again | What each feels for the other, chemistry included; no true dealbreaker of either, including ones no profile showed; family approval where it is required |

That table is the answer key. It is fixed by the seed, and so is all the
luck: whether a given person opens the app in a given week is drawn in
advance, per person and per week, never per rule. Two rules in the same
world meet the same people with the same luck.

From the key the engine computes the **best possible**: the most couples
anyone could form, with any information and any amount of time (a maximum
matching). A rule's score is the couples it formed who both want to meet
again, as a share of that. Beside it the scorecard shows what a reader of
perfect profiles, blind to chemistry and luck, would have reached. A rule
far below that benchmark is wasting information it has or lacks information
it could have; the gap from the benchmark to 100% is chemistry, luck and
time, which no rule can see.

A run lasts several weeks. People who are in a couple leave. People who
keep turning introductions down, or keep saying yes and hearing nothing,
open the app less and eventually give up. So a rule that wastes attention
early has fewer people to work with later.

### What a rule cannot do

See the answer. A rule receives a `View` built from a copy of what the app
observed and nothing else; tests check that nothing reachable from it is,
or shares memory with, the hidden truth or the key. Its tables are
read-only. What it returns is checked (pairs must cross the two sides,
nobody twice, nobody who has left, nobody introduced before) and a mistake
is reported in plain words with the round it happened in.

### For a browser

`wavelength.web` is the same thing as plain data: `describe_settings()`
(every setting with its range, default, meaning and basis),
`list_presets()`, `list_rules()`, `run(settings, rule_source_or_name, seed,
people, rounds)` and `compare(...)`. A rule can be sent as source text; a
problem with it comes back as `{"error": {"line": ..., "message": ...}}`
and nothing is raised. The package needs only numpy.

## Simulation

`wavelength simulate` is a closer look at one round. It generates a pool of
made-up people, runs the article's three strategies on the same pool, and
reports how each proposed introduction would have gone, with a calibration
check on the predicted probabilities. It uses the same world and the same
settings as the game. Its purpose is to let you ask "what happens to these
strategies in a world like this?" and to make you say, in numbers, what
"a world like this" means.

The three strategies are similarity only (Stage 1); features and hard
filters with global weights, one-directional (Stage 2); and per-user weights
with bidirectional matching (Stage 3).

This README shows no output and draws no conclusion from any run, on
purpose. Run it and read the numbers next to the assumptions that produced
them.

### Commands

```
uv run wavelength scenarios                      # list the preset worlds
uv run wavelength describe serious               # every field, its value, its meaning
uv run wavelength simulate --scenario today      # one world
uv run wavelength simulate --scenario today --calibrate
uv run wavelength simulate --scenario today --compare casual,serious,swipe,dystopia
uv run wavelength simulate --scenario today --set yes_rate=0.2 --seed 1 --users 800 --trials 10
uv run wavelength simulate --scenario my_world.toml
```

`--seed` makes a run repeatable, `--users` overrides the pool size and
`--trials` sets how many independent pools are drawn and added up. With
`--compare`, the same `--set`, `--users`, `--seed`, `--trials` and
`--calibrate` apply to every scenario, and the result is one table per
measure with a column per scenario.

### What a run reports

First, facts about the generated world, measured before any strategy runs:
the realised yes-rate to a random eligible introduction, the share of
people who answer, how many eligible candidates a typical person has on
each side and what share of them the top N is, and how closely the
matcher's score tracks what people actually feel (a rank correlation,
shown for each channel alone, for features under per-user and global
weights, and next to the most that a perfect read of both profiles could
reach).
Read these first. If a person's top 10 is most of the people they are
eligible for, or the score barely tracks feeling, no comparison of
strategies in that world means much.

Then, per strategy:

| Column | Meaning |
| --- | --- |
| both yes (all) | Share of all proposed pairs in which both people said yes. A pair that expired unanswered counts against it. This is what the product delivers. |
| both yes (answered) | The same, among pairs in which both people answered. Not answering is not a no, so this is the fair test of the matcher. |
| one yes | Share of all proposed pairs with exactly one yes. |
| expired | Share of all proposed pairs in which at least one person never answered. |
| no proposal | Share of users left without an introduction this round. |
| matches/100 | Pairs in which both said yes, per 100 users in the pool. |
| violations | Share of proposed pairs that break a dealbreaker the app knew about. |
| hidden mismatch | Share that looked fine and truly fail a hard filter, because someone did not state what they want as it is. No rule could have seen these. |

Then calibration of the predicted P(mutual), on pairs in which both people
answered: predicted against observed, per bin, and beside them the
both-yes rate the bin would have if the two decisions were independent
(the product of the two sides' observed yes-rates). The last column
separates two different ways the prediction can be wrong: a per-person
curve that is off, or two decisions that move together so that
`P(A) x P(B)` understates the pair however good the curve is.

### Calibration: the article's curve and a fitted one

The article's `P(rank) = 1 - 0.5 x (rank - 1) / (N - 1)` is the default
everywhere and is unchanged in `matching.py`. It was set by hand for one
app. A probability curve is a claim about a particular world, so the
simulation can also fit one:

- `--calibrate` runs the bidirectional matcher on a separate set of pools
  and records, for each introduction and each person who answered, the rank
  they held the other at and whether they said yes. This is only what a
  real system could log.
- `calibration.fit_rank_curve` turns that into N numbers: the yes-rate at
  each rank, adjusted just enough that it never rises as rank gets worse.
- The fitted curve is then used on different pools, with different seeds,
  for both the order of assignment and the predicted probability. The
  report shows the two curves by rank and a calibration table for each.

`matching.mutual_top_n`, `assign_by_probability` and `bidirectional_match`
take the curve as an optional argument, so the same can be done with real
outcomes.

### Scenarios

A scenario is a `Scenario` dataclass (`src/wavelength/scenario.py`). Every
field is a quantity about people, a product or a dating market that you
could argue about without reading code. Each carries its meaning, a
sensible range and the basis of its default; `wavelength describe` prints
all three.

| Field | What it means in the world |
| --- | --- |
| **The market** | |
| `pool_size` | People in one matching pool, such as one city's users in one age band. |
| `majority_share` | The pool has two sides and introductions only cross them. Share of the pool on the larger side; 0.5 is balanced. |
| **What people want** | |
| `archetypes` | The mix of what people want. Each archetype (settling down, relationship-seeking, exploring, casual) has a share of the pool and its own odds for relationship timeline, commitment style, family plans and trust boundaries. This is what makes dealbreakers correlated: people who want marriage more often want children. |
| `trait_shares` | Population share of each value of the eight features that are not hard-filtered. |
| `dealbreaker_shares` | Per core feature, the share of people for whom it is a true dealbreaker (they refuse anyone clearly different) and not just a preference. |
| `personality_dealbreaker_share`, `lifestyle_dealbreaker_share` | The same, for any one personality or lifestyle feature. |
| **How people choose** | |
| `yes_rate` | Share of eligible, randomly chosen people on the other side that a typical person would say yes to if introduced. It describes the person and the product, before any matcher has chosen well. |
| `yes_rate_spread` | How much people differ in their own yes-rate; 0 means everyone is alike. |
| `minority_yes_ratio` | How often the scarce side says yes, relative to the larger side. |
| `desirability_consensus` | How much people agree on who is desirable. At 0 everyone has a different favourite; near 1 attention piles onto a few people. |
| `chemistry_share` | Share of attraction no system can observe. |
| `chemistry_mutuality` | How much of chemistry is felt by both people. Above 0, the two decisions in a pair are not independent. |
| `decision_noise` | Mood and luck on the day: the share of a single decision that has nothing to do with the other person. |
| `attraction_mix` | What personal fit is made of: agreement on values, what people say, how they live, how they talk. Each is a separate hidden trait. |
| `preference_diversity` | How much people differ in what matters to them. At 0 one set of weights describes everybody; at 1 one person's dealbreaker is another's shrug. |
| **After the yes** | |
| `conversation_rate`, `date_rate`, `second_date_rate` | Of eligible, randomly chosen pairs, the share that would pass each later stage. They set how many pairs pass, not which: that follows from the pairs' hidden fit. |
| `leave_after_stage` | How far a couple must get before both stop taking introductions. |
| **Attention over time** | |
| `open_rate`, `open_rate_spread` | How often a typical person opens the app in a week and so answers before an introduction expires, and how much people differ in that. |
| `fatigue` | Share of a person's willingness to open the app lost with each letdown: an introduction they turned down, or a yes that was not returned. |
| `patience` | How many letdowns a typical person takes before giving up. |
| **Culture** | |
| `community_shares` | Shares of the pool in each community: whatever people in this world marry within, when they marry within anything. The names are placeholders. |
| `endogamy_share` | Share of people for whom a partner from their own community is a dealbreaker. |
| `family_approval_share` | Share of people who need their family's approval for a relationship to continue. |
| `family_visibility` | How well the app can see family expectations. |
| `marriage_pressure` | How strongly age pushes people towards wanting marriage. |
| `intent_openness` | Share of people who state what they want, and what they will not accept, as it really is. The rest give the usual answer, and the mismatch surfaces in person. |
| `larger_side_initiates`, `scarce_side_initiates` | Share of each side who will make the first move. If neither person will, the conversation never starts. |
| `distance_tolerance` | How far a typical person will travel to meet, as a share of the area's width. |
| `photos_hidden_share` | Share of people who keep photos hidden until later; the app has no photo signal for them. |
| **What the app knows** | |
| `self_knowledge` | How often the importance people state at onboarding is the importance they act on. |
| `onboarding_reach`, `match_feedback_reach`, `post_date_reach` | Share of features whose importance each source reveals: once at onboarding, and again each time a person explains a decision or talks through a date. |
| `match_feedback_accuracy`, `post_date_accuracy` | How often importance learned from behaviour is the true importance. |
| `feature_coverage` | Profile completeness: share of non-dealbreaker features extracted, across all users. |
| `full_profile_share` | Share of users with every feature extracted. |
| `dealbreakers_known_share` | Share of users whose four hard-filter values are all known. The filters fail closed for everyone else. |
| `extraction_accuracy` | How often an extracted value is the true one. |
| `text_signal_quality`, `photo_signal_quality`, `voice_signal_quality` | How well each channel reflects what a person really says, how they live and how they talk. |
| `intent_visible_in_bio` | How much the way people describe themselves gives away what they want. |

The culture settings are variables, not portraits. No preset here stands
for a country, and no figure in this repository describes a real
population. A world built from sourced figures can be loaded from a preset
file in which every changed field carries its own citation
(`presets.load_preset_file`); none ships with the repository.

How the article's scoring mix relates to these. The article's
`0.25 x text + 0.15 x photo + 0.60 x features` is a property of a rule, not
of the world: it is what `rules/article.py` computes, and a rule is free to
give voice a share. The world's `attraction_mix` is how people actually
feel, and is deliberately different.

Where the defaults come from. Two are figures the article reports about its
own app, both inputs and not outcomes: `feature_coverage` (85% coverage
from conversational extraction; the article gives 30% for surveys, which
the `swipe` and `dystopia` presets use) and `full_profile_share` (about 34%
of users with full extraction). Two lean on Hitsch, Hortacsu and Ariely
(2010), cited by the article, for direction only: `minority_yes_ratio` and
`desirability_consensus`. Every other default is an assumption and is
labelled as one in the code. That includes `yes_rate`, which matters more
than any other single number and which nobody should take from here if
they have their own data.

### Product type

What a yes means depends on the product, so each preset names the one it
assumes.

In an **introductions** product there is no feed. A person talks at length
with the app when they join, and is then offered a small number of chosen
matches one at a time, each with the reasons for it. They answer yes or no;
two people are introduced only when both have said yes; an offer nobody
answers expires after a couple of days. After a date, people come back and
talk about how it went, which is where post-date signals come from. This is
the kind of product the article describes, and the one the matcher was
designed for.

In a **swipe feed** a person judges a long stream of profiles in seconds, a
yes costs nothing, and nobody is asked for a reason.

In the simulation the difference is carried by ordinary fields: a much lower
`yes_rate`, more `decision_noise`, thinner profiles and less behavioural
signal. The simulation still makes one introduction per person per round in
both, so it does not model the volume of a feed.

### Presets

Five named worlds. `today` is the defaults; each of the others is a short
list of overrides in `src/wavelength/presets.py`, with a reason next to
every number.

| Preset | Product | The world it depicts |
| --- | --- | --- |
| `today` | introductions | Today's mix of people: mixed intentions, somewhat more people on one side, a typical person saying yes to about three in ten eligible strangers, imperfect self-knowledge, about a third of profiles complete. |
| `casual` | introductions | Most people are exploring or want something casual. Few firm dealbreakers, people say yes more readily, and attraction is more about how someone comes across. Same market as `today`. |
| `serious` | introductions | A matrimony-style pool: marriage-minded, many firm dealbreakers, a lower yes-rate, family approval and community mattering to many, but high intent, fuller profiles and near-balanced sides. |
| `swipe` | swipe feed | The same people as `today` on a feed: a yes-rate a third as high, noisier decisions, thin form-based profiles, no voice, little behavioural signal, people opening the app less and tiring faster. |
| `dystopia` | introductions | Three seeking for every one sought, most of attraction the same for everyone, a scarce side that accepts half as often, wide variation in yes-rates, low self-knowledge and openness, thin profiles, people who open the app barely one week in two and give up early. Intentions are left as in `today`. |

The inputs that differ between them (these are settings, not results):

| Input | today | casual | serious | swipe | dystopia |
| --- | --- | --- | --- | --- | --- |
| Archetype shares: settling down / relationship-seeking / exploring / casual | 20% / 35% / 30% / 15% | 5% / 15% / 40% / 40% | 55% / 35% / 8% / 2% | 20% / 35% / 30% / 15% | 20% / 35% / 30% / 15% |
| `dealbreaker_shares`: timeline / commitment / family / trust / emotional availability | 0.55 / 0.5 / 0.45 / 0.3 / 0.3 | 0.25 / 0.2 / 0.1 / 0.15 / 0.15 | 0.85 / 0.8 / 0.75 / 0.5 / 0.4 | 0.55 / 0.5 / 0.45 / 0.3 / 0.3 | 0.55 / 0.5 / 0.45 / 0.3 / 0.3 |
| `majority_share` | 0.6 | 0.6 | 0.55 | 0.6 | 0.75 |
| `yes_rate` | 0.3 | 0.4 | 0.2 | 0.1 | 0.3 |
| `yes_rate_spread` | 0.5 | 0.5 | 0.5 | 0.5 | 1 |
| `minority_yes_ratio` | 0.7 | 0.7 | 0.7 | 0.7 | 0.5 |
| `desirability_consensus` | 0.25 | 0.4 | 0.25 | 0.45 | 0.6 |
| `decision_noise` | 0.1 | 0.1 | 0.1 | 0.25 | 0.2 |
| `attraction_mix`: values / vibe / lifestyle / manner | 0.4 / 0.25 / 0.15 / 0.2 | 0.25 / 0.3 / 0.2 / 0.25 | 0.6 / 0.15 / 0.1 / 0.15 | 0.4 / 0.25 / 0.15 / 0.2 | 0.4 / 0.25 / 0.15 / 0.2 |
| `open_rate` | 0.7 | 0.7 | 0.85 | 0.6 | 0.55 |
| `fatigue` | 0.1 | 0.1 | 0.1 | 0.15 | 0.2 |
| `patience` | 8 | 8 | 8 | 8 | 4 |
| `endogamy_share` | 0.15 | 0.15 | 0.4 | 0.15 | 0.15 |
| `family_approval_share` | 0.1 | 0.1 | 0.4 | 0.1 | 0.1 |
| `family_visibility` | 0.3 | 0.3 | 0.7 | 0 | 0.3 |
| `marriage_pressure` | 0.2 | 0.2 | 0.5 | 0.2 | 0.2 |
| `intent_openness` | 0.85 | 0.85 | 0.95 | 0.7 | 0.6 |
| `self_knowledge` | 0.55 | 0.55 | 0.65 | 0.55 | 0.35 |
| `onboarding_reach` | 0.7 | 0.7 | 0.9 | 0.4 | 0.5 |
| `match_feedback_reach` | 0.4 | 0.4 | 0.4 | 0.2 | 0.15 |
| `post_date_reach` | 0.15 | 0.15 | 0.15 | 0.05 | 0.03 |
| `feature_coverage` | 0.85 | 0.85 | 0.9 | 0.3 | 0.3 |
| `full_profile_share` | 0.34 | 0.34 | 0.5 | 0.05 | 0.05 |
| `dealbreakers_known_share` | 0.9 | 0.9 | 0.98 | 0.7 | 0.75 |
| `extraction_accuracy` | 0.9 | 0.9 | 0.9 | 0.9 | 0.75 |
| `text_signal_quality` | 0.6 | 0.6 | 0.6 | 0.4 | 0.3 |
| `photo_signal_quality` | 0.5 | 0.5 | 0.5 | 0.5 | 0.3 |
| `voice_signal_quality` | 0.5 | 0.5 | 0.5 | 0 | 0.3 |

Every preset's own pool is 1200; `play` and `tournament` use 600 unless told otherwise.

The presets were not tuned to produce any outcome. No value was set by
working backwards from the article's 22%, 35% and 68%, and there is no
reason to expect a run to land on them: those were measured on a live app
with real people. The only constraint applied to the worlds themselves is
a floor, checked by tests: each preset must leave couples to be found (a
reader of perfect profiles must be able to form some in a default game),
must produce enough mutual yeses in a default `simulate` run to be read at
all, and must leave a typical person more eligible candidates than the
top-N cutoff. `swipe` and `dystopia` are harsh by design and sit close to
that floor, so rules form few couples in them and scores move a good deal
from seed to seed; use more people, seeds or trials before reading anything
into them.

### Setting your own

Everything can be set three ways.

In Python:

```python
from dataclasses import replace
from wavelength.presets import PRESETS
from wavelength.simulate import run_simulation

world = replace(PRESETS["serious"].scenario, yes_rate=0.15, majority_share=0.7)
report = run_simulation(world, seed=0, calibrate=True)
```

In a TOML or JSON file, which has the same shape as a preset. `base` names
the preset to start from (default `today`); everything else overrides one
field:

```toml
base = "serious"
pool_size = 800
yes_rate = 0.15

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
--set yes_rate=0.2
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
  archetype's odds, independently within the archetype, then nudged towards
  marriage with age by `marriage_pressure`.
- **Three hidden traits, three channels.** What people say, how they live
  and how they talk are drawn independently, each around a few loose
  styles. Text, photos and voice each read one of them, with their own
  noise. No channel stands in for another. How much of what someone wants
  leaks into what they say is `intent_visible_in_bio`: two people can sound
  alike and want opposite lives. This is the article's account of why
  similarity fails, built into the world as a premise.
- **Attraction.** How drawn A is to B is a mix of personal fit (by A's own
  priorities), B's general appeal, and chemistry. The rules here have no
  signal for general appeal, and nothing can observe chemistry. Part of
  chemistry is shared by the pair, so the two decisions are correlated.
- **Priorities differ.** Each person's attention is spread over the
  features in their own way, by `preference_diversity`.
- **Dealbreakers are refusals.** If a feature is a true dealbreaker for A
  and B is clearly different on it, A says no. At an introduction A can
  only act on what B's profile states: the four dealbreaker answers and
  community. Every other dealbreaker, and intent that was not stated as it
  is, surfaces in person and ends things at the last stage.
- **A personal bar.** Each person has a fixed level of attraction above
  which they say yes. It is placed so that they would accept their own
  `yes_rate` share of eligible people chosen at random. Someone shown
  better-than-random candidates therefore says yes more often, and how much
  more is not set anywhere: it depends on how well the rule's choices track
  what the person feels. This is the article's central premise (people
  accept those near the top of their own list) in its weakest form.
- **Later stages are tests of fit, not coin flips.** Each has a score made
  of named hidden quantities plus luck, and a bar that the configured share
  of random eligible pairs clears. See the table under Play.
- **Mood and luck** are drawn once per pair from the seed.
- **Answering has causes.** A personal habit, worn down by letdowns, against
  luck drawn per person and week. An unanswered introduction expires; it is
  not a no. The rules do not know who will answer, only who has.
- **The app records what people state.** Dealbreaker answers are recorded
  as stated or not at all. What is stated is true except where
  `intent_openness` says otherwise.
- **One introduction per person per week.** No messaging model, no second
  offer of the same person, no new arrivals.

The comparison is not circular: nobody's answer is computed from a
matcher's score. Acceptance and everything after it come from true values
and true priorities the matcher never sees, refusals the scorer does not
have, general appeal and chemistry it cannot observe, mood, and a different
mix of signals from the scorer's. Every pair's fate is drawn once before
any rule runs, so the same introduction has the same outcome whichever rule
proposes it.

Known gaps between this world and the method, stated so they are not
mistaken for findings:

- The article's curve says a person's first choice is accepted with
  probability 1. No world with chemistry, mood and an imperfect score
  behaves that way, so the hand-set curve is not expected to be calibrated
  here. That is a statement about applying one app's curve to another
  world, which is what `--calibrate` is for.
- The matchers have no popularity signal. In a world with high
  `desirability_consensus`, much of what could be predicted is invisible to
  all three strategies alike.
- Learned per-user weights are noisy. Whether they track a person's
  feelings better than global weights depends on `preference_diversity`,
  `self_knowledge` and the reach and accuracy of feedback; the "features vs
  feeling" rows show which way it went in the world you ran.
- The best possible assumes everyone answers and there is all the time in
  the world. No rule gets that, so no rule should expect to come near 100%;
  the perfect-profiles benchmark is the fairer comparison.

### What it cannot prove

A simulation built on a premise cannot confirm the premise. This one
assumes that people say yes more readily to those they are more drawn to,
that dealbreakers are real, and that behaviour reveals importance better
than stated preference. A method designed around those ideas is then run in
a world where they hold. What the simulation can show is how much each idea
is worth under stated conditions, where it stops mattering, and whether a
predicted probability is calibrated in a given world. Whether real people
behave like any of these worlds is an empirical question that only real
outcomes can answer. A curve fitted here is fitted to simulated people; it
says nothing about what the curve should be for a real product. The same
goes for the game: a rule that tops the leaderboard has found what works in
these worlds, under these assumptions. That is a reason to test it on real
outcomes, not a substitute for doing so.

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
- **One decay curve for everyone,** hand-set rather than fitted. The curve
  can be replaced (`curve=` in `matching.py`) and fitted
  (`calibration.fit_rank_curve`), but the default is the article's.
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

  calibration.py fit P(accept) by rank from observed decisions

  scenario.py    the assumptions about a world, as documented fields
  presets.py     named worlds, and the loader for cited preset files
  scenario_io.py scenarios as plain data: files and --set overrides
  population.py  hidden truth: who the simulated people are
  decisions.py   hidden truth: who would say yes to whom
  key.py         hidden truth: how far every pair would get (the answer key)
  observation.py what the app has learned; all a rule is built from
  world.py       one pool: truth, key and observations from one seed

  best.py        the most couples possible (maximum matching)
  view.py        what a rule sees, and helpers to turn scores into pairs
  engine.py      several rounds of play, with attention and feedback
  play.py        the rule decorator, run() and the scorecard
  tournament.py  every rule against every world, with spread
  web.py         the same as plain data, for a browser

  strategies.py  the three stages for the one-round simulation
  diagnostics.py facts about a world to read before any score
  simulate.py    one round in detail, with calibration
  report.py      plain-text rendering
  cli.py         the `wavelength` command
rules/           demo rules, one small file each; add yours here
tests/           one file per module, plus property checks on random inputs
docs/PRINCIPLES.md
```

## License

MIT. See [LICENSE](LICENSE).
