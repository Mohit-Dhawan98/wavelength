# How matching should work

Eleven principles behind this implementation. The substance comes from
[the article](https://themlnerd.substack.com/p/why-the-girl-you-want-doesnt-want);
each principle points at the function that embodies it.

## 1. The objective is two-sided

A film does not turn you down. A person can. Recommending B to A is only
useful if B would also choose A, and A liking B says nothing about that. So
the thing to optimise is not "how good is this candidate for this user" but
"how likely are both to say yes". Every later principle follows from taking
that seriously.

Code: `matching.bidirectional_match`, which only ever considers pairs that
appear in both people's lists.

## 2. Dealbreakers are filters, not features

A weighted score averages. Two people who agree on everything except whether
to have children still average out to a high score, and the one disagreement
that decides the outcome disappears into it. Raising the weight does not fix
this, because enough agreement elsewhere can always outvote it. Dealbreakers
are therefore checked before scoring, and a pair that fails is never scored
at all. When a dealbreaker value is unknown, this implementation refuses the
pair instead of guessing.

Code: `filters.hard_filter_violations` and `filters.HARD_FILTERS`.

## 3. Distance is graduated

Different is not the same as incompatible. Someone who raises problems
directly and someone who needs time first both deal with conflict, and can
suit each other well. Someone direct with someone who avoids conflict
entirely is a much harder pairing. A yes/no compatibility flag treats those
two cases the same. Each feature therefore has named categories and an
explicit distance between every pair of them, from 0.0 to 1.0.

Code: `features.Feature.distance` and the matrices in `features.py`.

## 4. No single signal sees the whole person

How people describe themselves captures tone. How they appear to live
captures what they do, which often differs from what they say. Structured
preferences capture what they need. Similar-sounding profiles can hide
opposite life plans, so similarity is necessary and nowhere near sufficient.
The score combines all three and gives most of the weight to structured
features, because those decide outcomes.

Code: `scoring.tri_signal_score`. The text and visual signals in this
repository are plain vectors supplied by the caller.

## 5. Importance is personal, so scores are directed

The same feature is one person's dealbreaker and another's afterthought. A
single global weight is too low for the first and too high for the second.
Once each person has their own weights, the same pair gets two scores: the
gap between them is identical in both directions, but each person's weights
make it loom large or barely register. Treating a pair as one symmetric
number hides exactly the cases where one person is keen and the other is
not.

Code: `scoring.directed_score`, which takes the viewer's weights as an
argument.

## 6. Behaviour beats stated preference, and every weight is auditable

What people say they want at signup is a guess about themselves. Rejecting a
real match for a stated reason is evidence. Reflecting on a real date is
better evidence still. When these disagree, the stronger source wins, then
the more confident signal, then the more recent. Nothing is overwritten: the
profile keeps every signal, derives the current weight from them, and can
show which signal set any weight and what the person said.

Code: `weights.WeightProfile.resolved` and `weights.WeightProfile.explain`.

## 7. Optimise mutual probability, not individual rank

Being someone's first choice is worth little if they are your fortieth.
Keep only the pairs where each person is near the top of the other's list,
estimate the chance that both accept, and assign the most likely pairs
first, one match per person per round. Some people will not get their own
favourite; they get the best pairing where the interest runs both ways. Some
will get no match this round, which is better than an introduction expected
to fail: being ignored after saying yes teaches people the system does not
understand them.

Code: `matching.mutual_top_n`, `matching.mutual_probability` and
`matching.assign_by_probability`.

## 8. Be honest about calibration and limits

A success probability is a claim that can be checked: of the pairs given
0.8, about 80% should work out. The rank-decay formula here is a simple
hand-set curve that assumes the two decisions are independent. That is good
enough to order pairs and not automatically good enough to believe as a
number: before trusting it, compare predicted against observed outcomes on
real data. The same applies to the method as a whole: the independence
assumption, the cost of ranking from every user's side, and thin data for
new users are stated limits, not footnotes.

Code: `matching.acceptance_probability` and `matching.mutual_probability`,
whose docstrings state the assumption.

## 9. State a world's assumptions as named, arguable parameters

A simulation cannot tell you how real people behave. It can tell you what
follows from what you assumed, and that is only useful if the assumptions
are the easiest part to read and to change. So every assumption about the
simulated world is a named field with a real-world meaning: what share of
the pool wants something casual, how many people a typical person says yes
to, how often what people say matches what they do. A number someone can
disagree with is worth more than a constant buried in a generator, and a
uniform random draw is an assumption too, only an unexamined one. Worlds
that differ (a casual app, a matrimony service, a broken market) are small
sets of overrides with a reason beside each. The people's real preferences
and answers are kept hidden from the matcher, their answers are never
computed from the matcher's own score, and the output is read as a property
of the stated world, not as evidence about the real one.

Code: `scenario.Scenario`, whose fields carry their meaning, sensible range
and basis; `presets.PRESETS`; and `decisions.decide`, the hidden rule for
who says yes.

## 10. A probability curve belongs to one world: fit it there, check it elsewhere

"People accept their first choice almost always and their tenth half the
time" is not a fact about people. It is a fact, if it is one, about a
particular product, population and matcher. A product that sends a few
considered introductions and a feed that shows hundreds of profiles will
have different curves; so will a good score and a poor one, because rank 1
by a poor score is not much of a first choice. A hand-set curve carried
into a new setting is a guess.

So the curve is an argument, not a constant. Fit it from decisions that
were actually observed, keep the fit simple enough to read (one number per
rank, never rising as rank gets worse), and judge it only on data it was
not fitted on. Then check the other assumption separately: the product
`P(A) x P(B)` can be wrong even when each person's curve is right, if the
two decisions move together. Comparing the observed both-yes rate with the
product of the two observed yes-rates shows which of the two is at fault.

Code: the `curve` argument of `matching.mutual_top_n`,
`matching.assign_by_probability` and `matching.bidirectional_match`;
`calibration.fit_rank_curve`; and `simulate.run_simulation(calibrate=True)`,
which fits on one set of pools and reports on another.

## 11. Grade a matcher against an answer key it did not write

"We assumed 30% of matches work out" is not a test of anything: whatever
the matcher does, 30% work out. A matcher can only be graded against
outcomes that were settled without it. So the world decides first, from
what its people are really like, how far every possible pair would get:
whether both would say yes, whether they would keep talking, whether they
would meet, whether both would want to meet again. Each step depends on
named things about the two people, and all the luck is drawn once, in
advance. Rates in the settings say how many pairs pass each step; they do
not say which.

Then the score has a meaning. From the key, work out the most couples
anyone could have formed, and report a matcher's couples as a share of
that. Beside it, report what a reader of perfect profiles could have
reached without seeing chemistry or luck, so that "the rule is weak" and
"the information is not there" can be told apart. Give every matcher the
same people and the same luck, so they differ only in what they chose. And
never let the matcher see the key: hand it a copy of what the app observed
and nothing else, and test that.

This is still a simulation. The key is true of the simulated world and of
nothing else. What it buys is that within that world the grading is honest.

Code: `key.build_key`, the stages; `best.most_couples`, the ceiling;
`engine.Arena`, which keeps truth and view apart; `view.View`, all a rule
gets; and `play.scorecard`.
