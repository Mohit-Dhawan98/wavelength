# Rules

A rule is a few lines of Python that decide who gets introduced to whom.
Each file in this directory is one rule. They are all played against the
same simulated worlds and scored against an answer key none of them can
see.

## Submit a rule in three steps

1. **Write it.** Copy a file here, rename it, and change the function. A
   rule takes a view `v` of what the app can see this week and returns the
   pairs to introduce:

   ```python
   from wavelength.play import rule


   @rule("vibes first")
   def vibes_first(v):
       s = 0.5 * v.text + 0.3 * v.voice + 0.2 * v.features()
       return v.pair_mutual(s, top=10)
   ```

2. **Score it.**

   ```
   uv run wavelength play rules/vibes_first.py
   uv run wavelength tournament rules/
   ```

   The first prints your rule's scorecard in one world. The second plays
   every rule here in every preset world for several seeds and prints the
   table.

3. **Open a pull request** that adds your one file. Rules from anyone are
   welcome. A rule is code, so the owner reads and approves each pull
   request from an outside contributor before the automated scoring runs
   on it. Once approved, the tournament runs on the pull request and
   attaches the leaderboard to it.

Give the rule a name nobody else has used, and a docstring saying in one
line what it does. Keep it to numpy and what `v` offers.

## What a rule can see

`v` holds only what an app could know. All pair tables are `n x n` and read
"me, then them": `s[i, j]` is how good person `j` looks for person `i`, from
0 to 1.

| On `v` | What it is |
| --- | --- |
| `v.features()` | Agreement on structured features by what the app has learned each person cares about. `v.features("global")` uses one set of weights for everyone; `v.features({"family_plans": 1})` uses yours. |
| `v.text`, `v.photo`, `v.voice` | How alike two people are in what they say, how they live and how they talk. Each is noisy, and each carries something the others do not. |
| `v.family`, `v.distance`, `v.age`, `v.same_community` | What the app knows about family expectations, how far apart two people live, their ages, and whether they share a community. |
| `v.ok` | Pairs you may introduce that break no dealbreaker the app knows about. |
| `v.open` | Every pair you may introduce: opposite sides, both still here, not introduced before. |
| `v.available`, `v.larger_side` | Who is still in the pool, and which side each person is on. |
| `v.round`, `v.rounds` | This week's number, from 0, and how many weeks there are. |
| `v.history` | What has happened so far: `introduced`, `answered`, `said_yes` per pair; `introductions`, `unanswered`, `yeses`, `declines`, `recent_declines`, `idle_rounds` per person. |
| `v.random()` | A table of random numbers, the same for every rule. |

And ways to turn a score into pairs. Each uses `v.ok` unless you pass
`among=`:

| Helper | What it does |
| --- | --- |
| `v.pair_best(s)` | Gives each person, in turn, their best-scored candidate who is still free. |
| `v.pair_mutual(s, top=10)` | Pairs people who are each in the other's top `top`, most promising first. An approximation of the article's method. |
| `v.pair_stable(s)` | A stable matching on the scores (deferred acceptance). |
| `v.rank(s)` | Each person's ranking of their candidates, if you want to build your own. |

You can also return your own list of `(a, b)` pairs. The engine checks it:
each pair must cross the two sides, nobody may appear twice, everyone must
still be in the pool, and no pair may be introduced a second time. If
something is wrong it stops and says what, and in which round.

## What a rule cannot see

Who would say yes, who would hit it off, how much anyone truly cares about
anything, who is about to give up. All of that is fixed when the world is
made, before any rule runs, and the view contains none of it. Two rules in
the same world and seed meet the same people with the same luck; the only
difference is what each rule chose to do.

## The score

How many couples your rule formed who both wanted to meet again, as a share
of the most that anyone could have formed in that world. The scorecard also
shows how far a reader of perfect profiles would have got, so you can tell
a rule that is short of information from one that is wasting it.

A score describes a rule in a made-up world whose assumptions are written
down in `wavelength describe <world>`. It is not evidence about real people.
