# A first-time visitor's run through the page

Two runs, both driven in headless Chrome with Playwright the way a person would use the page, at
1440×900 and at 390×844, with a screenshot at every step. Part 1 is the page as it was. Part 2 is
the same run on the rebuilt page.

Screenshots are written to `web/shots/`, which git ignores. Part 1's are in `web/shots/before/`.
Part 2's are rewritten every time `web/dev_check.py` runs:

```
./web/build_wheel.sh
uv run --with playwright python web/dev_check.py
```

What a still cannot show is said at the end.

## Part 1. The page as it was ("wavelength.play")

Engine ready after 2.8 to 3.9 seconds on this machine, with everything cached. A first visit
downloads Pyodide and numpy first and is slower.

| Step | Expected | What happened | Confusing or broken | Screenshot |
|---|---|---|---|---|
| Understand the page in ten seconds | One line saying what I do here, and something to press | A 64px headline, "Design a dating world. Write a matching rule. Count the couples.", then a four-line paragraph. Below it, a card of ten sliders and a card with seven grey chips and a code box. | The first verb is "design a dating world" and the first control is ten sliders, so the page says the job is world-building. Nothing says what a "rule" is in plain words. The page ran itself on load, and at 1440×900 no result is on screen: the only sign is "ran in 425 ms" in 12px grey type. | `before/desktop_01_loading.png`, `before/desktop_02_first_view.png`, `before/phone_02_first_view.png` |
| Find the result | A score near the Run button | The graph starts at y=903 and the score panel at y=1561 on a 900px screen. On the phone the graph starts at y=1912 and the score at y=2585. | **Broken as a layout.** The buttons and the score can never be on one screen. On the phone it is three screens down with no pointer to it. | `before/desktop_03_full.png`, `before/phone_03_full.png` |
| Pick a different world | The score changes where I can see it | Clicking "dystopia" reran at once and the description line changed. The score moved to 1.6%, off-screen. | Works, and I could not tell. The ten sliders moved, which is the only visible response. Labels such as "endogamy share" and "majority share" are jargon with bare decimals. | `before/desktop_04_world_dystopia.png` |
| Run a different rule | A list of rules with what each does | Seven chips with names only ("spread attention", "best for me"). Clicking one swaps the code and starts a 17 second animation below the fold. | The one-line description of each rule exists only as a hover tooltip. The chip is the only "run" control for a demo rule, and it silently replaces whatever was in the code box. | `before/desktop_05_rule_clicked.png` |
| Edit a rule and run it | An Edit button, some help on what I can write, a clear Run | I changed `v.text` to `v.voice` in the box. Nothing happened until I pressed "Run the 8 weeks". | Nothing says what `v` is or what else it offers. After the edit every chip goes unpressed and the page shows no "edited" state. The box is 190px tall, so the two lines that are the rule sit below its own fold behind a docstring and an import. No line numbers. | `before/desktop_06_edited_not_run.png`, `before/desktop_07_edited_run.png` |
| Find out how well it did, and why | One number and a reason | "1.6% of the best possible" in 12px mono in the corner of the second dark panel. Four pairs of bars, each on its own scale (86, 79, 75, 63 couples all drawn the same width). Eight tiles in mixed units. | The answer is the smallest thing in the panel. A grey bar under each orange one reads as a second rule that is winning. It is the "perfect reader" benchmark. "People who hit it off: 2" sits beside counts of introductions. | `before/desktop_08_score.png` |
| Find out what one person went through | Click a dot | 400 dots and 793 lines. Clicking a dot does nothing. No hover, no tooltip, no cursor change. | **Does not work at all.** There is no way to ask about anyone. The dots sit at meaningless positions, the two sides are mixed, and the picture looks the same for a good rule and a bad one. Orange, the page's accent, means "a no". | `before/desktop_09_stage_end.png`, `before/desktop_10_person_clicked.png` |
| Step through the weeks | Earlier and later weeks | "Next" on week 8 wraps to week 1. Lines for that week brighten in the tangle. | The control works and changes nothing I can read. The weekly numbers appear only in a line that is typed out and erased every two seconds. | `before/desktop_11_week_next.png` |
| Break a rule | The line and the message | "Your rule, line 5: AttributeError: 'View' object has no attribute 'nope'". | Clear message. No line numbers to find line 5 with. The graph and score underneath keep showing the previous run as if it were current. | `before/desktop_12_broken.png` |
| Compare rules | A table | "Run every demo rule" changes its label for about a second and changes back. | **Looks broken.** The leaderboard is appended 1,400px below the button with no scroll and no pointer. Once found: bars are scaled to the leader, so a rule that found 4.8% of the couples fills its whole track. One seed per rule, ranked as if the order meant something. | `before/desktop_13_compare_running.png`, `before/desktop_14_board.png` |
| Share a link | A short link, clean address bar | Button says "Link copied". The link is 649 characters. It round-trips, and the address bar is cleaned after it is read. | Works. Long, because it carries the whole rule source as base64 even for a demo rule. | `before/desktop_15_share.png` |
| Open "Every setting" | The rest of the settings | Seven groups, 58 more sliders, inside a column that already scrolls inside the page. | Two nested scrollbars. The explanation of the slider under the pointer is written at the bottom of the column, outside the visible part. | `before/desktop_16_every_setting.png` |
| The same on a phone | The same tasks | No sideways scrolling of the page. | Both figures are cut off at the right: the graph loses about a third of its people and the score chart loses every number, because each has a 560px minimum width in a 358px panel. Order on the phone is headline, ten sliders, rule, Run, graph, score. | `before/phone_09_stage_end.png`, `before/phone_08_score.png` |

No page errors and no failed requests in either run.

### The short version

1. The result is not where the controls are. This is the fault everything else hides behind.
2. The graph cannot be asked anything and its picture means nothing.
3. A rule is a chip with no description, and choosing one overwrites your edits.
4. "Run every demo rule" appears to do nothing.
5. The score chart puts the answer in the corner and invites misreading the benchmark as a rival rule.
6. On a phone both figures are cropped.

## Part 2. The rebuilt page ("Dating Environment")

The same tasks, checked by `web/dev_check.py` (62 checks at 1440×900, 61 at 390×844 and one
under reduced motion, all passing) and by looking at every screenshot it writes.

| Step | What happens now | Screenshot |
|---|---|---|
| Loading | A black panel in the portfolio's dot-matrix face says what is loading, ticks off Python, numpy, the engine and the worlds, and says what is coming next. People pair up in orange as each step lands. | `desktop_01_loading.png`, `phone_01_loading.png` |
| Understand the page in ten seconds | "Design a matching rule. Choose a world. Count the couples." Five worlds across the top, strategies on the left with a sentence each, and on the right a result that is already there: "Of the 137 couples who could have hit it off, this rule found 30." At 1440×900 nothing scrolls. | `desktop_02_first_view.png` |
| Pick a different world | Five named worlds with one line each. Choosing one reruns the current strategy; the sentence, the bar and the graph change in place beside it. The 59 settings are behind "Tune this world". | `desktop_03_world_running.png`, `desktop_04_world_dystopia.png`, `desktop_19_tune_world.png` |
| Run a different rule | Each strategy is a row: name, what it does in plain words, its score in this world, Run and Edit. While it runs the row says so, a bar moves across the top of the result and the old result dims. When it lands the result's border flashes once. | `desktop_05_run_in_progress.png` |
| Edit a rule and run it | Edit opens a copy named "my most similar" with line numbers, a guide to everything a rule can see and return (press a name to insert it), and Save and run. The original cannot be overwritten. The new strategy is listed under "Yours" and survives a reload. | `desktop_14_editor.png`, `desktop_15_editor_after_run.png`, `desktop_17_yours.png` |
| Break a rule | "Line 5. AttributeError: 'View' object has no attribute 'nope'" under the editor, line 5 marked in the gutter, and the old result dimmed under a note saying it is the last run that worked. | `desktop_16_editor_error.png` |
| How well did it do, and why | One sentence, then one bar: the whole track is every possible couple, the orange part is what the rule found, a dashed tick is the perfect reader of profiles ("a reference, not a strategy"), and the hatched part past it is what no rule can reach from profiles. The funnel tab shows six steps in one unit on one scale with each loss named, the eight weeks as small multiples, and the rule's couples against what was possible. | `desktop_02_first_view.png`, `desktop_08_funnel.png`, `desktop_09_funnel_couples.png` |
| What one person went through | Click anyone, or type a number. Everyone else fades, the view moves to them, and a card tells their story week by week, how it turned out, and what the answer key says was possible for them. | `desktop_11_person_story.png` |
| The weeks | A slider and a play button. Week 3 shows what had happened by week 3; play draws each week's introductions. Unanswered introductions are hidden until asked for. | `desktop_12_week_3.png`, `desktop_13_unanswered_shown.png` |
| Does the graph say anything | Couples sit in the middle, people nobody answered on the rim. Random introductions draws a small orange centre and the article's rule, approximated, a large one. | `desktop_06_worst_rule_graph.png`, `desktop_07_best_rule_graph.png` |
| Compare rules | "Compare all" opens the comparison in the result panel. Every strategy is a dot on one axis from no couples to all the possible couples, with a line from its worst seed to its best over five seeds, and the perfect reader and random introductions as reference lines. | `desktop_10_all_strategies.png` |
| Share a link | A link to a demo strategy is `#w=today&s=stable_matching`. A link to your own rule carries it compressed (350 characters for a five-line rule). The address bar changes only when someone opens a link, and is cleaned once it is read. | `desktop_18_share_copied.png` |
| Dark | One switch. The black band stays black. | `desktop_20_dark.png`, `desktop_21_dark_result.png` |
| The same on a phone | Worlds, then strategies, then the result. A pinned line at the bottom gives the score ("30 of 137 couples, 22%. See the result") and jumps to it. The band keeps the sound and theme switches; the article and source links move to the footer. Tapping a person brings the start of their story up under the graph. Nothing is cropped and nothing scrolls sideways. A finger on the graph scrolls the page; two fingers move the graph. | `phone_02_first_view.png`, `phone_03_first_view_full.png`, `phone_04_after_jump.png`, `phone_10_funnel.png`, `phone_13_person_story.png` |

### What still reads badly

- The settings behind "Tune this world" keep the engine's names ("majority share", "endogamy
  share"). Each explains itself when touched, but the names are still jargon.
- At 1024×768 the two columns hold, but only the top half of the graph is above the fold.
- On a phone the result is still below seven strategies. The pinned line is the bridge.
- The code editor is a plain text box with line numbers. No syntax colour.
- In a world of more than 800 people only the first 400 on each side are drawn, and the other
  seeds are not run for worlds over 600 people.

### What a still cannot show

Motion and sound were built and not judged: the reveal when the engine is ready, the spring on
pressed buttons, the theme change opening as a circle, the introductions drawing in when the weeks
play, the move to a focused person, and the synthesized sounds. Headless Chrome confirms they do
not break the page and that reduced motion still reaches every end state
(`reduced_motion_first_view.png`). Whether they feel right needs a person and a real screen.
