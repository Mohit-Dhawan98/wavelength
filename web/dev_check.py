"""Use the page the way a first-time visitor would, in headless Chrome, and check what must hold.

    ./web/build_wheel.sh
    uv run --with playwright python web/dev_check.py [port]

Serves web/ itself. Runs the visit at 1440x900 and at 390x844, saves a screenshot of every step
into web/shots/, prints one line per check, and exits non-zero if any check fails. The screenshots
are for looking at: a passing run says the page works, not that it looks right.
"""
import functools
import http.server
import pathlib
import sys
import threading

from playwright.sync_api import sync_playwright

WEB = pathlib.Path(__file__).parent
OUT = WEB / "shots"
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
BASE = f"http://127.0.0.1:{port}/index.html"

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
handler.log_message = lambda *a, **k: None
server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
OUT.mkdir(exist_ok=True)
for old in OUT.glob("*.png"):
    old.unlink()

failures = []
BROKEN = "from wavelength.play import rule\n\n@rule('broken')\ndef f(v):\n    return v.nope\n"


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(name)


def visit(browser, name, width, height):
    phone = width < 700
    context = browser.new_context(viewport={"width": width, "height": height}, has_touch=phone, is_mobile=phone)
    context.grant_permissions(["clipboard-read", "clipboard-write"])
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append("console: " + m.text))
    page.on("requestfailed", lambda r: errors.append("failed: " + r.url))
    step = [0]

    def shot(label, full=False):
        step[0] += 1
        page.screenshot(path=str(OUT / f"{name}_{step[0]:02d}_{label}.png"), full_page=full)

    def overflow():
        return page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")

    def finished():
        page.wait_for_function("!document.body.classList.contains('running') && window.__env.state.running == null", timeout=60000)
        page.wait_for_timeout(700)

    def to_result():
        if phone:
            page.locator("#result").scroll_into_view_if_needed()
            page.wait_for_timeout(350)

    def to_strategies():
        if phone:
            page.locator("#strategies").scroll_into_view_if_needed()
            page.wait_for_timeout(250)

    def in_view(selector):
        box = page.locator(selector).bounding_box()
        return box is not None and box["y"] >= 0 and box["y"] + box["height"] <= height + 1 and box["height"] > 40

    print(f"\n{name} ({width}x{height})")
    # 1. what is this page: the first ten seconds
    page.goto(BASE)
    page.wait_for_timeout(900)
    shot("loading")
    check("the loading state says what is coming", "worlds" in page.inner_text("#boot"))
    page.wait_for_selector("#result-body:not([hidden])", timeout=120000)
    page.wait_for_function("document.querySelector('#force svg')", timeout=20000)
    finished()
    shot("first_view")
    check("the title and the headline carry the name", page.title() == "Dating Environment" and "Design a matching rule" in page.inner_text("h1"))
    check("a result is shown without asking", "couples who could have hit it off" in page.inner_text("#result-h"), page.inner_text("#result-h"))
    if not phone:
        check("headline score visible without scrolling", in_view("#result-h"))
        check("graph visible without scrolling", in_view("#fwrap"), str(page.locator("#fwrap").bounding_box()))
        check("strategies visible beside it", in_view(".strat >> nth=0"))
        check("the page does not scroll at this size", page.evaluate("document.documentElement.scrollHeight <= innerHeight + 1"))
    else:
        shot("first_view_full", full=True)
        check("a pinned line gives the score and jumps to the result", page.locator("#jump").is_visible() and "couples" in page.inner_text("#jump"), page.inner_text("#jump"))
        page.click("#jump")
        page.wait_for_timeout(900)
        check("the jump lands on the result", page.locator("#result-h").bounding_box()["y"] < height)
        shot("after_jump")
        page.evaluate("scrollTo(0, 0)")
        page.wait_for_timeout(300)
    check("seven strategies, each explained, each with Run and Edit",
          page.locator("#strat-list .strat").count() == 7 and page.locator("#strat-list .strat-what").count() == 7
          and page.locator("#strat-list [data-act=run]").count() == 7 and page.locator("#strat-list [data-act=edit]").count() == 7)
    check("the article's rule says it is an approximation", "not the production implementation" in page.inner_text('.strat[data-id="article"]'))
    check("no strategies of their own yet is a designed state", page.locator("#mine-empty").is_visible())
    check("no sideways overflow", overflow() == 0, str(overflow()))

    # 2. pick a different world
    before = page.inner_text("#result-h")
    page.click('.world[data-world="dystopia"]')
    page.wait_for_timeout(120)
    shot("world_running")
    finished()
    shot("world_dystopia")
    check("choosing a world reruns and says which world", "Dystopia" in page.inner_text("#verdict-what") and page.inner_text("#result-h") != before, page.inner_text("#verdict-what"))
    check("the address bar is untouched", page.url == BASE, page.url)
    page.click('.world[data-world="today"]')
    finished()

    # 3. run every strategy; 4. open every strategy's code
    names = page.eval_on_selector_all("#strat-list .strat", "els => els.map(e => [e.dataset.id, e.querySelector('h3').textContent])")
    ran, opened = [], []
    for sid, label in names:
        to_strategies()
        row = page.locator(f'.strat[data-id="{sid}"]')
        row.locator("[data-act=run]").click()
        if sid == "random_introductions":
            page.wait_for_timeout(60)
            check("a run in progress is announced", page.evaluate("document.body.classList.contains('running')") and "Running" in row.inner_text(), row.locator(".strat-foot").inner_text())
            shot("run_in_progress")
        finished()
        ran.append(label in page.inner_text("#verdict-what") and "shown" in row.get_attribute("class") and "couples" in row.locator(".strat-score").inner_text())
    check("Run works on every strategy and the row shows its score", all(ran), str(ran))
    to_result()
    shot("worst_rule_graph")
    worst = page.locator('.gn[data-state="matched"]').count()
    check("a finished run is announced", "Finished" in page.inner_text("#live"), page.inner_text("#live"))
    for sid, label in names:
        to_strategies()
        page.locator(f'.strat[data-id="{sid}"] [data-act=edit]').click()
        page.wait_for_timeout(250)
        opened.append("@rule(" in page.input_value("#code") and page.locator("#gutter span").count() > 5 and page.locator("#guide-list .sig").count() > 15)
        page.click("#ed-back")
        page.wait_for_timeout(150)
    check("Edit opens every strategy's code with line numbers and the guide", all(opened), str(opened))
    check("looking at code created nothing", page.locator("#mine-list .strat").count() == 0)
    to_strategies()
    page.locator('.strat[data-id="article"] [data-act=run]').click()
    finished()
    to_result()
    shot("best_rule_graph")
    best = page.locator('.gn[data-state="matched"]').count()
    check("two rules draw visibly different graphs", best >= worst + 10, f"people in couples: {best} against {worst}")

    # 5. how well did it do, and why
    check("the score is one sentence in words", page.inner_text("#result-h").startswith("Of the ") and "this rule found" in page.inner_text("#result-h"))
    check("the perfect reader is marked as a reference, not a strategy", "a reference, not a strategy" in page.inner_text("#scorebar"))
    page.click("#t-funnel")
    page.wait_for_timeout(900)
    shot("funnel")
    counts = page.eval_on_selector_all(".ifunnel .irow .icell:not(.zoom) b", "els => els.map(e => +e.textContent.replace(/,/g, ''))")
    check("the funnel has six steps that only shrink", len(counts) == 6 and counts == sorted(counts, reverse=True), str(counts))
    check("each loss is labelled with its reason", "never answered by both" in page.inner_text(".ifunnel") and "went quiet" in page.inner_text(".ifunnel"))
    check("only the last bar is orange", page.locator(".ifunnel .icell:not(.zoom) .fill.rule").count() == 1)
    check("the weeks are small multiples", page.locator(".wk-col:not(.labels)").count() == 8)
    page.locator(".sub >> nth=1").scroll_into_view_if_needed()
    page.wait_for_timeout(300)
    shot("funnel_couples")
    check("no sideways overflow with the funnel open", overflow() == 0, str(overflow()))

    # 6. compare rules
    to_strategies()
    page.click("#run-all")
    page.wait_for_timeout(400)
    check("Compare all lands on the comparison", page.locator("#p-all").is_visible() and (not phone or page.locator("#p-all").bounding_box()["y"] < height))
    page.wait_for_function("window.__env.seedsLeft() === 0", timeout=240000)
    page.wait_for_timeout(500)
    shot("all_strategies")
    rows = page.locator("#p-all tbody tr:not(.unrun)").count()
    check("every strategy is on one axis with a range over seeds", rows == 7 and page.locator("#p-all .plot .range").count() == 7 and "5 seeds" in page.inner_text("#p-all"), f"{rows} rows")
    check("the perfect reader and random are reference lines", page.locator("#p-all .ref.bench").count() == 7 and page.locator("#p-all .ref.floor").count() == 7)
    dots = page.eval_on_selector_all("#p-all .plot .dot", "els => els.map(e => parseFloat(e.style.left))")
    check("nothing is scaled to the leader", max(dots) < 60, f"best mean sits at {max(dots):.0f}% of the axis")
    check("no sideways overflow with the comparison open", overflow() == 0, str(overflow()))

    # 7. what one person went through
    page.click("#t-graph")
    page.wait_for_timeout(500)
    check("unanswered introductions are hidden until asked for", page.locator("#force .gl.k-none:not(.off)").count() == 0 and page.locator("#force .gl.k-none").count() > 100,
          f'{page.locator("#force .gl.k-none:not(.off)").count()} of {page.locator("#force .gl.k-none").count()} shown')
    page.locator('.gn[data-state="looking"] >> nth=3').click(force=True)
    page.wait_for_timeout(900)
    shot("person_story")
    story = page.inner_text("#card")
    check("clicking a person tells their story week by week", "Week" in story and "Introduced to #" in story and "How it turned out" in story and "What was possible" in story, story[:120].replace("\n", " "))
    check("everything else fades", page.evaluate("document.querySelector('#force svg').classList.contains('focus')") and page.locator(".gn.on").count() < 30)
    page.fill("#g-find", "41")
    page.press("#g-find", "Enter")
    page.wait_for_timeout(700)
    check("a person can be found by number, from the keyboard", "Person #41" in page.inner_text("#card"))
    page.click("#g-reset")
    page.wait_for_timeout(600)
    check("Reset view clears the focus", "How to read it" in page.inner_text("#card h3") and page.locator(".gn.on").count() == 0)
    check("How to read it defines the marks", all(w in page.inner_text("#card") for w in ("circle", "square", "Bigger", "Orange", "Hollow", "never answered", "Where people sit")))

    # 8. the week control
    page.locator("#g-week").focus()
    for _ in range(5):
        page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(500)
    shot("week_3")
    shown3 = page.locator("#force .gl:not(.off)").count()
    check("the week slider goes back to week 3", page.inner_text("#g-week-label") == "Week 3 of 8" and page.inner_text("#weekline").startswith("Week 3"), page.inner_text("#weekline"))
    page.click("#g-play")
    page.wait_for_timeout(1900)
    check("play moves the weeks on and draws more introductions", page.inner_text("#g-week-label") != "Week 3 of 8" and page.locator("#force .gl:not(.off)").count() > shown3, page.inner_text("#g-week-label"))
    page.click("#g-play")
    page.locator("#g-week").focus()
    page.keyboard.press("End")
    page.click("#g-unanswered")
    page.wait_for_timeout(500)
    shot("unanswered_shown")
    check("Unanswered shows the introductions nobody answered", page.locator("#force .gl.k-none:not(.off)").count() > 100)
    page.click("#g-unanswered")

    # 9. edit a rule and run it; make it theirs; break it
    to_strategies()
    page.locator('.strat[data-id="most_similar"] [data-act=edit]').click()
    page.wait_for_timeout(700)
    shot("editor")
    check("the editor opens on a uniquely named copy", page.input_value("#ed-name") == "my most similar" and "A copy of most similar" in page.inner_text("#ed-origin"), page.input_value("#ed-name"))
    page.fill("#code", page.input_value("#code").replace("v.text", "0.5 * v.text + 0.5 * v.voice"))
    page.click("#ed-run")
    finished()
    shot("editor_after_run")
    check("Save and run runs the edited rule and says so", "my most similar" in page.inner_text("#verdict-what") and "Finished" in page.inner_text("#ed-status"), page.inner_text("#ed-status"))
    check("the original is untouched", page.evaluate("window.__env.state.builtins.find(b => b.id === 'most_similar').source.includes('pair_best(v.text')"))
    good = page.input_value("#code")
    page.fill("#code", BROKEN)
    page.click("#ed-run")
    finished()
    shot("editor_error")
    check("a broken rule reports its line beside the editor", page.locator("#ed-err").is_visible() and "Line 5" in page.inner_text("#ed-err") and "nope" in page.inner_text("#ed-err"), page.inner_text("#ed-err"))
    check("the broken line is marked in the gutter", page.inner_text("#gutter .bad") == "5")
    check("the old result is marked as stale, not passed off as new", page.locator("#stale").is_visible())
    page.fill("#code", good)
    page.click("#ed-run")
    finished()
    check("fixing it clears the error", not page.locator("#ed-err").is_visible() and not page.locator("#stale").is_visible())
    page.click("#ed-back")
    page.wait_for_timeout(300)
    to_strategies()
    page.locator("#mine-list").scroll_into_view_if_needed()
    page.wait_for_timeout(300)
    shot("yours")
    check("the new strategy is listed as theirs", page.locator("#mine-list .strat").count() == 1 and "yours" in page.inner_text("#mine-list .strat"))
    page.click("#new-strategy")
    page.wait_for_timeout(300)
    check("New strategy starts from a short example", "my rule" in page.input_value("#ed-name") and "pair_mutual" in page.input_value("#code"))
    page.click("#ed-back")

    # 10. a share link round-trips, and the address bar stays clean
    to_result()
    page.click("#share")
    page.wait_for_timeout(400)
    shot("share_copied")
    link = page.evaluate("navigator.clipboard.readText()")
    check("the copied link carries the run", link.startswith(BASE + "#w=today&c="), f"{len(link)} characters")
    page.locator('.strat[data-id="stable_matching"] [data-act=run]').click()
    finished()
    page.click("#share")
    page.wait_for_timeout(300)
    short = page.evaluate("navigator.clipboard.readText()")
    check("a link to a demo strategy is short", short == BASE + "#w=today&s=stable_matching", short)
    check("sharing leaves the address bar alone", page.url == BASE, page.url)

    # 11. the strategy survives a reload
    page.reload()
    page.wait_for_selector("#result-body:not([hidden])", timeout=120000)
    finished()
    check("their strategy survives a reload", page.locator("#mine-list .strat").count() == 1 and "my most similar" in page.inner_text("#mine-list"))

    other = context.new_page()
    other.on("pageerror", lambda e: errors.append("pageerror (shared link): " + str(e)))
    other.goto(link.replace("w=today", "w=casual"))
    other.wait_for_selector("#result-body:not([hidden])", timeout=120000)
    other.wait_for_function("window.__env.state.running == null")
    other.wait_for_timeout(600)
    check("opening a share link runs that rule in that world", "my most similar" in other.inner_text("#verdict-what") and "Casual" in other.inner_text("#verdict-what"), other.inner_text("#verdict-what"))
    check("the link is read once and the address is cleaned", other.url == BASE, other.url)
    other.close()

    # 12. tune the world
    page.evaluate("scrollTo(0, 0)")
    page.click("#adjust")
    page.wait_for_timeout(600)
    page.evaluate("""document.querySelector('.knob[data-key="yes_rate"]').closest('details').open = true""")
    slider = page.locator('.knob[data-key="yes_rate"] input')
    slider.focus()
    for _ in range(10):
        page.keyboard.press("ArrowRight")
    page.wait_for_timeout(200)
    shot("tune_world")
    check("the 59 settings are behind Tune this world", page.locator(".knob").count() == 58 and page.locator("#sheet .row3 input").count() == 3, str(page.locator(".knob").count()))
    check("a share is shown as a percentage, with its meaning under it", page.inner_text('.knob[data-key="yes_rate"] output').endswith("%") and page.locator('.knob[data-key="yes_rate"] .knob-why').is_visible(), page.inner_text('.knob[data-key="yes_rate"] output'))
    page.click("#sheet-apply")
    finished()
    check("applying it reruns in the adjusted world", "adjusted" in page.inner_text("#verdict-what") and "1 setting changed" in page.inner_text("#world-size"), page.inner_text("#world-size"))

    # 13. the dark theme
    page.evaluate("scrollTo(0, 0)")
    page.click("[data-theme-toggle]")
    page.wait_for_timeout(1300)
    shot("dark")
    check("one switch turns the page dark", page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(17, 17, 16)")
    to_result()
    shot("dark_result")
    page.evaluate("localStorage.clear()")

    check("no sideways overflow at the end", overflow() == 0, str(overflow()))
    check("no page errors and no failed requests", not errors, "; ".join(errors)[:300])
    context.close()


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    visit(browser, "desktop", 1440, 900)
    visit(browser, "phone", 390, 844)
    # reduced motion: the page must still arrive at its end states
    context = browser.new_context(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
    page = context.new_page()
    page.goto(BASE)
    page.wait_for_selector("#result-body:not([hidden])", timeout=120000)
    page.wait_for_function("document.querySelector('#force svg')", timeout=20000)
    page.wait_for_timeout(600)
    page.screenshot(path=str(OUT / "reduced_motion_first_view.png"))
    print("\nreduced motion")
    check("with reduced motion the result and graph still arrive", page.locator(".gn").count() > 100 and "this rule found" in page.inner_text("#result-h"))
    context.close()
    browser.close()
server.shutdown()

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
