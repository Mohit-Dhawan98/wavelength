"""Load the playground in headless Chrome, wait for the engine, run it, and save screenshots.

    uv run --with playwright python web/dev_check.py [port]

Serves web/ itself, so nothing else needs to be running.
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

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
handler.log_message = lambda *a, **k: None
server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
OUT.mkdir(exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    for name, width, height in [("desktop", 1440, 900), ("phone", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height})
        errors = []
        page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append("console: " + m.text))
        page.on("requestfailed", lambda r: errors.append("failed: " + r.url))
        page.goto(f"http://127.0.0.1:{port}/index.html")
        page.wait_for_selector("#app:not([hidden])", timeout=120000)
        page.wait_for_function("document.getElementById('score-big').textContent.includes('%')", timeout=60000)
        page.wait_for_timeout(2500)
        print(name, "score:", page.inner_text("#score-big"), "|", page.inner_text("#took"))
        print(name, "week line:", page.inner_text("#stage-text")[:110])
        print(name, "knobs:", page.locator(".knob").count(), "presets:", page.locator("#presets .chip").count(), "rules:", page.locator("#rules .chip").count())
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        print(name, "sideways overflow:", overflow, "| errors:", errors)
        if name == "desktop":
            page.screenshot(path=str(OUT / "desktop_top.png"))
            page.locator("#stage-fig").scroll_into_view_if_needed(); page.wait_for_timeout(400)
            page.locator("#stage-fig").screenshot(path=str(OUT / "desktop_stage.png"))
            page.locator("#score-fig").screenshot(path=str(OUT / "desktop_score.png"))
            # a slider change reruns; a broken rule reports its line
            page.locator('.knob[data-key="yes_rate"] input').first.fill("0.6"); page.wait_for_timeout(1500)
            print("after yes_rate=0.6:", page.inner_text("#score-big"))
            page.fill("#code", "from wavelength.play import rule\n\n@rule('broken')\ndef f(v):\n    return v.nope\n")
            page.click("#run"); page.wait_for_timeout(1500)
            print("error shown:", page.inner_text("#err"))
            page.click('#rules .chip >> nth=0'); page.wait_for_timeout(1500)
            page.click("#compare"); page.wait_for_selector("#board-card:not([hidden])", timeout=120000)
            print("leaderboard rows:", page.locator("#board li").count(), "| top:", page.inner_text("#board li >> nth=0").replace("\n", " "))
            page.locator("#board-card").screenshot(path=str(OUT / "desktop_board.png"))
        else:
            page.screenshot(path=str(OUT / "phone_top.png"))
        page.close()
    browser.close()
server.shutdown()
