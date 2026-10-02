# A prover trap on a wasm-bindgen-rayon pool worker must reach the page as the
# crash message. Exits non-zero on the first failure.
#
#   node server.mjs &            # restart after every wasm/frontend rebuild (it caches)
#   uv run --with playwright python bench/test_pool_worker_trap.py [--url http://localhost:8080] [--shots DIR]
#
# The wasm memory maximum is lowered by rewriting /pkg/jolt_wasm_prover.js in
# flight, so the SHA-256 proof runs out of memory inside a parallel phase: the
# failing allocation is on a pool worker while worker.js sits blocked in the
# prove call. The page must show the crash message, log the pool worker's trap
# and disable the controls. A second page without the cap proves normally.
# The page is pinned to 8 rayon threads; at that pool size 1300-1700 pages trap
# on a pool worker, <= 1200 on thread zero, and 2000 proves.

import argparse
import contextlib
import os
import sys
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

PANEL = "[role=tabpanel][data-state=active]"
CRASH_TEXT = "The prover crashed"
POOL_TRAP = "wbg_rayon_start_worker"


def check(cond, msg):
    if not cond:
        sys.stderr.write(f"FAIL: {msg}\n")
        sys.exit(1)
    sys.stderr.write(f"ok: {msg}\n")


def open_page(context, url, max_pages):
    page = context.new_page()
    page.on("console", lambda m: sys.stderr.write(f"[webkit] {m.text}\n") if m.type == "error" else None)
    if max_pages:

        def cap_memory(route):
            body = route.fetch().text()
            if "maximum:65536" not in body:
                sys.exit("FAIL: the wasm glue no longer declares maximum:65536")
            route.fulfill(
                status=200,
                body=body.replace("maximum:65536", f"maximum:{max_pages}"),
                headers={"content-type": "text/javascript", "cross-origin-resource-policy": "same-origin"},
            )

        page.route("**/pkg/jolt_wasm_prover.js", cap_memory)
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#status')?.classList.contains('ready')")
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("BENCH_URL", "http://localhost:8080"))
    ap.add_argument("--shots", default=None, help="directory for the crashed-page screenshot")
    ap.add_argument("--max-pages", type=int, default=1500, help="capped wasm memory maximum in 64 KiB pages")
    ap.add_argument("--timeout", type=int, default=60, help="seconds to wait for the crash message")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.webkit.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.set_default_timeout(600_000)
        context.add_init_script("localStorage.setItem('jolt-prover-mode', 'cpu')")
        # The cap is calibrated for the UI's 8-thread pool (min(hardwareConcurrency, 8)).
        context.add_init_script("Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 })")

        page = open_page(context, args.url, args.max_pages)
        page.locator(f"{PANEL} .prove-btn").click()
        page.wait_for_function("() => document.querySelector('#status')?.classList.contains('proving')")
        with contextlib.suppress(PlaywrightTimeoutError):
            page.wait_for_function(
                f"() => document.body.innerText.includes('{CRASH_TEXT}')", timeout=args.timeout * 1000
            )
        if args.shots:
            Path(args.shots).mkdir(parents=True, exist_ok=True)
            page.screenshot(path=Path(args.shots) / "pool-worker-trap.png", full_page=True)
        status = page.locator("#status").inner_text().strip()
        check(
            CRASH_TEXT in page.locator("body").inner_text(), f"crash message shown (status badge: {status})"
        )
        check(
            POOL_TRAP in page.locator(f"{PANEL} .output").inner_text(), "output logs the pool worker's trap"
        )
        check(page.locator(f"{PANEL} .prove-btn").is_disabled(), "prove button disabled")
        page.close()

        page = open_page(context, args.url, None)
        page.locator(f"{PANEL} .prove-btn").click()
        page.wait_for_function(
            f"() => /Proof SHA-256: [0-9a-f]{{64}}/.test(document.querySelector('{PANEL} .output')?.innerText)"
        )
        sys.stderr.write("ok: uncapped SHA-256 proof generated\n")
        page.close()
        browser.close()
    print("test_pool_worker_trap: all checks passed")


if __name__ == "__main__":
    main()
