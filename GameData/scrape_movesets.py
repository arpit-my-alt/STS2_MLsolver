"""
Scrapes enemy movesets from slaythespire2.gg into the ENEMY_MOVESETS schema
used by sts2_state.py.

IMPORTANT: this could not be tested against the live site from the sandbox
this was written in (slaythespire2.gg isn't reachable from there). Run this
on your own machine, and expect to need to adjust the CSS/regex patterns
after inspecting real output -- use --debug on one enemy first before
running --all.

Strategy (in order of reliability):
  1. Move table (Move / Base / Asc. / Target / Notes) -- damage, hits,
     target, and mechanical constraints ("at most N in a row", "cannot be
     used on consecutive turns") are parsed straight out of this table.
  2. Prose pattern-description text ("Cycles through A -> B -> C",
     "Alternates between A and B", "Starts with A then B, then uses B
     every turn") -- fully describes deterministic/cyclic graphs when
     present, no probability guessing needed.
  3. "Always followed by X" notes -- gives a deterministic single-target
     edge straight from the move's own notes cell.
  4. Anything left unresolved (e.g. Inklet's Jab -> 50/50 Whirlwind or
     Piercing Gaze split) falls back to a uniform distribution over the
     plausible next moves, and the enemy is flagged needs_review=True in
     the output so you know to sanity-check it by hand -- possibly against
     the interactive diagram directly in a browser, since the percentage
     labels may only exist in a rendered SVG/canvas layer, not in scrapable
     text.

Output: one JSON file per enemy in ./scraped_movesets/<slug>.json, plus a
combined ./scraped_movesets/_all.json for bulk review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://slaythespire2.gg"
OUT_DIR = Path("scraped_movesets")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; STS2-moveset-research-scraper/0.1; personal ML project)"
}
REQUEST_DELAY_S = 1.5  # be polite -- this is someone else's site and bandwidth


def fetch(session: requests.Session, url: str) -> str:
    resp = session.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s


# ---------------------------------------------------------------------------
# Step 1: discover enemy slugs from the /enemies index
# ---------------------------------------------------------------------------

def scrape_enemy_list(session: requests.Session) -> list[str]:
    html = fetch(session, f"{BASE_URL}/enemies")
    # Look for every href="/enemies/<slug>" that isn't the index itself.
    slugs = sorted(set(re.findall(r'href="/enemies/([a-z0-9\-]+)"', html)))
    if not slugs:
        print(
            "WARNING: found 0 enemy links. The page may be client-rendered "
            "and this simple GET isn't seeing the hydrated content -- try "
            "--debug-raw to inspect what was actually returned, or fall "
            "back to a headless-browser fetch (e.g. Playwright) if so.",
            file=sys.stderr,
        )
    return slugs


# ---------------------------------------------------------------------------
# Step 2: per-enemy page parsing
# ---------------------------------------------------------------------------

_DMG_RE = re.compile(r"Deal\s+(\d+)(?:[×x](\d+))?\s*damage", re.IGNORECASE)
_MAX_CONSECUTIVE_RE = re.compile(r"can be used at most (\d+) times? in a row", re.IGNORECASE)
_NO_CONSECUTIVE_RE = re.compile(r"cannot be used on consecutive turns", re.IGNORECASE)
_ALWAYS_FOLLOWED_RE = re.compile(r"always followed by ([A-Za-z0-9 ]+?)\.", re.IGNORECASE)
_OPENER_RE = re.compile(r"always used at the start of combat", re.IGNORECASE)

_CYCLE_RE = re.compile(r"cycles through ([A-Za-z0-9 ,>\->]+?)(?:\.|$)", re.IGNORECASE)
_ALTERNATES_RE = re.compile(r"alternates between ([A-Za-z0-9 ]+?) and ([A-Za-z0-9 ]+?)(?:\.|$)", re.IGNORECASE)
_STARTS_THEN_RE = re.compile(
    r"starts with ([A-Za-z0-9 ]+?)\s*->\s*([A-Za-z0-9 ]+?)\.\s*then uses ([A-Za-z0-9 ]+?) every turn",
    re.IGNORECASE,
)


@dataclass
class RawMove:
    name: str
    hits: int
    base_dmg: int
    asc_dmg: int
    target: str
    notes: str
    max_consecutive: int | None = None   # None = unconstrained / unknown
    always_followed_by: str | None = None
    is_opener: bool = False


@dataclass
class ScrapedEnemy:
    slug: str
    display_name: str
    moves: list = field(default_factory=list)          # list[RawMove]
    pattern_text: str = ""                               # raw prose pattern description, if any
    needs_review: bool = False
    review_reason: str = ""


def parse_move_table(soup: BeautifulSoup) -> list[RawMove]:
    """Finds the move table by header text (Move/Base/Asc./Target/Notes)
    rather than CSS class, since class names are far more likely to change
    across site redesigns than the column headers themselves."""
    moves: list[RawMove] = []

    target_table = None
    for table in soup.find_all("table"):
        header_text = table.get_text(" ", strip=True).lower()
        if "move" in header_text and "base" in header_text and "notes" in header_text:
            target_table = table
            break

    if target_table is None:
        return moves

    for row in target_table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 5:
            continue
        name_cell, base_cell, asc_cell, target_cell, notes_cell = [c.get_text(" ", strip=True) for c in cells[:5]]

        if name_cell.lower() == "move":  # header row
            continue

        base_match = _DMG_RE.search(base_cell)
        asc_match = _DMG_RE.search(asc_cell)
        if not base_match:
            # Non-damage move (Buff/Defend/Summon/etc) -- still record it
            # with 0 damage so it exists in the graph; the simulator's
            # own effect resolver (from CARD_DB-style logic) should handle
            # the actual non-damage effect via its own text/id, not this
            # damage-focused scraper.
            hits, base_dmg = 0, 0
        else:
            base_dmg = int(base_match.group(1))
            hits = int(base_match.group(2)) if base_match.group(2) else 1

        asc_dmg = int(asc_match.group(1)) if asc_match else base_dmg

        max_consec = None
        if _NO_CONSECUTIVE_RE.search(notes_cell):
            max_consec = 1
        m = _MAX_CONSECUTIVE_RE.search(notes_cell)
        if m:
            max_consec = int(m.group(1))

        followed_by_match = _ALWAYS_FOLLOWED_RE.search(notes_cell)
        followed_by = followed_by_match.group(1).strip() if followed_by_match else None

        # Strip the trailing intent-icon label duplication frequently glued
        # onto the move name cell (e.g. "Jab Aggressive" -> "Jab") -- best
        # effort; verify against real output since this depends on how the
        # intent icon's alt/label text renders as adjacent text.
        clean_name = re.sub(r"(Aggressive|Defensive|Strategic|Empower|Malicious|Sleeping|Stunned|Cowardly|Heal)\s*$", "", name_cell).strip()

        moves.append(
            RawMove(
                name=clean_name,
                hits=hits,
                base_dmg=base_dmg,
                asc_dmg=asc_dmg,
                target=target_cell,
                notes=notes_cell,
                max_consecutive=max_consec,
                always_followed_by=followed_by,
                is_opener=bool(_OPENER_RE.search(notes_cell)),
            )
        )

    return moves


def find_pattern_text(soup: BeautifulSoup) -> str:
    """Looks for a short prose summary near a 'Behavior'/'Notes'/'Attack
    pattern' heading that describes the move cycle in plain language.
    This is the most reliable source for fully-deterministic movesets."""
    candidates = []
    for heading in soup.find_all(["h2", "h3", "strong"]):
        text = heading.get_text(strip=True).lower()
        if "behavior" in text or "pattern" in text or "notes" in text:
            sib = heading.find_next_sibling()
            if sib:
                candidates.append(sib.get_text(" ", strip=True))
    return " ".join(candidates)


def compile_transitions(moves: list[RawMove], pattern_text: str) -> tuple[dict, str, bool, str]:
    """Best-effort compile of moves + notes + pattern_text into a
    transitions dict of move_name -> ((next_name, prob), ...). Returns
    (transitions, start_move_guess, needs_review, review_reason)."""

    move_names = [m.name for m in moves]
    transitions: dict[str, list] = {}
    needs_review = False
    review_reason = ""

    # 1. Explicit "cycles through A -> B -> C" text -- fully deterministic.
    cyc = _CYCLE_RE.search(pattern_text)
    if cyc:
        chain = [s.strip() for s in re.split(r"->|,", cyc.group(1)) if s.strip()]
        for i, name in enumerate(chain):
            nxt = chain[(i + 1) % len(chain)]
            transitions[name] = [(nxt, 1.0)]
        start_guess = chain[0]
        return transitions, start_guess, False, ""

    # 2. Explicit "always followed by X" per-move notes -- deterministic edges.
    has_any_explicit = any(m.always_followed_by for m in moves)
    for m in moves:
        if m.always_followed_by:
            transitions[m.name] = [(m.always_followed_by, 1.0)]

    # 3. Anything left without an outgoing edge: fall back to uniform split
    # over all OTHER moves that aren't otherwise constrained -- this is a
    # guess, not scraped fact, and must be flagged.
    unresolved = [m.name for m in moves if m.name not in transitions]
    if unresolved:
        needs_review = True
        review_reason = (
            f"No explicit transition found for: {unresolved}. Defaulted to "
            f"uniform distribution over remaining moves -- verify against "
            f"the live diagram (percentages may only render in SVG/canvas, "
            f"not in scraped text)."
        )
        others = [n for n in move_names]
        for name in unresolved:
            targets = [n for n in others if n != name] or others
            prob = 1.0 / len(targets)
            transitions[name] = [(t, prob) for t in targets]

    opener_moves = [m.name for m in moves if m.is_opener]
    start_guess = opener_moves[0] if opener_moves else (move_names[0] if move_names else "")
    if len(opener_moves) > 1:
        needs_review = True
        review_reason += f" Multiple moves flagged as combat-openers: {opener_moves} -- likely position-dependent (e.g. front/middle/back in a pack); needs manual disambiguation."

    return transitions, start_guess, needs_review, review_reason


def scrape_enemy(session: requests.Session, slug: str) -> ScrapedEnemy:
    html = fetch(session, f"{BASE_URL}/enemies/{slug}")
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1")
    display_name = title_tag.get_text(strip=True) if title_tag else slug

    moves = parse_move_table(soup)
    pattern_text = find_pattern_text(soup)

    enemy = ScrapedEnemy(slug=slug, display_name=display_name, moves=moves, pattern_text=pattern_text)

    if not moves:
        enemy.needs_review = True
        enemy.review_reason = "No move table found -- page structure may differ from expected, or content is client-rendered and not present in the raw GET response."

    return enemy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enemy", help="Scrape a single enemy by slug (e.g. 'inklet')")
    parser.add_argument("--all", action="store_true", help="Scrape every enemy in the index")
    parser.add_argument("--debug-raw", action="store_true", help="Dump raw HTML of the first fetch to stdout and exit, for troubleshooting selectors")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    session = requests.Session()

    if args.debug_raw:
        target = args.enemy or "inklet"
        html = fetch(session, f"{BASE_URL}/enemies/{target}")
        print(html[:5000])
        return

    if args.enemy:
        slugs = [args.enemy]
    elif args.all:
        print("Discovering enemy list...")
        slugs = scrape_enemy_list(session)
        print(f"Found {len(slugs)} enemies.")
    else:
        parser.print_help()
        return

    all_results = {}
    for i, slug in enumerate(slugs):
        print(f"[{i+1}/{len(slugs)}] Scraping {slug}...")
        try:
            enemy = scrape_enemy(session, slug)
        except requests.HTTPError as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            time.sleep(REQUEST_DELAY_S)
            continue

        transitions, start_guess, needs_review, reason = compile_transitions(enemy.moves, enemy.pattern_text)
        enemy.needs_review = enemy.needs_review or needs_review
        enemy.review_reason = (enemy.review_reason + " " + reason).strip()

        result = {
            "enemy": asdict(enemy),
            "compiled": {
                "start_move": start_guess,
                "transitions": transitions,
            },
        }
        all_results[slug] = result

        with open(OUT_DIR / f"{slug}.json", "w") as f:
            json.dump(result, f, indent=2)

        flag = " [NEEDS REVIEW]" if enemy.needs_review else ""
        print(f"  -> {len(enemy.moves)} moves parsed.{flag}")

        time.sleep(REQUEST_DELAY_S)

    with open(OUT_DIR / "_all.json", "w") as f:
        json.dump(all_results, f, indent=2)

    review_count = sum(1 for r in all_results.values() if r["enemy"]["needs_review"])
    print(f"\nDone. {len(all_results)} enemies scraped, {review_count} flagged for manual review.")
    print(f"See {OUT_DIR}/_all.json for the combined output.")


if __name__ == "__main__":
    main()
