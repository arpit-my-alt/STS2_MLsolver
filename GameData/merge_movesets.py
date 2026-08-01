"""
Compiles scraped_movesets/_all.json (raw scraper output) + overrides.py
(your manual corrections) into TWO generated files, matching the
moves_db.py / sts2_state.py split:

    enemy_moves_generated.json     -> loaded by moves_db.py (move definitions)
    enemy_movesets_generated.json  -> loaded by sts2_state.py (transitions only)

Usage:
    python merge_movesets.py                      # uses default paths
    python merge_movesets.py --report              # just print review status, don't write

Workflow:
    1. Run scrape_movesets.py --all
    2. Open scraped_movesets/_all.json, note which enemies have needs_review=True
    3. For each one, add an entry to overrides.py with the corrected fields
    4. Run this script -- it patches overrides on top of the scrape and
       writes both generated files.
    5. Re-running the scraper later won't lose your overrides -- they're
       stored separately and reapplied each merge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moves_db import to_id, enemy_move_id, TargetType

try:
    from overrides import OVERRIDES
except ImportError:
    OVERRIDES = {}
    print("No overrides.py found (or no OVERRIDES dict in it) -- proceeding with zero manual corrections.")


_TARGET_MAP = {
    "player": TargetType.ANY_PLAYER.value,
    "self": TargetType.SELF.value,
    "—": TargetType.NONE.value,
    "-": TargetType.NONE.value,
    "": TargetType.NONE.value,
}


def normalize_target(raw_target: str) -> str:
    return _TARGET_MAP.get((raw_target or "").strip().lower(), TargetType.ANY_PLAYER.value)


def compile_enemy(slug: str, scraped: dict) -> tuple[list[dict], dict]:
    """Returns (moves_for_moves_db, moveset_for_sts2_state)."""
    enemy = scraped["enemy"]
    compiled = scraped["compiled"]
    override = OVERRIDES.get(slug, {})
    enemy_base_id = to_id(slug)

    # --- moves (-> enemy_moves_generated.json, feeds moves_db.py) --------
    moves_out = []
    for mv in enemy["moves"]:
        mid = enemy_move_id(enemy_base_id, mv["name"])
        target = normalize_target(mv["target"])
        max_consecutive = mv["max_consecutive"] if mv["max_consecutive"] is not None else None

        moves_out.append({
            "id": mid, "is_upgraded": False, "hits": mv["hits"], "dmg": mv["base_dmg"],
            "target": target, "card_type": None, "keywords": [],
            "effects": [], "max_consecutive": max_consecutive, "ascension_threshold": None,
        })
        if mv["asc_dmg"] != mv["base_dmg"]:
            moves_out.append({
                "id": mid, "is_upgraded": True, "hits": mv["hits"], "dmg": mv["asc_dmg"],
                "target": target, "card_type": None, "keywords": [],
                "effects": [], "max_consecutive": max_consecutive, "ascension_threshold": 1,
                # ascension_threshold=1 is a guess (the site's "Asc." column
                # doesn't say WHICH ascension) -- verify before trusting at
                # a specific ascension level.
            })

    # Manual effect overrides, since the scraper can't reliably infer
    # "heals for 15 x player count" or "gains 3 Steam Eruption" from prose.
    move_effects_override = override.get("move_effects", {})
    for m in moves_out:
        raw_name_matches = [n for n in move_effects_override if enemy_move_id(enemy_base_id, n) == m["id"]]
        if raw_name_matches:
            m["effects"] = move_effects_override[raw_name_matches[0]]

    # --- moveset (-> enemy_movesets_generated.json, feeds sts2_state.py) --
    transitions = {}
    for src_name, edges in compiled["transitions"].items():
        src_id = enemy_move_id(enemy_base_id, src_name)
        transitions[src_id] = [[enemy_move_id(enemy_base_id, t), p] for t, p in edges]

    start_move = enemy_move_id(enemy_base_id, compiled["start_move"]) if compiled["start_move"] else ""

    needs_review = enemy["needs_review"]
    review_reason = enemy["review_reason"]
    start_move_by_position = {}

    if "transitions" in override:
        for src_name, edges in override["transitions"].items():
            transitions[enemy_move_id(enemy_base_id, src_name)] = [
                [enemy_move_id(enemy_base_id, t), p] for t, p in edges
            ]

    if "start_move_by_position" in override:
        start_move_by_position = {
            position: enemy_move_id(enemy_base_id, name)
            for position, name in override["start_move_by_position"].items()
        }

    position_key_fn = override.get("position_key_fn", "front_middle_back")
    spawn_placeholder_move = enemy_move_id(enemy_base_id, override["spawn_placeholder_move"]) if "spawn_placeholder_move" in override else None
    on_death_move = enemy_move_id(enemy_base_id, override["on_death_move"]) if "on_death_move" in override else None
    terminal_moves = [enemy_move_id(enemy_base_id, n) for n in override.get("terminal_moves", [])]
    passive_effects = override.get("passive_effects", [])

    if "start_move" in override:
        start_move = enemy_move_id(enemy_base_id, override["start_move"])

    if override.get("resolved"):
        needs_review = False
        review_reason = ""

    # Terminal moves never have a real "next move" -- drop whatever the
    # scraper's uniform-fallback guess put there.
    for tid in terminal_moves:
        transitions.pop(tid, None)

    moveset_out = {
        "enemy_base_id": enemy_base_id,
        "start_move": start_move,
        "transitions": transitions,
        "start_move_by_position": start_move_by_position,
        "position_key_fn": position_key_fn,
        "spawn_placeholder_move": spawn_placeholder_move,
        "on_death_move": on_death_move,
        "terminal_moves": terminal_moves,
        "passive_effects": passive_effects,
        "needs_review": needs_review,
        "review_reason": review_reason,
    }

    return moves_out, moveset_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scraped", default="scraped_movesets/_all.json")
    parser.add_argument("--moves-out", default="enemy_moves_generated.json")
    parser.add_argument("--movesets-out", default="enemy_movesets_generated.json")
    parser.add_argument("--report", action="store_true", help="Print review status only, don't write output")
    args = parser.parse_args()

    scraped_path = Path(args.scraped)
    if not scraped_path.exists():
        print(f"ERROR: {scraped_path} not found -- run scrape_movesets.py --all first.")
        return

    with open(scraped_path) as f:
        all_scraped = json.load(f)

    all_moves = []
    all_movesets = {}
    still_needs_review = []

    for slug, scraped in all_scraped.items():
        moves_out, moveset_out = compile_enemy(slug, scraped)
        all_moves.extend(moves_out)
        all_movesets[moveset_out["enemy_base_id"]] = moveset_out
        if moveset_out["needs_review"]:
            still_needs_review.append((slug, moveset_out["review_reason"]))

    print(f"Compiled {len(all_movesets)} enemies, {len(all_moves)} move definitions.")
    if still_needs_review:
        print(f"\n{len(still_needs_review)} still need review (no override applied yet):")
        for slug, reason in still_needs_review:
            print(f"  - {slug}: {reason}")
    else:
        print("All enemies resolved (either scraped cleanly or covered by overrides).")

    if args.report:
        return

    with open(args.moves_out, "w") as f:
        json.dump(all_moves, f, indent=2)
    with open(args.movesets_out, "w") as f:
        json.dump(all_movesets, f, indent=2)
    print(f"\nWrote {args.moves_out} and {args.movesets_out}")


if __name__ == "__main__":
    main()
