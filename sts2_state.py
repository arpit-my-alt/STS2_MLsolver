"""
Static content DB + lightweight parser for STS2 combat states.

Design:
- STATIC data (name, flavor text, keyword glossary) lives in code, keyed by id.
  It is looked up once, never re-parsed from JSON on every search node.
- DYNAMIC data (hp, block, status amounts, intents, hand) is parsed out of a
  single get_state() call at the *root* of a decision, into a compact
  dataclass tree using only ints/enums/tuples — no strings that aren't
  needed for search logic.
- This parsed CombatState is the input to YOUR OWN simulator (not shown
  here) that does the actual expectimax/MCTS. The live game is only ever
  touched at the root (read state) and after a decision is made (execute
  one action) — never mid-search.

Extend CARD_DB / ENEMY_DB as you encounter new ids; treat this file as a
growing cache, not a one-time build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


# ---------------------------------------------------------------------------
# Static card DB. Key: (card_id, is_upgraded). Values you'd actually use in
# a simulator (base_cost is redundant with live state's "cost" field since
# cost can be modified by relics/effects mid-combat -- keep cost from the
# LIVE state, not this table. What belongs here is stuff that never changes
# turn to turn: target type, keyword tags, and a symbolic effect spec your
# simulator's resolver switches on.
# ---------------------------------------------------------------------------

class TargetType(Enum):
    NONE = "None"
    SELF = "Self"
    ANY_ENEMY = "AnyEnemy"
    ALL_ENEMIES = "AllEnemies"
    ANY_PLAYER = "AnyPlayer"  # potions that can target allies in co-op; harmless here


@dataclass(frozen=True)
class CardDef:
    card_id: str
    is_upgraded: bool
    card_type: str            # "Attack" | "Skill" | "Power" | "Curse" | "Status"
    target_type: TargetType
    keywords: tuple           # e.g. ("Exhaust", "Doom") -- tags your resolver checks
    effect: str                # symbolic key into your simulator's effect-resolver dict


CARD_DB: dict[tuple[str, bool], CardDef] = {
    ("DECAY", False): CardDef("DECAY", False, "Curse", TargetType.NONE, ("Unplayable",), "end_turn_self_damage_2"),
    ("COUNTDOWN", False): CardDef("COUNTDOWN", False, "Power", TargetType.SELF, ("Doom",), "power_doom_random_6_start_of_turn"),
    ("DREDGE", False): CardDef("DREDGE", False, "Skill", TargetType.SELF, ("Exhaust",), "discard_to_hand_3_exhaust"),
    ("SCOURGE", False): CardDef("SCOURGE", False, "Skill", TargetType.ANY_ENEMY, ("Doom",), "doom_13_draw_1"),
    ("FRIENDSHIP", False): CardDef("FRIENDSHIP", False, "Power", TargetType.SELF, ("Energy",), "lose_2_str_gain_energy_start_of_turn"),
    ("DOUBT", False): CardDef("DOUBT", False, "Curse", TargetType.NONE, ("Unplayable",), "end_turn_self_weak_1"),
    # add entries as you encounter them; consider auto-populating this from
    # a one-time scrape of every card_reward / draw_pile / discard_pile
    # payload you see across many runs, rather than hand-typing all ~200+.
}


# ---------------------------------------------------------------------------
# Static enemy DB. Only put things here that DON'T change per-encounter
# (e.g. keyword glossary text). Do NOT cache hp/max_hp/moveset probabilities
# here from a single observation -- those scale with ascension/act and
# you'd be baking in stale numbers. If you want enemy AI transition
# probabilities for deeper search (beyond the 1-ply intent the API already
# gives you for free), that has to come from datamined movesets (community
# wikis/slaythespire2.gg), not from your own play logs alone.
# ---------------------------------------------------------------------------

ENEMY_DB: dict[str, dict] = {
    "SCROLL_OF_BITING": {"display_name": "Scroll of Biting"},
    "INKLET": {"display_name": "Inklet"},
}


# ---------------------------------------------------------------------------
# Enemy movesets: the probabilistic AI transition graphs (sourced from
# community datamining, e.g. slaythespire2.gg). NOT needed for depth-1
# search since get_state() already hands you the immediate intent for
# free -- this exists for predicting turns BEYOND the one currently shown,
# which is where multi-enemy fights and any 2+ ply search actually need it.
#
# Schema notes:
# - "dmg" is keyed by the ascension tier at which that damage value first
#   applies (not every individual ascension number) -- use get_move_damage()
#   below, which picks the highest key <= current ascension.
# - "max_consecutive" caps how many times in a row a move can repeat.
# - Trust the game's textual constraint notes over any auto-rendered graph
#   diagram when transcribing these -- diagrams from generic layout tools
#   can visually imply edges (e.g. bidirectional arrows) that the actual
#   rule text contradicts.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveDef:
    move_id: str
    hits: int
    dmg_by_ascension: dict
    max_consecutive: int
    target: str = "Player"


@dataclass(frozen=True)
class EnemyMoveset:
    enemy_base_id: str
    moves: dict
    start_move: str
    transitions: dict


ENEMY_MOVESETS: dict[str, EnemyMoveset] = {
    "INKLET": EnemyMoveset(
        enemy_base_id="INKLET",
        moves={
            "JAB": MoveDef("JAB", hits=1, dmg_by_ascension={0: 3, 1: 4}, max_consecutive=2),
            "WHIRLWIND": MoveDef("WHIRLWIND", hits=3, dmg_by_ascension={0: 2, 1: 3}, max_consecutive=1),
            "PIERCING_GAZE": MoveDef("PIERCING_GAZE", hits=1, dmg_by_ascension={0: 10, 1: 11}, max_consecutive=1),
        },
        start_move="JAB",
        transitions={
            "JAB": (("PIERCING_GAZE", 0.5), ("WHIRLWIND", 0.5)),
            "WHIRLWIND": (("JAB", 1.0),),
            "PIERCING_GAZE": (("JAB", 1.0),),
        },
    ),
    # add more enemies here as you scrape them from slaythespire2.gg
}


def get_move_damage(move: MoveDef, ascension: int) -> int:
    """Highest ascension-tier damage value at or below the current ascension."""
    applicable_tiers = [tier for tier in move.dmg_by_ascension if tier <= ascension]
    tier = max(applicable_tiers) if applicable_tiers else min(move.dmg_by_ascension)
    return move.dmg_by_ascension[tier]


def predict_next_move(enemy_base_id: str, last_move, consecutive_count: int):
    """Returns ((move_id, probability), ...) for the enemy's NEXT move, given
    what it just did. Pass last_move=None to get the fight-opening move
    (deterministic: probability 1.0 on start_move).

    This does NOT consult get_state() -- it's a pure prediction for planning
    beyond the turn the API already tells you about. Respects
    max_consecutive by masking out the just-used move and renormalizing
    over the remaining transition options when the repeat cap is hit.
    """
    moveset = ENEMY_MOVESETS[enemy_base_id]

    if last_move is None:
        return ((moveset.start_move, 1.0),)

    options = moveset.transitions[last_move]
    last_move_def = moveset.moves[last_move]

    if consecutive_count < last_move_def.max_consecutive:
        return options

    # Repeat cap hit: drop last_move from the distribution and renormalize.
    filtered = [(m, p) for m, p in options if m != last_move]
    total = sum(p for _, p in filtered)
    if total == 0:
        # No valid alternative in the transition table -- data gap, fall
        # back to the raw options rather than crash.
        return options
    return tuple((m, p / total) for m, p in filtered)


# ---------------------------------------------------------------------------
# Dynamic parse targets -- compact, numeric, no flavor text.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Intent:
    kind: str          # "Attack" | "Buff" | "Debuff" | "Block" | "Unknown" ...
    hits: int          # number of separate hits this turn (0 if non-attack)
    dmg_per_hit: int    # 0 if non-attack or unknown


_INTENT_LABEL_RE = re.compile(r"^(\d+)(?:x(\d+))?$")


def parse_intent(raw_intent: dict) -> Intent:
    label = raw_intent.get("label") or ""
    m = _INTENT_LABEL_RE.match(label)
    if not m:
        return Intent(kind=raw_intent["type"], hits=0, dmg_per_hit=0)
    dmg = int(m.group(1))
    hits = int(m.group(2)) if m.group(2) else 1
    return Intent(kind=raw_intent["type"], hits=hits, dmg_per_hit=dmg)


@dataclass(frozen=True)
class StatusEffect:
    status_id: str      # e.g. "PAPER_CUTS_POWER" -- look up meaning in your own glossary, not per-state
    amount: int


@dataclass(frozen=True)
class EnemyState:
    entity_id: str       # e.g. "SCROLL_OF_BITING_0" -- unique per-fight instance
    base_id: str         # e.g. "SCROLL_OF_BITING" -- key into ENEMY_DB
    hp: int
    block: int
    status: tuple[StatusEffect, ...]
    intents: tuple[Intent, ...]


@dataclass(frozen=True)
class CardInstance:
    card_id: str
    is_upgraded: bool
    cost: int            # -1 for "X" cost cards; check the raw cost string for "X" before int()
    can_play: bool
    target_type: TargetType
    index: int            # position in hand THIS turn only -- do not persist across actions


@dataclass(frozen=True)
class PlayerState:
    hp: int
    block: int
    energy: int
    hand: tuple[CardInstance, ...]
    draw_pile_count: int
    discard_pile_count: int
    exhaust_pile_count: int
    status: tuple[StatusEffect, ...]
    osty_hp: int | None      # None if character has no companion / it's dead


@dataclass(frozen=True)
class CombatState:
    round_num: int
    is_play_phase: bool
    enemies: tuple[EnemyState, ...]
    player: PlayerState


def _entity_base_id(entity_id: str) -> str:
    """'SCROLL_OF_BITING_0' -> 'SCROLL_OF_BITING'. Assumes trailing '_<int>'
    instance suffix, which is what combat_id-per-copy implies here."""
    return re.sub(r"_\d+$", "", entity_id)


def parse_combat_state(raw: dict) -> CombatState:
    battle = raw["battle"]
    player_raw = raw["player"]

    enemies = tuple(
        EnemyState(
            entity_id=e["entity_id"],
            base_id=_entity_base_id(e["entity_id"]),
            hp=e["hp"],
            block=e["block"],
            status=tuple(StatusEffect(s["id"], s["amount"]) for s in e["status"]),
            intents=tuple(parse_intent(i) for i in e["intents"]),
        )
        for e in battle["enemies"]
    )

    hand = tuple(
        CardInstance(
            card_id=c["id"],
            is_upgraded=c["is_upgraded"],
            cost=(-1 if c["cost"] == "X" else int(c["cost"])),
            can_play=c["can_play"],
            target_type=TargetType(c["target_type"]),
            index=c["index"],
        )
        for c in player_raw["hand"]
    )

    pets = player_raw.get("pets") or []
    osty = next((p for p in pets if p["id"] == "OSTY" and p["alive"]), None)

    player = PlayerState(
        hp=player_raw["hp"],
        block=player_raw["block"],
        energy=player_raw["energy"],
        hand=hand,
        draw_pile_count=player_raw["draw_pile_count"],
        discard_pile_count=player_raw["discard_pile_count"],
        exhaust_pile_count=player_raw["exhaust_pile_count"],
        status=tuple(StatusEffect(s["id"], s["amount"]) for s in player_raw["status"]),
        osty_hp=(osty["hp"] if osty else None),
    )

    return CombatState(
        round_num=battle["round"],
        is_play_phase=battle["is_play_phase"],
        enemies=enemies,
        player=player,
    )


if __name__ == "__main__":
    # Smoke test using the payload shape you already confirmed works.
    import json
    from sts2_client import STS2Client

    client = STS2Client()
    raw = client.get_state()
    if raw.get("state_type") == "monster":
        state = parse_combat_state(raw)
        print(f"Round {state.round_num}, player hp {state.player.hp}/{state.player.block} block")
        for e in state.enemies:
            print(f"  {e.entity_id}: hp={e.hp} intents={e.intents}")
    else:
        print(f"Not in combat right now (state_type={raw.get('state_type')})")
