"""
Static content DB (enemy identity + AI transition graphs) + lightweight
parser for STS2 combat states.

Card/enemy-move DEFINITIONS (damage, effects, target) live in moves_db.py.
This file only holds:
- ENEMY_DB: enemy-type-level static properties (entry status, etc.)
- ENEMY_MOVESETS: the TRANSITION GRAPH (which move follows which, at what
  probability) -- referencing move ids that moves_db.MOVES_DB defines.
- The parser that turns a raw get_state() JSON blob into a compact
  CombatState your simulator consumes.

Design principles (unchanged from earlier):
- STATIC data lives in code, keyed by id, looked up once -- never re-parsed
  from JSON on every search node.
- DYNAMIC data (hp, block, status amounts, intents, hand) is parsed out of
  a single get_state() call at the *root* of a decision.
- The parsed CombatState feeds YOUR OWN simulator (not shown here), which
  does the actual expectimax/MCTS entirely in memory. The live game is only
  ever touched at the root (read state) and after a decision is made
  (execute one action) -- never mid-search, since play_card is a real,
  irreversible mutation of the one true combat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from moves_db import TargetType, MOVES_DB, get_enemy_move, to_id, enemy_move_id


# ---------------------------------------------------------------------------
# Static enemy DB. Only put things here that DON'T change per-encounter.
# Do NOT cache hp/max_hp here from a single observation -- those scale with
# ascension/act and you'd be baking in stale numbers.
# ---------------------------------------------------------------------------

ENEMY_DB: dict[str, dict] = {
    "SCROLL_OF_BITING": {"display_name": "Scroll of Biting"},
    # entry_status: statuses applied automatically at combat start, before
    # any move is chosen. Tuple of (status_id, amount). This is a property
    # of the enemy TYPE (always present), not something any single move
    # grants -- keep it here, not in ENEMY_MOVESETS.
    "INKLET": {"display_name": "Inklet", "entry_status": (("SLIPPERY", 1),)},
    "WRIGGLER": {"display_name": "Wriggler"},
    "WATERFALL_GIANT": {"display_name": "Waterfall Giant"},
}


# ---------------------------------------------------------------------------
# Enemy movesets: TRANSITIONS ONLY. Move definitions (damage, effects,
# target, max_consecutive) live in moves_db.MOVES_DB, keyed by the same
# namespaced move ids used here (see moves_db.enemy_move_id).
#
# Sourced from community datamining (e.g. slaythespire2.gg). Not needed for
# depth-1 search since get_state() already hands you the immediate intent
# for free -- this exists for predicting turns BEYOND the one currently
# shown, which is where multi-enemy fights and any 2+ ply search need it.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnemyMoveset:
    enemy_base_id: str
    start_move: str                      # fallback opener when position is unknown/irrelevant
    transitions: dict
    # Optional: overrides start_move based on where this copy sits in a
    # multi-copy pack. Keys are whatever labels position_key_fn produces
    # (see POSITION_KEY_FNS below). Leave as {} for enemies whose opener
    # doesn't depend on position.
    start_move_by_position: dict = field(default_factory=dict)
    # Which resolver to use for start_move_by_position's keys --
    # "front_middle_back" (Inklet-style: outer two vs inner) or
    # "odd_even_slot" (Wriggler-style: alternating by absolute slot
    # number). Add new resolvers to POSITION_KEY_FNS as new patterns show
    # up; most enemies never touch this since pack position only matters
    # for a minority of same-type packs.
    position_key_fn: str = "front_middle_back"
    # If set, this move fires ONCE when the enemy is summoned mid-combat
    # (rather than present at combat start), before falling through to the
    # normal position-based loop on its following turns. None for enemies
    # with no such entry state (the overwhelming majority).
    spawn_placeholder_move: str | None = None
    # If set, the simulator should substitute this move instead of letting
    # the enemy die when its HP would hit 0 (rare "doesn't truly die"
    # bosses, e.g. Waterfall Giant's Steam Eruption death-interrupt). NOT
    # reached via normal transitions -- a condition the simulator checks
    # directly. Fires once per enemy instance.
    on_death_move: str | None = None
    # Moves after which no further move should be predicted -- combat (or
    # this enemy's part in it) ends here. predict_next_move() should never
    # be called with one of these as last_move.
    terminal_moves: frozenset = field(default_factory=frozenset)
    # Effects applied every turn regardless of which move is chosen (e.g.
    # Waterfall Giant gaining 3 Steam Eruption on EVERY move) -- one entry
    # here instead of repeating the same effect on every move in MOVES_DB.
    # Same (op_name, kwargs) format as Move.effects.
    passive_effects: tuple = ()


def _front_middle_back(index: int, pack_size: int) -> str:
    if pack_size == 1:
        return "front"
    if index == 0:
        return "front"
    if index == pack_size - 1:
        return "back"
    return "middle"


def _odd_even_slot(index: int, pack_size: int) -> str:
    slot = index + 1  # 1-indexed, matching how these are described in-game ("slots 1 and 3")
    return "odd" if slot % 2 == 1 else "even"


POSITION_KEY_FNS = {
    "front_middle_back": _front_middle_back,
    "odd_even_slot": _odd_even_slot,
}


ENEMY_MOVESETS: dict[str, EnemyMoveset] = {
    "INKLET": EnemyMoveset(
        enemy_base_id="INKLET",
        start_move="INKLET_JAB",
        transitions={
            "INKLET_JAB": (("INKLET_PIERCING_GAZE", 0.5), ("INKLET_WHIRLWIND", 0.5)),
            "INKLET_WHIRLWIND": (("INKLET_JAB", 1.0),),
            "INKLET_PIERCING_GAZE": (("INKLET_JAB", 1.0),),
        },
        # Front/back land on odd slots (1, 3); middle lands on even (2) --
        # odd_even_slot happens to fully capture front/middle/back for any
        # 3-pack specifically (Inklet packs are always size 3).
        start_move_by_position={"odd": "INKLET_JAB", "even": "INKLET_WHIRLWIND"},
        position_key_fn="odd_even_slot",
    ),
    "WRIGGLER": EnemyMoveset(
        enemy_base_id="WRIGGLER",
        start_move="WRIGGLER_NASTY_BITE",  # arbitrary fallback; real packs always resolve via slot below
        transitions={
            "WRIGGLER_NASTY_BITE": (("WRIGGLER_WRIGGLE", 1.0),),
            "WRIGGLER_WRIGGLE": (("WRIGGLER_NASTY_BITE", 1.0),),
            # WRIGGLER_SPAWNED has no entry -- only reached via
            # spawn_placeholder_move, never a normal move's outgoing edge.
        },
        start_move_by_position={"odd": "WRIGGLER_NASTY_BITE", "even": "WRIGGLER_WRIGGLE"},
        position_key_fn="odd_even_slot",
        spawn_placeholder_move="WRIGGLER_SPAWNED",
    ),
    "WATERFALL_GIANT": EnemyMoveset(
        enemy_base_id="WATERFALL_GIANT",
        start_move="WATERFALL_GIANT_PRESSURIZE",
        transitions={
            "WATERFALL_GIANT_PRESSURIZE": (("WATERFALL_GIANT_STOMP", 1.0),),
            "WATERFALL_GIANT_STOMP": (("WATERFALL_GIANT_RAM", 1.0),),
            "WATERFALL_GIANT_RAM": (("WATERFALL_GIANT_SIPHON", 1.0),),
            "WATERFALL_GIANT_SIPHON": (("WATERFALL_GIANT_PRESSURE_GUN", 1.0),),
            "WATERFALL_GIANT_PRESSURE_GUN": (("WATERFALL_GIANT_PRESSURE_UP", 1.0),),
            "WATERFALL_GIANT_PRESSURE_UP": (("WATERFALL_GIANT_STOMP", 1.0),),
            "WATERFALL_GIANT_ABOUT_TO_BLOW": (("WATERFALL_GIANT_EXPLODE", 1.0),),
            # WATERFALL_GIANT_EXPLODE deliberately has no entry -- terminal.
        },
        on_death_move="WATERFALL_GIANT_ABOUT_TO_BLOW",
        terminal_moves=frozenset({"WATERFALL_GIANT_EXPLODE"}),
        passive_effects=(("apply_status", {"status_id": "STEAM_ERUPTION", "amount": 3, "target": "self"}),),
    ),
    # add more enemies here as you scrape them from slaythespire2.gg
}


def resolve_pack_position(index: int, pack_size: int, key_fn_name: str = "front_middle_back") -> str:
    """Maps an enemy's position within a same-type pack (0-indexed, in
    spawn/slot order) to a position label, via the resolver named on that
    enemy's EnemyMoveset.position_key_fn."""
    return POSITION_KEY_FNS[key_fn_name](index, pack_size)


def resolve_start_move(moveset: EnemyMoveset, index: int, pack_size: int, entered_via_spawn: bool = False) -> str:
    """The opener to use when YOUR OWN simulator spawns this enemy at a
    known pack position (self-play only -- when playing against the live
    game, just use the intent get_state() already gives you instead).

    entered_via_spawn=True for enemies summoned mid-combat rather than
    present at combat start -- returns spawn_placeholder_move for that one
    turn if the enemy has one, then normal position resolution applies on
    its following turns regardless (call this again with
    entered_via_spawn=False from then on)."""
    if entered_via_spawn and moveset.spawn_placeholder_move:
        return moveset.spawn_placeholder_move
    if not moveset.start_move_by_position:
        return moveset.start_move
    position = resolve_pack_position(index, pack_size, moveset.position_key_fn)
    return moveset.start_move_by_position.get(position, moveset.start_move)


def predict_next_move(enemy_base_id: str, last_move, consecutive_count: int):
    """Returns ((move_id, probability), ...) for the enemy's NEXT move, given
    what it just did. Pass last_move=None to get the fight-opening move
    (deterministic: probability 1.0 on start_move).

    This does NOT consult get_state() -- it's a pure prediction for planning
    beyond the turn the API already tells you about. Respects
    max_consecutive (read from moves_db.MOVES_DB, base tier -- this
    constraint doesn't change with ascension) by masking out the just-used
    move and renormalizing over the remaining options when the repeat cap
    is hit.

    Raises if last_move is one of the enemy's terminal_moves -- the
    simulator should check moveset.terminal_moves itself and treat that as
    an end state BEFORE calling this."""
    moveset = ENEMY_MOVESETS[enemy_base_id]

    if last_move is not None and last_move in moveset.terminal_moves:
        raise ValueError(
            f"{last_move} is terminal -- the simulator should treat this as "
            f"an end state instead of predicting a next move."
        )

    if last_move is None:
        return ((moveset.start_move, 1.0),)

    options = moveset.transitions[last_move]
    max_consecutive = MOVES_DB[(last_move, False)].max_consecutive

    if max_consecutive is None or consecutive_count < max_consecutive:
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
# Load scraper-generated movesets (from merge_movesets.py) on top of the
# hand-authored ones above. Hand-authored entries always win -- this only
# fills in enemies you haven't manually added/reviewed yet, and never
# overwrites something already verified by hand.
# ---------------------------------------------------------------------------

def load_generated_movesets(path: str = "enemy_movesets_generated.json") -> dict:
    import json as _json
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        return {}

    with open(p) as f:
        raw = _json.load(f)

    loaded = {}
    for enemy_id, data in raw.items():
        transitions = {src: tuple((t, p) for t, p in edges) for src, edges in data["transitions"].items()}
        loaded[enemy_id] = EnemyMoveset(
            enemy_base_id=data["enemy_base_id"],
            start_move=data["start_move"],
            transitions=transitions,
            start_move_by_position=data.get("start_move_by_position", {}),
            position_key_fn=data.get("position_key_fn", "front_middle_back"),
            spawn_placeholder_move=data.get("spawn_placeholder_move"),
            on_death_move=data.get("on_death_move"),
            terminal_moves=frozenset(data.get("terminal_moves", [])),
        )
    return loaded


for _enemy_id, _moveset in load_generated_movesets().items():
    ENEMY_MOVESETS.setdefault(_enemy_id, _moveset)


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
