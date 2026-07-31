"""
Thin synchronous client for the Slay the Spire 2 mod's local REST API.

This talks directly to the mod's HTTP server (same one STS2MCP's server.py
wraps for MCP/LLM use) with no MCP, FastMCP, or httpx-async dependency —
just `requests`. This is meant to sit underneath your own ML agents
(card-reward selector, combat solver) as the sole interface layer to the
game: everything else in your project should go through this class rather
than talking to the socket directly.

Usage:
    client = STS2Client()
    state = client.get_state()
    if state["state_type"] in ("monster", "elite", "boss"):
        client.play_card(0, target="JAW_WORM_0")
        client.end_turn()
"""

from __future__ import annotations

import time
from typing import Any

import requests


class STS2ActionError(RuntimeError):
    """Raised when the mod's API returns an error response."""


class STS2Client:
    def __init__(self, host: str = "localhost", port: int = 15526, timeout: float = 10.0):
        self.base_url = f"http://{host}:{port}/api/v1/singleplayer"
        self.timeout = timeout
        self._session = requests.Session()

    # -- low-level plumbing --------------------------------------------------

    def _get(self, params: dict | None = None) -> dict[str, Any]:
        r = self._session.get(self.base_url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        r = self._session.post(self.base_url, json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("status") == "error":
            raise STS2ActionError(data.get("error", "unknown error"))
        return data

    # -- state -----------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Full game state as a dict. state['state_type'] tells you the current
        screen: e.g. 'monster' / 'elite' / 'boss' (combat), 'map', 'event',
        'shop', 'fake_merchant', 'rewards', 'card_reward', 'rest_site',
        'relic_select', 'treasure', 'card_select', 'bundle_select',
        'crystal_sphere', 'menu', 'game_over'.
        """
        return self._get({"format": "json"})

    def wait_for_state_type(self, *state_types: str, poll_s: float = 0.1, timeout_s: float = 30.0) -> dict[str, Any]:
        """Block until get_state()['state_type'] is one of state_types. Useful
        after actions that trigger animations/transitions before the next
        decision point is ready."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self.get_state()
            if state.get("state_type") in state_types:
                return state
            time.sleep(poll_s)
        raise TimeoutError(f"Timed out waiting for state_type in {state_types}")

    # -- menu / meta -------------------------------------------------------

    def menu_select(self, option: str, seed: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"action": "menu_select", "option": option}
        if seed is not None:
            body["seed"] = seed
        return self._post(body)

    def use_potion(self, slot: int, target: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"action": "use_potion", "slot": slot}
        if target is not None:
            body["target"] = target
        return self._post(body)

    def discard_potion(self, slot: int) -> dict[str, Any]:
        return self._post({"action": "discard_potion", "slot": slot})

    def proceed_to_map(self) -> dict[str, Any]:
        """Works from: rewards, rest site, shop, fake merchant. NOT events —
        use event_choose_option() with the Proceed option's index instead."""
        return self._post({"action": "proceed"})

    # -- combat --------------------------------------------------------------

    def play_card(self, card_index: int, target: str | None = None) -> dict[str, Any]:
        """Playing a card shifts remaining hand indices — always re-read
        get_state() before your next call rather than reusing stale indices.
        Playing right-to-left keeps indices more stable within a turn."""
        body: dict[str, Any] = {"action": "play_card", "card_index": card_index}
        if target is not None:
            body["target"] = target
        return self._post(body)

    def end_turn(self) -> dict[str, Any]:
        return self._post({"action": "end_turn"})

    # -- in-combat hand selection (exhaust/discard prompts) ------------------

    def combat_select_card(self, card_index: int) -> dict[str, Any]:
        return self._post({"action": "combat_select_card", "card_index": card_index})

    def combat_confirm_selection(self) -> dict[str, Any]:
        return self._post({"action": "combat_confirm_selection"})

    # -- rewards ---------------------------------------------------------------

    def claim_reward(self, reward_index: int) -> dict[str, Any]:
        """Claiming shifts remaining reward indices left — re-read state,
        or claim right-to-left."""
        return self._post({"action": "claim_reward", "index": reward_index})

    def pick_card_reward(self, card_index: int) -> dict[str, Any]:
        return self._post({"action": "select_card_reward", "card_index": card_index})

    def skip_card_reward(self) -> dict[str, Any]:
        return self._post({"action": "skip_card_reward"})

    # -- map -------------------------------------------------------------------

    def choose_map_node(self, node_index: int) -> dict[str, Any]:
        return self._post({"action": "choose_map_node", "index": node_index})

    # -- rest site ---------------------------------------------------------

    def choose_rest_option(self, option_index: int) -> dict[str, Any]:
        return self._post({"action": "choose_rest_option", "index": option_index})

    # -- shop / fake merchant ------------------------------------------------

    def shop_purchase(self, item_index: int) -> dict[str, Any]:
        return self._post({"action": "shop_purchase", "index": item_index})

    # -- events ------------------------------------------------------------

    def choose_event_option(self, option_index: int) -> dict[str, Any]:
        return self._post({"action": "choose_event_option", "index": option_index})

    def advance_dialogue(self) -> dict[str, Any]:
        """Ancient events: call repeatedly until options appear."""
        return self._post({"action": "advance_dialogue"})

    # -- out-of-combat card selection (transform/upgrade/remove/choose) ------

    def select_deck_card(self, card_index: int) -> dict[str, Any]:
        """Toggles selection for multi-select screens; picks immediately for
        choose-a-card screens."""
        return self._post({"action": "select_card", "index": card_index})

    def confirm_deck_selection(self) -> dict[str, Any]:
        return self._post({"action": "confirm_selection"})

    def cancel_deck_selection(self) -> dict[str, Any]:
        return self._post({"action": "cancel_selection"})

    # -- relic selection / treasure ------------------------------------------

    def select_relic(self, relic_index: int) -> dict[str, Any]:
        return self._post({"action": "select_relic", "index": relic_index})

    def skip_relic_selection(self) -> dict[str, Any]:
        return self._post({"action": "skip_relic_selection"})

    def claim_treasure_relic(self, relic_index: int) -> dict[str, Any]:
        return self._post({"action": "claim_treasure_relic", "index": relic_index})

    # -- bundle selection ----------------------------------------------------

    def select_bundle(self, bundle_index: int) -> dict[str, Any]:
        return self._post({"action": "select_bundle", "index": bundle_index})

    def confirm_bundle_selection(self) -> dict[str, Any]:
        return self._post({"action": "confirm_bundle_selection"})

    def cancel_bundle_selection(self) -> dict[str, Any]:
        return self._post({"action": "cancel_bundle_selection"})

    # -- crystal sphere minigame ----------------------------------------------

    def crystal_sphere_set_tool(self, tool: str) -> dict[str, Any]:
        assert tool in ("big", "small")
        return self._post({"action": "crystal_sphere_set_tool", "tool": tool})

    def crystal_sphere_click_cell(self, x: int, y: int) -> dict[str, Any]:
        return self._post({"action": "crystal_sphere_click_cell", "x": x, "y": y})

    def crystal_sphere_proceed(self) -> dict[str, Any]:
        return self._post({"action": "crystal_sphere_proceed"})


if __name__ == "__main__":
    # Smoke test: connect and print whatever screen the game is currently on.
    client = STS2Client()
    try:
        state = client.get_state()
        print(f"Connected. Current state_type: {state.get('state_type')}")
    except requests.ConnectionError:
        print("Could not connect — is Slay the Spire 2 running with the mod enabled?")
