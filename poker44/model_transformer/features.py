"""Hero-centric per-hand feature extraction (self-contained copy).

Each poker hand exposes ``metadata.hero_seat``, the subject player being judged
bot vs human. Hole cards and board cards are obfuscated in the benchmark, so all
signal is behavioral: which actions the hero takes, bet sizings in big blinds,
pot geometry, street depth, aggression, and fold/continuation tendencies.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List

ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}
_STREET_MAX_DEPTH = 4.0

FEATURE_NAMES: List[str] = [
    "hero_acted",
    "num_players_norm",
    "hero_start_stack_bb_norm",
    "streets_reached_norm",
    "hero_reached_flop",
    "hero_reached_turn",
    "hero_reached_river",
    "hero_reached_showdown",
    "showdown_flag",
    "hero_fold_ratio",
    "hero_check_ratio",
    "hero_call_ratio",
    "hero_bet_ratio",
    "hero_raise_ratio",
    "hero_action_count_norm",
    "hero_aggression_factor",
    "hero_aggression_freq",
    "hero_vpip",
    "hero_pfr",
    "hero_folded_preflop",
    "hero_limped",
    "hero_bet_bb_mean_norm",
    "hero_bet_bb_std_norm",
    "hero_bet_bb_min_norm",
    "hero_bet_bb_max_norm",
    "hero_has_allin_like",
    "hero_pot_before_mean_norm",
    "hero_pot_growth_mean",
    "hero_first_to_act",
    "hero_last_to_act",
    "hero_action_entropy",
    "hero_action_run_max_share",
    "hero_bet_bucket_entropy",
]

FEATURE_DIM = len(FEATURE_NAMES)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(max(var, 0.0))


def _entropy(values: list) -> float:
    """Normalized Shannon entropy in [0, 1] over a sequence of discrete values."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log(p + 1e-12)
    return ent / math.log(len(counts))


def _max_run_share(values: list) -> float:
    """Length of the longest run of identical consecutive values / total."""
    if not values:
        return 0.0
    longest = 1
    cur = 1
    for prev, cur_value in zip(values, values[1:]):
        if prev == cur_value:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return longest / len(values)


def _amount_bucket(value: float) -> str:
    if value <= 0.0:
        return "z"
    if value <= 0.5:
        return "xs"
    if value <= 1.0:
        return "s"
    if value <= 2.0:
        return "m"
    if value <= 5.0:
        return "l"
    return "xl"


class HandFeatureExtractor:
    """Turns a single hand dict into a fixed-length hero-centric vector."""

    feature_names = FEATURE_NAMES
    feature_dim = FEATURE_DIM

    _MAX_SEATS = 9.0
    _MAX_STACK_BB = 250.0
    _MAX_BET_BB = 200.0
    _MAX_POT_BB = 250.0
    _MAX_HERO_ACTIONS = 12.0

    def extract(self, hand: dict) -> List[float]:
        hand = hand or {}
        metadata = hand.get("metadata") or {}
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        actions = hand.get("actions") or []
        outcome = hand.get("outcome") or {}

        hero_seat = metadata.get("hero_seat")
        bb = _safe_float(metadata.get("bb"), 0.0)

        hero_actions = [
            a for a in actions if a and a.get("actor_seat") == hero_seat
        ]
        counts = {k: 0 for k in ACTION_TYPES}
        for a in hero_actions:
            atype = a.get("action_type")
            if atype in counts:
                counts[atype] += 1
        hero_total = sum(counts.values())

        hero_acted = 1.0 if hero_total > 0 else 0.0
        num_players_norm = _clip(len(players) / self._MAX_SEATS)

        hero_stack = 0.0
        for p in players:
            if p and p.get("seat") == hero_seat:
                hero_stack = _safe_float(p.get("starting_stack"), 0.0)
                break
        hero_start_stack_bb = (hero_stack / bb) if bb > 0 else 0.0
        hero_start_stack_bb_norm = _clip(hero_start_stack_bb / self._MAX_STACK_BB)

        street_names = [s.get("street") for s in streets if s]
        streets_reached = len(street_names)
        streets_reached_norm = _clip(streets_reached / _STREET_MAX_DEPTH)

        hero_streets = {a.get("street") for a in hero_actions}
        hero_reached_flop = 1.0 if "flop" in hero_streets else 0.0
        hero_reached_turn = 1.0 if "turn" in hero_streets else 0.0
        hero_reached_river = 1.0 if "river" in hero_streets else 0.0
        hero_reached_showdown = 1.0 if "showdown" in street_names and hero_acted else 0.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        denom = max(hero_total, 1)
        hero_fold_ratio = counts["fold"] / denom
        hero_check_ratio = counts["check"] / denom
        hero_call_ratio = counts["call"] / denom
        hero_bet_ratio = counts["bet"] / denom
        hero_raise_ratio = counts["raise"] / denom
        hero_action_count_norm = _clip(hero_total / self._MAX_HERO_ACTIONS)

        aggressive = counts["bet"] + counts["raise"]
        passive = counts["call"] + counts["check"]
        hero_aggression_factor = _clip(aggressive / max(passive, 1) / 3.0)
        hero_aggression_freq = aggressive / denom

        pre = [a for a in hero_actions if a.get("street") == "preflop"]
        pre_types = [a.get("action_type") for a in pre]
        hero_vpip = 1.0 if any(t in ("call", "bet", "raise") for t in pre_types) else 0.0
        hero_pfr = 1.0 if "raise" in pre_types else 0.0
        hero_folded_preflop = 1.0 if pre_types and pre_types[-1] == "fold" else 0.0
        hero_limped = 1.0 if ("call" in pre_types and "raise" not in pre_types) else 0.0

        sizes = [
            _safe_float(a.get("normalized_amount_bb"), 0.0)
            for a in hero_actions
            if a.get("action_type") in ("bet", "raise")
        ]
        sizes = [s for s in sizes if s > 0]
        hero_bet_bb_mean_norm = _clip(_mean(sizes) / self._MAX_BET_BB)
        hero_bet_bb_std_norm = _clip(_std(sizes) / self._MAX_BET_BB)
        hero_bet_bb_min_norm = _clip((min(sizes) if sizes else 0.0) / self._MAX_BET_BB)
        hero_bet_bb_max_norm = _clip((max(sizes) if sizes else 0.0) / self._MAX_BET_BB)
        hero_has_allin_like = 1.0 if any(s >= 100.0 for s in sizes) else 0.0

        pots_before = [_safe_float(a.get("pot_before"), 0.0) for a in hero_actions]
        pots_before_bb = [(p / bb) if bb > 0 else 0.0 for p in pots_before]
        hero_pot_before_mean_norm = _clip(_mean(pots_before_bb) / self._MAX_POT_BB)
        growths = []
        for a in hero_actions:
            pb = _safe_float(a.get("pot_before"), 0.0)
            pa = _safe_float(a.get("pot_after"), 0.0)
            if pb > 0:
                growths.append(_clip((pa - pb) / pb, 0.0, 5.0) / 5.0)
        hero_pot_growth_mean = _mean(growths)

        actor_order = [a.get("actor_seat") for a in actions if a]
        hero_first_to_act = 1.0 if actor_order and actor_order[0] == hero_seat else 0.0
        hero_last_to_act = 1.0 if actor_order and actor_order[-1] == hero_seat else 0.0

        hero_action_seq = [a.get("action_type") for a in hero_actions]
        hero_action_entropy = _entropy(hero_action_seq)
        hero_action_run_max_share = _max_run_share(hero_action_seq)
        hero_bet_buckets = [_amount_bucket(s) for s in sizes]
        hero_bet_bucket_entropy = _entropy(hero_bet_buckets)

        return [
            hero_acted,
            num_players_norm,
            hero_start_stack_bb_norm,
            streets_reached_norm,
            hero_reached_flop,
            hero_reached_turn,
            hero_reached_river,
            hero_reached_showdown,
            showdown_flag,
            hero_fold_ratio,
            hero_check_ratio,
            hero_call_ratio,
            hero_bet_ratio,
            hero_raise_ratio,
            hero_action_count_norm,
            hero_aggression_factor,
            hero_aggression_freq,
            hero_vpip,
            hero_pfr,
            hero_folded_preflop,
            hero_limped,
            hero_bet_bb_mean_norm,
            hero_bet_bb_std_norm,
            hero_bet_bb_min_norm,
            hero_bet_bb_max_norm,
            hero_has_allin_like,
            hero_pot_before_mean_norm,
            hero_pot_growth_mean,
            hero_first_to_act,
            hero_last_to_act,
            hero_action_entropy,
            hero_action_run_max_share,
            hero_bet_bucket_entropy,
        ]

    def extract_batch(self, hands: List[dict]) -> List[List[float]]:
        """Per-hand feature matrix for a batch of hands."""
        if not hands:
            return [[0.0] * self.feature_dim]
        return [self.extract(h) for h in hands]
