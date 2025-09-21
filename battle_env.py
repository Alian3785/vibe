# ============================================================
# Gymnasium Environment: "RED vs BLUE" battle
# - Action space = 6 (targets pos1..pos6)
# - Includes big-units, poison/burn effects, accuracy2
# This module only defines data and the env. No training or viz here.
# ============================================================

from __future__ import annotations

import random
from operator import itemgetter
from typing import List, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
#5555555
#4y3443hjerjdfjdsfjsdfj
# --- encoding vocabularies ---
TYPE_LIST = [
    "Archer", "gargoil", "Mage", "Воин", "Demon", "Death", "lord", "Dead dragon"
]
ATTACK_TYPES = ["Weapon", "earth", "Fire", "poison", "death", "Mind"]


def _one_hot(value: str, vocab: List[str]) -> List[float]:
    return [1.0 if value == v else 0.0 for v in vocab]


def _multi_hot(values: List[str], vocab: List[str]) -> List[float]:
    values_set = set(values or [])
    return [1.0 if v in values_set else 0.0 for v in vocab]


# ------------------------ Units ------------------------
UNITS_RED = [
    {"Name": "скелет-рыцарь",  "Initiative": 50, "BaseInitiative": 50, "team": "red",  "position": 1, "stand": "ahead",
     "Type": "Воин", "Damage": 100, "Damage2": 0, "Health": 270, "maxhealth": 270, "Armor": 0, "Accuracy": 80, "Accuracy2": 0,
     "Immunity": ["death", "Poison"], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "", "big": False},

    {"Name": "рыцарь смерти",  "Initiative": 50, "BaseInitiative": 50, "team": "red",  "position": 2, "stand": "ahead",
     "Type": "Воин", "Damage": 120, "Damage2": 0, "Health": 255, "maxhealth": 255, "Armor": 0, "Accuracy": 87, "Accuracy2": 0,
     "Immunity": ["death"], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "", "big": False},

    {"Name": "Мертвый дракон", "Initiative": 35, "BaseInitiative": 35, "team": "red", "position": 3, "stand": "ahead",
     "Type": "Dead dragon", "Damage": 65, "Damage2": 20, "Health": 450, "maxhealth": 450, "Armor": 0, "Accuracy": 80, "Accuracy2": 40,
     "Immunity": ["death"], "Resilience": [], "AttackType1": "death", "AttackType2": "poison", "big": True},

    {"Name": "Архилич", "Initiative": 40, "BaseInitiative": 40, "team": "red", "position": 4, "stand": "behind",
     "Type": "Mage", "Damage": 90, "Damage2": 0, "Health": 170, "maxhealth": 170, "Armor": 0, "Accuracy": 80, "Accuracy2": 0,
     "Immunity": ["death"], "Resilience": [], "AttackType1": "earth", "AttackType2": "", "big": False},

    {"Name": "Смерть1", "Initiative": 40, "BaseInitiative": 40, "team": "red", "position": 5, "stand": "behind",
     "Type": "Death", "Damage": 100, "Damage2": 20, "Health": 125, "maxhealth": 125, "Armor": 0, "Accuracy": 80, "Accuracy2": 50,
     "Immunity": ["Weapon", "death"], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "poison", "big": False},

    {"Name": "пусто", "Initiative": 0, "BaseInitiative": 0, "team": "red", "position": 6, "stand": "behind",
     "Type": "Archer", "Damage": 0, "Damage2": 0, "Health": 0, "maxhealth": 0, "Armor": 0, "Accuracy": 0, "Accuracy2": 0,
     "Immunity": ["death"], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "", "big": False},
]

UNITS_BLUE = [
    {"Name": "Астерот",   "Initiative": 50, "BaseInitiative": 50, "team": "blue", "position": 7,  "stand": "ahead",
     "Type": "Demon","Damage": 150, "Damage2": 0, "Health": 1020, "maxhealth": 1020, "Armor": 0, "Accuracy": 80, "Accuracy2": 0,
     "Immunity": [], "Resilience": ["Mind", "Fire"], "AttackType1": "Weapon", "AttackType2": "", "big": True},

    {"Name": "Герцог",  "Initiative": 50, "BaseInitiative": 50, "team": "blue", "position": 8,  "stand": "ahead",
     "Type": "Воин", "Damage": 120, "Damage2": 0, "Health": 306, "maxhealth": 306, "Armor": 0, "Accuracy": 100, "Accuracy2": 0,
     "Immunity": [], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "", "big": False},

    {"Name": "Гаргулья", "Initiative": 60, "BaseInitiative": 60, "team": "blue", "position": 9,  "stand": "ahead",
     "Type": "gargoil","Damage": 85, "Damage2": 0, "Health": 170, "maxhealth": 170, "Armor": 65, "Accuracy": 80, "Accuracy2": 0,
     "Immunity": ["poison"], "Resilience": ["Mind"], "AttackType1": "earth", "AttackType2": "", "big": True},

    {"Name": "пусто",    "Initiative": 0,  "BaseInitiative": 0,  "team": "blue", "position": 10, "stand": "behind",
     "Type": "Archer","Damage": 0,  "Damage2": 0, "Health": 0, "maxhealth": 0, "Armor": 0,  "Accuracy": 0, "Accuracy2": 0,
     "Immunity": [], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "", "big": False},

    {"Name": "пусто",    "Initiative": 0,  "BaseInitiative": 0,  "team": "blue", "position": 11, "stand": "behind",
     "Type": "Archer","Damage": 0,  "Damage2": 0, "Health": 0, "maxhealth": 0, "Armor": 0,  "Accuracy": 0, "Accuracy2": 0,
     "Immunity": [], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "", "big": False},

    {"Name": "пусто",    "Initiative": 0,  "BaseInitiative": 0,  "team": "blue", "position": 12, "stand": "behind",
     "Type": "Archer","Damage": 0,  "Damage2": 0, "Health": 0, "maxhealth": 0, "Armor": 0,  "Accuracy": 0, "Accuracy2": 0,
     "Immunity": [], "Resilience": [], "AttackType1": "Weapon", "AttackType2": "", "big": False},
]

# Position ranges
RED_POSITIONS:  List[int] = list(range(1, 7))
BLUE_POSITIONS: List[int] = list(range(7, 13))

# Available targets for the agent — enemies only (pos1..pos6)
TARGET_POSITIONS: List[int] = RED_POSITIONS

# Constants for observation normalization
MAX_HP   = max([u["Health"] for u in (UNITS_RED + UNITS_BLUE)])
MAX_INIT = max([u["BaseInitiative"] for u in (UNITS_RED + UNITS_BLUE)])
MAX_DMG  = max([u["Damage"] for u in (UNITS_RED + UNITS_BLUE)])
MAX_DMG2 = max([u["Damage2"] for u in (UNITS_RED + UNITS_BLUE)])

POISON_TURNS  = 3
BURN_DAMAGE   = 10
BURN_TURNS    = 3


class BattleEnv(gym.Env):
    """
    Observation (540): 12 slots × 45 features (includes Accuracy2).
    Action space: Discrete(6) — choose a target among enemies (pos1..pos6).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        reward_step: float = 0.0,
        penalty_invalid_target: float = -0.10,
        penalty_unreachable_warrior: float = -0.20,
        log_enabled: bool = False,
    ):
        super().__init__()
        self.reward_win  = float(reward_win)
        self.reward_loss = float(reward_loss)
        self.reward_step = float(reward_step)

        self.penalty_invalid_target = float(penalty_invalid_target)
        self.penalty_unreachable_warrior = float(penalty_unreachable_warrior)

        self.log_enabled = bool(log_enabled)
        self._pretty_events: List[str] = []

        # Action space = 6
        self.action_space = spaces.Discrete(len(TARGET_POSITIONS))

        # Observation space
        low_unit = np.array(
            [0, 0, 0, 0, 0, 0, 1, 0]  # HP, INI, INI_base, DMG, DMG2, team, pos, stand
            + [0]*len(TYPE_LIST)       # type one-hot
            + [0]*len(ATTACK_TYPES)    # immunity
            + [0]*len(ATTACK_TYPES)    # atk1
            + [0]*len(ATTACK_TYPES)    # atk2
            + [0]*len(ATTACK_TYPES)    # resilience
            + [0]                      # armor
            + [0]                      # accuracy
            + [0]                      # accuracy2
            + [0, 0],                  # poison_left, burn_left
            dtype=np.float32,
        )
        high_unit = np.array(
            [MAX_HP, MAX_INIT, MAX_INIT, MAX_DMG, MAX_DMG2, 1, 12, 1]
            + [1]*len(TYPE_LIST)
            + [1]*len(ATTACK_TYPES)
            + [1]*len(ATTACK_TYPES)
            + [1]*len(ATTACK_TYPES)
            + [1]*len(ATTACK_TYPES)
            + [100]         # armor
            + [100]         # accuracy
            + [100]         # accuracy2
            + [POISON_TURNS, BURN_TURNS],
            dtype=np.float32,
        )
        low  = np.tile(low_unit, 12)
        high = np.tile(high_unit, 12)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Battle state
        self.rng = random.Random()
        self.combined: List[Dict] = []
        self.round_no: int = 1
        self.winner: Optional[str] = None
        self.current_blue_attacker_pos: Optional[int] = None
        self.blue_attacks_left: int = 0
        self._lord_applied_burn: Dict[int, bool] = {}

    # ------------------ Helpers ------------------
    def seed(self, seed=None):
        self.rng = random.Random(seed)

    def _alive(self, u) -> bool:
        return u["Health"] > 0

    def _team_alive(self, team: str) -> bool:
        return any(self._alive(u) and u["team"] == team for u in self.combined)

    def _unit_by_position(self, pos: int):
        return next((u for u in self.combined if u["position"] == pos), None)

    def _live_positions_of(self, team: str):
        return [u["position"] for u in self.combined if u["team"] == team and self._alive(u)]

    def _log(self, s: str):
        if self.log_enabled:
            self._pretty_events.append(s)

    def pop_pretty_events(self) -> List[str]:
        out = self._pretty_events[:]
        self._pretty_events.clear()
        return out

    def _col_of(self, pos: int) -> int:
        return (pos - 1) % 3

    def _enemy_rows(self, team: str):
        if team == "red":
            return [7, 8, 9], [10, 11, 12]
        else:
            return [1, 2, 3], [4, 5, 6]

    def _warrior_near_far_cols(self, col: int):
        if col == 0:
            return [0, 1], [2]
        if col == 1:
            return [0, 1, 2], []
        return [1, 2], [0]

    def _warrior_allowed_targets(self, attacker: Dict) -> List[int]:
        assert attacker.get("Type") in ("Воин", "Demon", "lord")
        if attacker["stand"] == "behind":
            if any(
                self._alive(u)
                and u["team"] == attacker["team"]
                and u.get("Type") in ("Воин", "Demon", "lord")
                and u["stand"] == "ahead"
                for u in self.combined
            ):
                return []
        ahead, behind = self._enemy_rows(attacker["team"])
        col = self._col_of(attacker["position"])
        near_cols, far_cols = self._warrior_near_far_cols(col)

        def alive(pos):
            uu = self._unit_by_position(pos)
            return uu is not None and self._alive(uu)

        near = [ahead[c] for c in near_cols if alive(ahead[c])]
        if near:
            return near
        far = [ahead[c] for c in far_cols if alive(ahead[c])]
        if far:
            return far

        near2 = [behind[c] for c in near_cols if alive(behind[c])]
        if near2:
            return near2
        far2 = [behind[c] for c in far_cols if alive(behind[c])]
        return far2

    def _pick_lowest_hp(self, positions: List[int]) -> Optional[int]:
        candidates = []
        for p in positions:
            u = self._unit_by_position(p)
            if u is not None and self._alive(u):
                candidates.append((u["Health"], p))
        if not candidates:
            return None
        min_hp = min(hp for hp, _ in candidates)
        best = [p for hp, p in candidates if hp == min_hp]
        return self.rng.choice(best)

    # ----------- start-of-turn effects (poison/burn) -----------
    def _apply_start_of_turn_effects(self, unit: Dict) -> bool:
        # Poison
        if unit.get("poison_turns_left", 0) > 0 and self._alive(unit):
            if "poison" in (unit.get("Immunity") or []):
                unit["poison_turns_left"] = 0
                unit["poison_damage_per_tick"] = 0
                self._log(
                    f"🛡 Иммунитет к эффекту 'poison' — яд не действует на {unit['team'].upper()} {unit['Name']}#{unit['position']}."
                )
            else:
                tick = int(unit.get("poison_damage_per_tick", 0) or 0)
                if tick > 0:
                    before = unit["Health"]
                    unit["Health"] -= tick
                    unit["poison_turns_left"] -= 1
                    after = unit["Health"]
                    self._log(
                        f"☠ Яд поражает {unit['team'].upper()} {unit['Name']}#{unit['position']}: "
                        f"{tick} ({before}→{max(0, after)}); осталось ходов: {unit['poison_turns_left']}"
                    )
                    if unit["Health"] <= 0:
                        unit["инициатива"] = 0
                        self._log(f"✖ {unit['team'].upper()} {unit['Name']}#{unit['position']} погибает от яда.")
                        self._check_victory_after_hit()
                        return False
                else:
                    unit["poison_turns_left"] -= 1
                if unit["poison_turns_left"] <= 0:
                    unit["poison_damage_per_tick"] = 0

        # Burn — tick damage from burn_damage_per_tick (for lord = его урон2)
        if unit.get("burn_turns_left", 0) > 0 and self._alive(unit):
            if "Fire" in (unit.get("Immunity") or []):
                unit["burn_turns_left"] = 0
                unit["burn_damage_per_tick"] = 0
                self._log(
                    f"🛡 Иммунитет к эффекту 'Fire' — поджог не действует на {unit['team'].upper()} {unit['Name']}#{unit['position']}."
                )
            else:
                tick = int(unit.get("burn_damage_per_tick", 0) or BURN_DAMAGE)
                if tick > 0:
                    before = unit["Health"]
                    unit["Health"] -= tick
                    unit["burn_turns_left"] -= 1
                    after = unit["Health"]
                    self._log(
                        f"🔥 Поджог поражает {unit['team'].upper()} {unit['Name']}#{unit['position']}: "
                        f"{tick} ({before}→{max(0, after)}); осталось ходов: {unit['burn_turns_left']}"
                    )
                    if unit["Health"] <= 0:
                        unit["инициатива"] = 0
                        self._log(f"✖ {unit['team'].upper()} {unit['Name']}#{unit['position']} погибает от поджога.")
                        self._check_victory_after_hit()
                        return False
                else:
                    unit["burn_turns_left"] -= 1
                if unit["burn_turns_left"] <= 0:
                    unit["burn_damage_per_tick"] = 0

        return True

    # ------------------ Core battle logic ------------------
    def _reset_state(self):
        self.combined = [u.copy() for u in (UNITS_RED + UNITS_BLUE)]
        for u in self.combined:
            u["Initiative"] = u.get("BaseInitiative", 0)
            u.setdefault("AttackType1", "Weapon")
            u.setdefault("AttackType2", "")
            u.setdefault("Immunity", [])
            u.setdefault("Resilience", [])
            u.setdefault("Armor", 0)
            u.setdefault("Accuracy", 100)
            u.setdefault("Accuracy2", 0)
            u.setdefault("Damage2", 0)
            u.setdefault("big", False)
            u["poison_turns_left"] = 0
            u["poison_damage_per_tick"] = 0
            u["burn_turns_left"] = 0
            u["burn_damage_per_tick"] = 0
            u["resilience_used_types"] = []
        self.round_no = 1
        self.winner = None
        self.current_blue_attacker_pos = None
        self.blue_attacks_left = 0
        self._lord_applied_burn = {}
        self._log(f"Эпизод начат. Раунд {self.round_no}.")

    def _candidates(self):
        return [u for u in self.combined if self._alive(u) and u["Initiative"] > 0]

    def _pop_next(self):
        cand = self._candidates()
        if not cand:
            return None
        self.rng.shuffle(cand)
        cand.sort(key=itemgetter("Initiative"), reverse=True)
        return cand[0]

    def _end_round_restore(self):
        for u in self.combined:
            u["Initiative"] = u.get("BaseInitiative", 0) if self._alive(u) else 0
        self._log("Восстановление инициативы. Новый раунд.")

    def _is_immune_damage(self, attacker: Dict, victim: Dict) -> bool:
        atk1 = attacker.get("AttackType1", "Weapon")
        return atk1 in (victim.get("Immunity") or [])

    def _is_immune_status(self, attacker: Dict, victim: Dict) -> bool:
        atk2 = attacker.get("AttackType2", "")
        return bool(atk2) and (atk2 in (victim.get("Immunity") or []))

    def _resilience_blocks(self, attacker: Dict, victim: Dict) -> bool:
        res_list = set(victim.get("Resilience") or [])
        if not res_list:
            return False
        used = set(victim.get("resilience_used_types") or [])
        available = res_list - used
        if not available:
            return False
        atk1 = attacker.get("AttackType1", "")
        atk2 = attacker.get("AttackType2", "")
        for tag in (atk1, atk2):
            if tag and tag in available:
                victim.setdefault("resilience_used_types", []).append(tag)
                self._log(
                    f"🧿 Стойкость — первый удар типа '{tag}' по "
                    f"{victim['team'].upper()} {victim['Name']}#{victim['position']} поглощён."
                )
                return True
        return False

    def _apply_damage_with_armor(self, base_dmg: float, victim: Dict) -> int:
        armor = float(victim.get("Armor", 0) or 0)
        factor = (1.0 - armor / 100.0) if armor > 0 else 1.0
        eff = base_dmg * max(0.0, factor)
        return int(round(eff))

    def _roll_hit(self, attacker: Dict) -> bool:
        acc = float(attacker.get("Accuracy", 100) or 0)
        return self.rng.random() < (acc / 100.0)

    def _roll_status(self, chance_percent: float) -> bool:
        cp = max(0.0, min(100.0, float(chance_percent or 0.0)))
        return self.rng.random() < (cp / 100.0)

    def _attack(self, attacker, target_pos: Optional[int]):
        # AoE for Mage and Dead dragon
        if attacker is not None and attacker.get("Type") in ("Mage", "Dead dragon"):
            enemy_team = "blue" if attacker["team"] == "red" else "red"
            targets = [u for u in self.combined if u["team"] == enemy_team and self._alive(u)]
            if targets:
                atype = attacker.get("Type")
                self._log(f"{attacker['team'].upper()} {attacker['Name']}#{attacker['position']} ({atype}) применяет массовую атаку.")
                for victim in targets:
                    if not self._roll_hit(attacker):
                        self._log(
                            f"💨 Промах: {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} по "
                            f"{victim['team'].upper()} {victim['Name']}#{victim['position']}."
                        )
                        continue
                    if self._is_immune_damage(attacker, victim):
                        self._log(
                            f"🛡 Иммунитет к урону '{attacker.get('AttackType1','')}' — "
                            f"{victim['team'].upper()} {victim['Name']}#{victim['position']}: урон 0."
                        )
                        continue
                    if self._resilience_blocks(attacker, victim):
                        continue
                    before = victim["Health"]
                    dmg = self._apply_damage_with_armor(attacker["Damage"], victim)
                    victim["Health"] -= dmg
                    after = victim["Health"]
                    self._log(
                        f"{attacker['team'].upper()} {attacker['Name']}#{attacker['position']} → "
                        f"{victim['team'].upper()} {victim['Name']}#{victim['position']}: {dmg} "
                        f"({before}→{max(0, after)})"
                    )
                    if victim["Health"] <= 0:
                        victim["Initiative"] = 0
                        self._log(f"✖ {victim['team'].upper()} {victim['Name']}#{victim['position']} выведен из строя.")
                    else:
                        if atype == "Dead dragon" and attacker.get("AttackType2", "") == "poison":
                            acc2 = float(attacker.get("Accuracy2", 0) or 0)
                            roll = self._roll_status(acc2)
                            self._log(
                                f"🧪 Шанс отравления {int(acc2)}% — " + ("успех" if roll else "неудача")
                                + f" по {victim['team'].upper()} {victim['Name']}#{victim['position']}."
                            )
                            if roll:
                                if self._is_immune_status(attacker, victim):
                                    self._log(
                                        f"🛡 Иммунитет к эффекту '{attacker.get('AttackType2','')}' — яд НЕ накладывается "
                                        f"на {victim['team'].upper()} {victim['Name']}#{victim['position']}."
                                    )
                                else:
                                    victim["poison_turns_left"] = POISON_TURNS
                                    victim["poison_damage_per_tick"] = int(attacker.get("Damage2", 0) or 0)
                                    self._log(
                                        f"☠ {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} накладывает яд "
                                        f"({victim['poison_damage_per_tick']} урона/ход) на "
                                        f"{victim['team'].upper()} {victim['Name']}#{victim['position']} на {POISON_TURNS} хода."
                                    )
                return True, "aoe"
            else:
                self._log(
                    f"{attacker['team'].upper()} {attacker['Name']}#{attacker['position']} ({attacker.get('Type')}) применяет массовую атаку: целей нет."
                )
                return False, "aoe_no_targets"

        # Single-target
        victim = self._unit_by_position(target_pos) if target_pos is not None else None
        if victim is not None and self._alive(victim):
            if not self._roll_hit(attacker):
                self._log(
                    f"💨 Промах: {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} по "
                    f"{victim['team'].upper()} {victim['Name']}#{victim['position']}."
                )
                return True, "miss"

            if self._is_immune_damage(attacker, victim):
                self._log(
                    f"🛡 Иммунитет к урону '{attacker.get('AttackType1','')}' — "
                    f"{victim['team'].upper()} {victim['Name']}#{victim['position']}: атака наносит 0."
                )
                return True, "immune_damage"

            if self._resilience_blocks(attacker, victim):
                return True, "resilience_block"

            before = victim["Health"]
            dmg = self._apply_damage_with_armor(attacker["Damage"], victim)
            victim["Health"] -= dmg
            after = victim["Health"]
            self._log(
                f"{attacker['team'].upper()} {attacker['Name']}#{attacker['position']} → "
                f"{victim['team'].upper()} {victim['Name']}#{victim['position']}: {dmg} "
                f"({before}→{max(0, after)})"
            )

            if self._alive(victim):
                if (
                    attacker.get("Type") == "Death"
                    and attacker.get("Тип атаки 2", "") == "poison"
                    and victim.get("poison_turns_left", 0) <= 0
                ):
                    acc2 = float(attacker.get("Accuracy2", 0) or 0)
                    roll = self._roll_status(acc2)
                    self._log(
                        f"🧪 Шанс отравления {int(acc2)}% — "
                        + ("успех" if roll else "неудача")
                        + f" по {victim['team'].upper()} {victim['Name']}#{victim['position']}."
                    )
                    if roll:
                        if self._is_immune_status(attacker, victim):
                            self._log(
                                f"🛡 Иммунитет к эффекту '{attacker.get('AttackType2','')}' — яд НЕ накладывается "
                                f"на {victim['team'].upper()} {victim['Name']}#{victim['position']}."
                            )
                        else:
                            victim["poison_turns_left"] = POISON_TURNS
                            victim["poison_damage_per_tick"] = int(attacker.get("Damage2", 0) or 0)
                            self._log(
                                f"☠ {attacker['team'].upper()} {attacker['Name']}#{attacker['position']} накладывает яд "
                                f"({victim['poison_damage_per_tick']} урона/ход) на "
                                f"{victim['team'].upper()} {victim['Name']}#{victim['position']} на {POISON_TURNS} хода."
                            )

                if attacker.get("Type") == "lord":
                    pos = attacker["position"]
                    if not self._lord_applied_burn.get(pos, False):
                        if self._is_immune_status(attacker, victim):
                            self._log(
                                f"🛡 Иммунитет к эффекту '{attacker.get('Тип атаки 2','')}' — поджог НЕ накладывается "
                                f"на {victim['team'].upper()} {victim['Name']}#{victim['position']}."
                            )
                        else:
                            victim["burn_turns_left"] = BURN_TURNS
                            victim["burn_damage_per_tick"] = int(attacker.get("Damage2", 0) or 0)
                            self._lord_applied_burn[pos] = True
                            self._log(
                                f"🔥 {attacker['team'].upper()} {attacker['Name']}#{pos} накладывает поджог "
                                f"({victim['burn_damage_per_tick']} урона/ход) на "
                                f"{victim['team'].upper()} {victim['Name']}#{victim['position']} на {BURN_TURNS} хода."
                            )

            if victim["Health"] <= 0:
                victim["Initiative"] = 0
                self._log(f"✖ {victim['team'].upper()} {victim['Name']}#{victim['position']} выведен из строя.")
            return True, "ok"
        else:
            self._log(
                f"{attacker['team'].upper()} {attacker['Name']}#{attacker['position']} бьёт pos{target_pos}: цели нет/мертва/недоступна."
            )
            return False, "dead_or_absent"

    def _check_victory_after_hit(self):
        if not self._team_alive("blue"):
            self.winner = "red";  self._log("🏆 Победа RED!")
        elif not self._team_alive("red"):
            self.winner = "blue"; self._log("🏆 Победа BLUE!")

    def _advance_until_blue_turn(self) -> bool:
        """Auto-advance until next BLUE turn or end of battle."""
        while self._team_alive("red") and self._team_alive("blue"):
            nxt = self._pop_next()
            if nxt is None:
                self._log(f"— Конец раунда {self.round_no}.")
                self._end_round_restore()
                self.round_no += 1
                self._log(f"— Начало раунда {self.round_no}.")
                continue

            if not self._apply_start_of_turn_effects(nxt):
                if self.winner is not None:
                    return False
                continue

            if nxt["team"] == "blue":
                self.current_blue_attacker_pos = nxt["position"]
                self.blue_attacks_left = 2 if (nxt.get("Type") == "Demon") else 1
                self._log(
                    f"Ход BLUE: {nxt['Name']}#{nxt['position']} (иниц {nxt['Initiative']}). "
                    + ("Ожидание действия (демон: 2 удара)." if self.blue_attacks_left == 2 else "Ожидание действия.")
                )
                return True

            # RED plays automatically
            nxt["Initiative"] = 0
            strikes = 2 if nxt.get("Type") == "Demon" else 1

            for hit_i in range(strikes):
                if self.winner is not None:
                    return False

                target_pos = None
                if nxt.get("Type") in ("Воин", "Demon", "lord"):
                    options = self._warrior_allowed_targets(nxt)
                    if options:
                        target_pos = self._pick_lowest_hp(options)
                        prefix = nxt.get("Type")
                        self._log(
                            f"RED ход: {nxt['Name']}#{nxt['position']} ({prefix}) → цель с мин. HP pos{target_pos} (удар {hit_i+1}/{strikes})."
                        )
                    else:
                        self._log(
                            f"RED ход: {nxt['Name']}#{nxt['position']} ({nxt.get('Type')}) не может атаковать: доступных целей нет."
                        )
                elif nxt.get("Type") in ("Mage", "Dead dragon"):
                    if hit_i == 0:
                        self._log(
                            f"RED ход: {nxt['Name']}#{nxt['position']} ({nxt.get('Type')}) выполняет массовую атаку."
                        )
                    else:
                        break
                else:
                    live_blue_positions = self._live_positions_of("blue")
                    target_pos = self._pick_lowest_hp(live_blue_positions) if live_blue_positions else None
                    ut = nxt.get("Type")
                    self._log(
                        f"RED ход: {nxt['Name']}#{nxt['position']} ({ut}) → цель с мин. HP pos{target_pos} (удар {hit_i+1}/{strikes})."
                    )

                if target_pos is not None or nxt.get("Type") in ("Mage", "Dead dragon"):
                    self._attack(nxt, target_pos)
                    self._check_victory_after_hit()
                    if self.winner is not None:
                        return False
        return False

    # ------------------ Observation API ------------------
    def _obs(self) -> np.ndarray:
        vec: List[float] = []
        for pos in RED_POSITIONS + BLUE_POSITIONS:
            u = self._unit_by_position(pos)
            hp      = float(max(0, u["Health"]))
            ini     = float(max(0, u["Initiative"]))
            ini_b   = float(u.get("BaseInitiative", 0))
            dmg     = float(u["Damage"])
            dmg2    = float(u.get("Damage2", 0))
            team_v  = 0.0 if u["team"] == "red" else 1.0
            pos_v   = float(u["position"])
            stand_v = 0.0 if u["stand"] == "ahead" else 1.0

            t_onehot   = _one_hot(u.get("Type","Archer"), TYPE_LIST)
            imm_mhot   = _multi_hot(u.get("Immunity", []), ATTACK_TYPES)
            atk1_oh    = _one_hot(u.get("AttackType1",""), ATTACK_TYPES) if u.get("AttackType1","") in ATTACK_TYPES else [0.0]*len(ATTACK_TYPES)
            atk2_oh    = _one_hot(u.get("AttackType2",""), ATTACK_TYPES) if u.get("AttackType2","") in ATTACK_TYPES else [0.0]*len(ATTACK_TYPES)
            res_mhot   = _multi_hot(u.get("Resilience", []), ATTACK_TYPES)
            armor_v    = float(u.get("Armor", 0))
            acc_v      = float(u.get("Accuracy", 100))
            acc2_v     = float(u.get("Accuracy2", 0))
            poison_v   = float(u.get("poison_turns_left", 0))
            burn_v     = float(u.get("burn_turns_left", 0))

            vec.extend([
                hp, ini, ini_b, dmg, dmg2, team_v, pos_v, stand_v,
                *t_onehot, *imm_mhot, *atk1_oh, *atk2_oh, *res_mhot,
                armor_v, acc_v, acc2_v, poison_v, burn_v,
            ])
        return np.array(vec, dtype=np.float32)

    # ------------------ Gymnasium API ------------------
    def reset(self, *, seed=None, options=None):
        self._reset_state()
        self._advance_until_blue_turn()
        return self._obs(), {}

    def step(self, action):
        assert self.winner is None, "Эпизод завершён — вызовите reset()."

        if self.current_blue_attacker_pos is None:
            self._advance_until_blue_turn()
            if self.winner is not None:
                base = self.reward_win if self.winner == "blue" else self.reward_loss
                return self._obs(), base, True, False, {}

        step_shaping = 0.0
        target_pos = TARGET_POSITIONS[int(action)]

        attacker = self._unit_by_position(self.current_blue_attacker_pos)
        can_strike = (
            attacker is not None
            and self._alive(attacker)
            and (attacker["Initiative"] > 0 or (attacker.get("Type") == "Demon" and self.blue_attacks_left > 0))
        )
        if can_strike:
            self._log(f"BLUE действие: {attacker['Name']}#{attacker['position']} → pos{target_pos}")
            attacker["Initiative"] = 0

            if attacker.get("Type") in ("Воин", "Demon", "lord"):
                allowed = self._warrior_allowed_targets(attacker)
                if target_pos not in allowed:
                    self._log(
                        f"BLUE {attacker['Name']}#{attacker['position']} ({attacker.get('Type')}) не может достать pos{target_pos}. "
                        f"Доступные цели: {(', '.join('pos'+str(p) for p in allowed)) if allowed else 'нет'}"
                    )
                    step_shaping += self.penalty_unreachable_warrior
                else:
                    hit, reason = self._attack(attacker, target_pos)
                    if not hit and reason == "dead_or_absent":
                        step_shaping += self.penalty_invalid_target
                    self._check_victory_after_hit()
            elif attacker.get("Type") in ("Mage", "Dead dragon"):
                hit, reason = self._attack(attacker, target_pos)
                self._check_victory_after_hit()
            else:
                hit, reason = self._attack(attacker, target_pos)
                if not hit and reason == "dead_or_absent":
                    step_shaping += self.penalty_invalid_target
                self._check_victory_after_hit()

        if self.winner is None:
            if self.blue_attacks_left > 0:
                self.blue_attacks_left -= 1
            if self.blue_attacks_left > 0:
                reward, terminated = self.reward_step + step_shaping, False
                return self._obs(), reward, terminated, False, {}
            self.current_blue_attacker_pos = None
            self._advance_until_blue_turn()

        if self.winner is None:
            base = self.reward_step
            terminated = False
        else:
            base = self.reward_win if self.winner == "blue" else self.reward_loss
            terminated = True

        reward = base + step_shaping
        return self._obs(), reward, terminated, False, {}


__all__ = [
    "BattleEnv",
    "TYPE_LIST",
    "ATTACK_TYPES",
    "UNITS_RED",
    "UNITS_BLUE",
    "TARGET_POSITIONS",
]


def _pos_row_label(pos: int) -> str:
    if pos in (1, 2, 3, 7, 8, 9):
        return "передний ряд"
    return "задний ряд"


def _pos_col_label(pos: int) -> str:
    col = (pos - 1) % 3
    return {0: "левый", 1: "центр", 2: "правый"}.get(col, str(col))


def print_action_space_info():
    print("\n=== Доступные действия агента (Discrete(6)) ===")
    print("Действие i соответствует выбору цели posX среди врагов (RED: pos1..pos6).")
    for i, pos in enumerate(TARGET_POSITIONS):
        print(
            f"  {i}: pos{pos} — RED, {_pos_row_label(pos)}, столбец {_pos_col_label(pos)}"
        )
    print("Примечания:")
    print("- 'Воин'/'Demon'/'lord': ближний бой — цель должна быть достижима по правилам колонок и рядов.")
    print("- 'Mage'/'Dead dragon': массовая атака — выбор posX игнорируется, бьёт по всем целям противника.")
    print(
        "- Если цель недоступна/мертва: действие считается неэффективным и может давать штраф shaping."
    )


def _build_slot_feature_names() -> List[str]:
    base = [
        "hp",
        "ini",
        "ini_base",
        "dmg",
        "dmg2",
        "team",  # 0=red, 1=blue
        "pos",
        "stand",  # 0=ahead, 1=behind
    ]
    type_oh = [f"type[{t}]" for t in TYPE_LIST]
    imm = [f"imm[{a}]" for a in ATTACK_TYPES]
    atk1 = [f"atk1[{a}]" for a in ATTACK_TYPES]
    atk2 = [f"atk2[{a}]" for a in ATTACK_TYPES]
    res = [f"res[{a}]" for a in ATTACK_TYPES]
    tail = ["armor", "accuracy", "accuracy2", "poison_left", "burn_left"]
    names = base + type_oh + imm + atk1 + atk2 + res + tail
    return names


def build_all_feature_names() -> List[str]:
    per_slot = _build_slot_feature_names()
    all_names: List[str] = []
    for pos in RED_POSITIONS + BLUE_POSITIONS:
        pref = f"pos{pos}."
        all_names.extend([pref + n for n in per_slot])
    return all_names


def print_observation_space_info(env: BattleEnv, obs: np.ndarray) -> None:
    names = build_all_feature_names()
    print("\n=== Observation space ===")
    print(f"Space: {env.observation_space}")
    print("Структура: 12 позиций × 45 признаков = 540 значений.")
    print("Индексация признаков (i → имя):")
    for i, nm in enumerate(names):
        print(f"  {i:3d}: {nm}")
    # Полные границы пространства
    np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
    try:
        print("\nГраницы low:")
        print(env.observation_space.low)
        print("Границы high:")
        print(env.observation_space.high)
    except Exception:
        pass
    # Текущее наблюдение
    print("\nТекущее наблюдение obs (после reset):")
    print(obs)


def _describe_unit(u: Dict) -> str:
    alive = u["Health"] > 0
    hp_str = f"{max(0, u['Health'])}/{u.get('maxhealth', u['Health'])}"
    ini_str = f"{u.get('Initiative', 0)}/{u.get('BaseInitiative', 0)}"
    row = _pos_row_label(u["position"])  # передний/задний ряд
    col = _pos_col_label(u["position"])  # левый/центр/правый
    status_parts: List[str] = []
    if u.get("poison_turns_left", 0) > 0:
        status_parts.append(
            f"☠ poison {u.get('poison_damage_per_tick', 0)}/ход · {u['poison_turns_left']} хода"
        )
    if u.get("burn_turns_left", 0) > 0:
        status_parts.append(
            f"🔥 burn {u.get('burn_damage_per_tick', 0)}/ход · {u['burn_turns_left']} хода"
        )
    status = ("; ".join(status_parts)) if status_parts else "—"
    type_tag = u.get("Type", "")
    atk1 = u.get("AttackType1", "")
    atk2 = u.get("AttackType2", "")
    acc = int(u.get("Accuracy", 0) or 0)
    acc2 = int(u.get("Accuracy2", 0) or 0)
    arm = int(u.get("Armor", 0) or 0)
    big = ", BIG" if u.get("big") else ""
    life = "ALIVE" if alive else "DEAD"
    return (
        f"pos{u['position']:>2} {u['team'].upper():>4} {u['Name']} ({type_tag}{big}) | "
        f"HP {hp_str} | INI {ini_str} | ARM {arm} | ATK {atk1}{('/'+atk2) if atk2 else ''} | "
        f"ACC {acc}/{acc2} | {row}, {col} | {life} | статус: {status}"
    )


def render_battle_state(env: BattleEnv) -> None:
    print("RED команда:")
    for pos in RED_POSITIONS:
        u = env._unit_by_position(pos)
        if u is not None:
            print("  " + _describe_unit(u))
    print("BLUE команда:")
    for pos in BLUE_POSITIONS:
        u = env._unit_by_position(pos)
        if u is not None:
            print("  " + _describe_unit(u))


def choose_action_min_hp(env: BattleEnv) -> int:
    attacker_pos = env.current_blue_attacker_pos
    if attacker_pos is None:
        return 0
    attacker = env._unit_by_position(attacker_pos)
    if attacker is None:
        return 0
    if attacker.get("Type") in ("Воин", "Demon", "lord"):
        allowed = env._warrior_allowed_targets(attacker)
    else:
        allowed = env._live_positions_of("red")
    if not allowed:
        allowed = env._live_positions_of("red")
    if not allowed:
        return 0
    # Choose min HP among allowed
    best_pos = None
    best_hp = None
    for p in allowed:
        uu = env._unit_by_position(p)
        if uu is None or uu["Health"] <= 0:
            continue
        if best_hp is None or uu["Health"] < best_hp:
            best_hp = uu["Health"]
            best_pos = p
    if best_pos is None:
        best_pos = allowed[0]
    try:
        return TARGET_POSITIONS.index(best_pos)
    except ValueError:
        return 0


def run_demo(num_actions: int, seed: int = 42) -> None:
    env = BattleEnv(log_enabled=True)
    env.seed(seed)
    env.reset()
    actions_done = 0
    while actions_done < num_actions and env.winner is None:
        act = choose_action_min_hp(env)
        _, _, terminated, _, _ = env.step(act)
        actions_done += 1
        # Print turn log and current state
        events = env.pop_pretty_events()
        if events:
            print("\n— События после действия", actions_done, "—")
            for line in events:
                print(line)
        print(f"\nСостояние после {actions_done} действия(й):")
        render_battle_state(env)
        if terminated:
            break


if __name__ == "__main__":
    # Демонстрация среды: вывод состояния после 1-го и 5-го действия агента
    print("=== Демонстрация: после 1 действия агента ===")
    run_demo(num_actions=1, seed=42)
    print("\n=== Демонстрация: после 5 действий агента ===")
    run_demo(num_actions=5, seed=42)

