"""Adapters around the existing AutoControl -> BotState.handle seam.

No OpenCV, NumPy, Win32 or input-driver imports here; replay/unit tests are headless.
The native detector actually returns (label, detections) pairs, not MobInfo objects.
"""
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from .core import FSMPolicy, JevPolicy


def stop_command():
    return "IDLE", {"command": "RELEASE_ALL"}


def point(value):
    try:
        if len(value) != 2 or any(isinstance(x, bool) for x in value):
            return None
        values = tuple(float(x) for x in value)
        return values if all(math.isfinite(x) for x in values) else None
    except (TypeError, ValueError, OverflowError):
        return None


def percentage(value):
    try:
        number = float(value)
        return number if not isinstance(value, bool) and math.isfinite(number) and 0 <= number <= 100 else None
    except (TypeError, ValueError, OverflowError):
        return None


def box(value):
    try:
        a, b = point(value["t_l"]), point(value["b_r"])
        if a is not None and b is not None and a[0] <= b[0] and a[1] <= b[1]:
            return {"t_l": a, "b_r": b}
    except (KeyError, TypeError):
        pass
    return None


def contains(rect, pos):
    return (rect["t_l"][0] <= pos[0] <= rect["b_r"][0]
            and rect["t_l"][1] <= pos[1] <= rect["b_r"][1])


def geometry(owner):
    # Preserve original indices. Invalid/oversize maps fail closed, never silently reindex.
    result = []
    for name in ("platforms", "vertical_passage"):
        items = getattr(owner, name, None)
        if not isinstance(items, list) or len(items) > 128:
            raise ValueError("invalid map geometry")
        boxes = [box(item) for item in items]
        if any(item is None for item in boxes):
            raise ValueError("invalid map geometry")
        result.append(boxes)
    if not result[0]:
        raise ValueError("map has no platforms")
    return result


class ObservationBuilder:
    def __init__(self, *, clock=time.monotonic, capabilities=None):
        self.clock = clock
        self.capabilities = capabilities or {"patrol": True, "combat": True, "stuck": True}
        self.reset()

    def reset(self):
        self.last_position = None
        self.motion_since = self.clock()
        self.empty_since = self.clock()
        self.last_action = None
        self.stuck_latched = False

    def build(self, state, owner):
        now = self.clock()
        pos = point(state.mini_player_loc)
        screen = point(state.player_center_loc)
        if pos is None or screen is None:
            raise ValueError("tracking missing")
        platforms, passages = geometry(owner)
        platform = next((i for i, rect in enumerate(platforms) if contains(rect, pos)), None)
        passage = next((i for i, rect in enumerate(passages) if contains(rect, pos)), None)
        reachable = []
        if platform is not None:
            rect = platforms[platform]
            for i, rope in enumerate(passages):
                if (rope["b_r"][0] >= rect["t_l"][0] and rope["t_l"][0] <= rect["b_r"][0]
                        and any(rect["t_l"][1] <= y <= rect["b_r"][1]
                                for y in (rope["t_l"][1], rope["b_r"][1]))):
                    reachable.append(i)
        # Never serialize a frame, player name, arbitrary object repr, or credentials.
        mobs = []
        roi = getattr(state, "roi_BBOX", None)
        for _, details in (state.mobs or []):
            for detail in details:
                loc = point(detail.get("top_left"))
                if loc is None or roi is None:
                    raise ValueError("invalid mob coordinates")
                mx, my = loc[0] + float(roi.x1), loc[1] + float(roi.y1)
                if not math.isfinite(mx) or not math.isfinite(my):
                    raise ValueError("invalid mob coordinates")
                mobs.append({"screen_xy": (mx, my), "horizontal_distance": abs(mx - screen[0])})
        mobs.sort(key=lambda mob: mob["horizontal_distance"])
        moving = self.last_action in {"MOVE", "JUMP", "ROPE", "CLIMB", "SMALL_MOVE", "JUMP_GRAB"}
        progressed = self.last_position is None or math.dist(pos, self.last_position) >= 1
        if progressed:
            self.stuck_latched = False
        if not moving or progressed:
            self.motion_since = now
        self.last_position = pos
        stuck_for = max(0, now - self.motion_since) if moving else 0
        self.stuck_latched = self.stuck_latched or stuck_for >= 1.5
        if self.stuck_latched:
            stuck_for = max(stuck_for, 1.5)
        if mobs:
            self.empty_since = now
        choices = ["IDLE"]
        if platform is not None and self.capabilities.get("patrol"):
            choices.append("PATROL")
        if mobs and self.capabilities.get("combat"):
            choices.append("COMBAT")
        if passage is not None or reachable:
            choices.append("NAVIGATE")
        if stuck_for >= 1.5 and self.capabilities.get("stuck"):
            choices.append("RECOVER_STUCK")
        fingerprint = hashlib.sha256(json.dumps([platforms, passages], sort_keys=True).encode()).hexdigest()[:16]
        context = (fingerprint, platform, passage, bool(mobs), stuck_for >= 1.5, tuple(choices))
        snapshot = {
            "schema_version": 1, "map_fingerprint": fingerprint,
            "player": {"minimap_xy": pos, "screen_xy": screen,
                       "hp_percent": percentage(state.player_hp), "mp_percent": percentage(state.player_mp)},
            "platform_id": platform, "passage_id": passage,
            "platforms": platforms, "passages": passages, "reachable_passages": reachable,
            "mobs": mobs[:32], "mob_count": len(mobs),
            "attack_range_screen_px": float(owner.player_attack_range),
            "previous_motor_action": self.last_action,
            "stuck_for_seconds": round(stuck_for, 3),
            "no_mobs_for_seconds": round(max(0, now - self.empty_since), 3),
        }
        if not math.isfinite(snapshot["attack_range_screen_px"]) or snapshot["attack_range_screen_px"] <= 0:
            raise ValueError("invalid attack range")
        return snapshot, choices, context


class JsonlLog:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")
        self.session_id = uuid.uuid4().hex

    def __call__(self, event):
        self.file.write(json.dumps({"session_id": self.session_id, "monotonic": time.monotonic(), **event}, ensure_ascii=False, allow_nan=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class PolicyBridge:
    """Replaces only bot_state.handle; all skills execute on the game-loop thread."""
    def __init__(self, owner, mode, *, client=None, settings=None, clock=time.monotonic,
                 submit=None, emit=None, capabilities=None):
        if mode not in {"fsm", "shadow", "jev"}:
            raise ValueError("policy must be fsm, shadow or jev")
        self.owner, self.mode = owner, mode
        self.emit = emit or (lambda event: None)
        self.fsm = FSMPolicy(owner.bot_state)
        self.builder = ObservationBuilder(clock=clock, capabilities=capabilities)
        self.jev = None
        self._baseline_for_log = None
        self._last_shadow_id = None
        if mode != "fsm":
            if client is None:
                raise ValueError("Jev client required")
            self.jev = JevPolicy(client, settings, clock=clock, submit=submit, emit=self._emit_jev)
        self.last_tactic = None

    def _emit_jev(self, event):
        if event.get("event") == "jev_request" and self.mode == "shadow":
            event = {**event, "baseline_motor_command": self._baseline_for_log}
        self.emit(event)

    def reset(self):
        self.builder.reset()
        self.last_tactic = None
        if self.jev:
            self.jev.reset()
        # Only clear Jev's motor continuation. Do not mutate the shadow FSM.
        if self.mode == "jev":
            self.owner.current_verti_target = None
            self.owner._verti_movement_timer = None

    def close(self):
        if self.jev:
            self.jev.close()

    def handle(self, state):
        if self.mode == "fsm":
            result = self.fsm.decide(state)
        else:
            snapshot, choices, context = self.builder.build(state, self.owner)
            # Snapshot is collected before the baseline can mutate its own state.
            baseline = self.fsm.decide(state) if self.mode == "shadow" else None
            self._baseline_for_log = baseline
            decision = self.jev.decide(snapshot, choices, context)
            if self.mode == "shadow":
                # Jev's result NEVER calls a skill or changes baseline state.
                result = baseline
                if decision is not None and self.jev.last_observation_id != self._last_shadow_id:
                    self._last_shadow_id = self.jev.last_observation_id
                    self.emit({"event": "shadow_observation", "request_id": self._last_shadow_id, "jev_tactic": decision.choice,
                               "baseline_motor_command": baseline,
                               "note": "different abstractions; agreement is not accuracy"})
            elif decision is None:
                result = stop_command()
            else:
                result = self._execute(decision.choice, state)
        if result is None:
            result = (None, None)
        self.builder.last_action = result[0]
        return result

    def _execute(self, tactic, state):
        owner = self.owner
        owner.current_platform = owner._check_current_platform()
        if tactic != self.last_tactic:
            owner.current_verti_target = None
            owner._verti_movement_timer = None
        self.last_tactic = tactic
        if tactic == "PATROL":
            result = owner._enable_player_patrol()
        elif tactic == "COMBAT":
            result = owner._fk_that_mob(state)
        elif tactic == "RECOVER_STUCK":
            result = owner._unstuck_player()
        elif tactic == "NAVIGATE":
            passage = owner._check_vertical_passage()
            if passage is not None:
                result = owner._verti_movement(passage)
            else:
                target = owner._find_nearest_verti_passage()
                result = owner._move_to_verti_passage(target) if target is not None else stop_command()
        else:
            result = stop_command()
        return result if result and result[0] is not None else stop_command()


class AutoControlAdapter:
    """Guard before AutoControl's health handling and cached-position reuse."""
    def __init__(self, owner, bridge, *, allowed=lambda: True, halt=lambda: None):
        self.owner, self.bridge = owner, bridge
        self.allowed, self.halt = allowed, halt
        owner.bot_state = bridge

    def get_debug_geometry(self):
        return self.owner.get_debug_geometry()

    def suspend(self):
        self.bridge.reset()
        self.halt()

    def run(self, state):
        if (not self.allowed() or point(state.mini_player_loc) is None
                or point(state.player_center_loc) is None or percentage(state.player_hp) == 0):
            self.suspend()
            return stop_command()
        try:
            result = self.owner.run(state)  # original health priority is retained
            if result and result[0] and result[0].startswith("HEAL") and self.bridge.jev:
                self.bridge.jev.reset()
            return result
        except Exception as exc:
            self.suspend()
            self.bridge.emit({"event": "controller_error", "type": type(exc).__name__})
            return stop_command()
