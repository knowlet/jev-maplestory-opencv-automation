import copy
from concurrent.futures import Future
import io
import json
import math
import threading
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import patch

from src.policy.client import (CRITERIA, ENDPOINT, Decision, JevClient, JevError,
                               NoRedirect, Settings, parse_response)
from src.policy.core import FSMPolicy, JevPolicy
from src.policy.integration import (AutoControlAdapter, ObservationBuilder, PolicyBridge,
                                    percentage, point, stop_command)
from src.policy.input_guard import HEAL_HOLD_SECONDS, GuardedActionHandler, KeyboardLeaseMixin

CHOICES = ("IDLE", "PATROL")


class Clock:
    def __init__(self):
        self.now = 10.0
    def __call__(self):
        return self.now
    def advance(self, dt):
        self.now += dt


class Submit:
    def __init__(self):
        self.jobs = []
    def __call__(self, fn, *args):
        future = Future()
        self.jobs.append((future, fn, args))
        return future
    def complete(self, choice="PATROL", confidence=0.9):
        self.jobs[-1][0].set_result(Decision(choice, confidence))


def payload(choice="PATROL", confidence=0.9):
    return {"model": "jev-test", "answers": {"tactic": {
        "type": "choice", "choice": choice, "confidence": confidence,
        "probabilities": {"IDLE": 0.1, "PATROL": 0.9},
    }}, "usage": {"input_tokens": 123}}


class Reply(io.BytesIO):
    status = 200


class ClientTests(unittest.TestCase):
    def test_documented_http_contract(self):
        calls = []
        def transport(req, timeout):
            calls.append((req, timeout))
            return Reply(json.dumps(payload()).encode())
        client = JevClient(Settings(), api_key="test-secret", transport=transport)
        answer = client.decide({"player": {"hp_percent": 50}}, CHOICES)
        req, timeout = calls[0]
        body = json.loads(req.data)
        self.assertEqual(req.full_url, ENDPOINT)
        self.assertEqual(req.method, "POST")
        self.assertEqual(req.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(body["questions"]["tactic"]["type"], "choice")
        self.assertEqual(set(body["questions"]["tactic"]["criteria"]), set(CHOICES))
        self.assertNotIn("messages", body)
        self.assertNotIn("test-secret", req.data.decode())
        self.assertEqual(answer.choice, "PATROL")
        self.assertEqual(answer.input_tokens, 123)
        self.assertEqual(timeout, 1.0)

    def test_missing_key_fails_before_network(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                JevClient()

    def test_auth_error_is_sanitized_and_not_retried(self):
        calls = []
        def transport(*args, **kwargs):
            calls.append(1)
            raise urllib.error.HTTPError(ENDPOINT, 401, "test-secret", {}, None)
        with self.assertRaisesRegex(JevError, "^http_401$"):
            JevClient(api_key="test-secret", transport=transport).decide({}, CHOICES)
        self.assertEqual(len(calls), 1)

    def test_redirect_rejected(self):
        with self.assertRaisesRegex(JevError, "redirect_rejected"):
            NoRedirect().redirect_request(None, None, 302, "", {}, "https://untrusted.invalid")

    def test_invalid_response_variants(self):
        variants = []
        for key, value in [("choice", "DELETE_FILES"), ("type", "score"), ("confidence", True),
                           ("confidence", float("nan")), ("confidence", 1.1)]:
            data = payload()
            data["answers"]["tactic"][key] = value
            variants.append(data)
        for probs in [{"IDLE": 0.1}, {"IDLE": 0.1, "PATROL": 0.9, "EXTRA": 0},
                      {"IDLE": 0.1, "PATROL": 0.1}, {"IDLE": True, "PATROL": 0},
                      {"IDLE": -0.1, "PATROL": 1.1}, {"IDLE": 0.1, "PATROL": float("inf")},
                      {"IDLE": 0.9, "PATROL": 0.1}]:
            data = payload()
            data["answers"]["tactic"]["probabilities"] = probs
            variants.append(data)
        variants.extend([{}, {"answers": []}, None])
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(JevError):
                parse_response(variant, CHOICES)

    def test_invalid_json_and_oversized_body(self):
        for raw in (b"not-json", b"x" * 256_001):
            with self.subTest(size=len(raw)), self.assertRaises(JevError):
                JevClient(api_key="x", transport=lambda *a, **k: Reply(raw)).decide({}, CHOICES)

    def test_probability_is_not_confidence(self):
        data = payload(confidence=0.2)
        decision = parse_response(data, CHOICES)
        self.assertEqual(decision.probabilities[decision.choice], 0.9)
        self.assertEqual(decision.confidence, 0.2)

    def test_invalid_settings(self):
        for kwargs in ({"interval": 0.01}, {"interval": float("nan")}, {"timeout": 0},
                       {"timeout": 2}, {"max_age": 1}, {"input_lease": -1},
                       {"min_confidence": float("inf")}, {"min_confidence": -0.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Settings(**kwargs)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.submit, self.events = Clock(), Submit(), []
        self.policy = JevPolicy(SimpleNamespace(decide=lambda *args: None),
                                clock=self.clock, submit=self.submit, emit=self.events.append)
    def tick(self, choices=CHOICES, context=("map", 0)):
        return self.policy.decide({"x": 1}, choices, context)

    def test_single_inflight_nonblocking(self):
        for _ in range(100):
            self.assertIsNone(self.tick())
        self.assertEqual(len(self.submit.jobs), 1)

    def test_accept_and_cache(self):
        self.tick()
        self.clock.advance(0.1)
        self.submit.complete()
        self.assertEqual(self.tick().choice, "PATROL")
        self.assertEqual(self.tick().choice, "PATROL")
        self.assertEqual(len(self.submit.jobs), 1)

    def test_low_confidence_stops(self):
        self.tick()
        self.submit.complete(confidence=0.1)
        self.assertIsNone(self.tick())
        self.assertEqual(self.policy.last_status, "context_or_confidence_rejected")

    def test_reset_discards_late_response(self):
        self.tick()
        self.policy.reset()
        self.submit.complete()
        self.assertIsNone(self.tick())

    def test_pre_pause_inflight_timeout_does_not_backoff(self):
        self.tick()
        self.policy.reset()
        self.clock.advance(5)
        self.assertIsNone(self.tick())
        self.assertEqual(self.policy.errors, 0)
        self.assertNotEqual(self.policy.last_status, "deadline")
        self.assertLess(self.policy.next_request, self.clock.now)
        self.submit.complete()
        self.assertIsNone(self.tick())
        self.assertEqual(self.policy.errors, 0)
        self.assertEqual(len(self.submit.jobs), 2)

    def test_timeout_does_not_spawn_unbounded_workers(self):
        self.tick()
        self.clock.advance(1.1)
        for _ in range(50):
            self.assertIsNone(self.tick())
        self.clock.advance(50)
        self.assertIsNone(self.tick())
        self.assertEqual(len(self.submit.jobs), 1)
        self.submit.complete()
        self.assertIsNone(self.tick())
        self.assertEqual(len(self.submit.jobs), 2)

    def test_completed_but_late_is_rejected(self):
        self.tick()
        self.submit.complete()
        self.clock.advance(1.1)
        self.assertIsNone(self.tick())
        self.assertEqual(self.policy.last_status, "deadline")

    def test_current_choice_mask_revalidated(self):
        self.tick()
        self.submit.complete()
        self.assertIsNone(self.tick(choices=("IDLE",)))

    def test_scene_change_rejects_response(self):
        self.tick()
        self.submit.complete()
        self.assertIsNone(self.tick(context=("different-map", 0)))

    def test_cache_age_measured_from_observation(self):
        self.tick()
        self.clock.advance(0.8)
        self.submit.complete()
        self.assertIsNotNone(self.tick())
        self.clock.advance(0.8)
        self.assertIsNone(self.tick())

    def test_unknown_error_is_redacted_and_backed_off(self):
        self.tick()
        self.submit.jobs[-1][0].set_exception(RuntimeError("secret-value"))
        self.assertIsNone(self.tick())
        self.clock.advance(0.6)
        self.tick()
        self.assertEqual(len(self.submit.jobs), 1)
        self.assertNotIn("secret-value", repr(self.events))

    def test_close_blocks_requests_and_cached_commands(self):
        self.tick()
        self.submit.complete()
        self.policy.close()
        self.clock.advance(2)
        self.assertIsNone(self.tick())
        self.assertEqual(len(self.submit.jobs), 1)

    def test_only_idle_does_not_call_api(self):
        self.assertIsNone(self.tick(choices=("IDLE",)))
        self.assertEqual(len(self.submit.jobs), 0)


class Machine:
    def __init__(self):
        self.calls = 0
    def handle(self, state):
        self.calls += 1
        return "MOVE", {"direction": "LEFT"}


class Owner:
    def __init__(self):
        self.bot_state = Machine()
        self.platforms = [{"t_l": (0, 0), "b_r": (100, 10)}]
        self.vertical_passage = [{"t_l": (50, 0), "b_r": (55, 60)}]
        self.player_attack_range = 50
        self.current_platform = 0
        self.current_verti_target = "UP"
        self._verti_movement_timer = 20
        self.mini_player_loc = (20, 5)
        self.skill_calls = []
        self.health = None
        self.run_calls = 0
    def run(self, state):
        self.run_calls += 1
        self.mini_player_loc = state.mini_player_loc
        return self.health or self.bot_state.handle(state)
    def _check_current_platform(self):
        return 0
    def _enable_player_patrol(self):
        self.skill_calls.append("PATROL")
        return "MOVE", {"direction": "RIGHT"}
    def _fk_that_mob(self, state):
        self.skill_calls.append("COMBAT")
        return "ATTACK", {"direction": "RIGHT"}
    def _unstuck_player(self):
        self.skill_calls.append("RECOVER_STUCK")
        return "JUMP", {"direction": "LEFT"}
    def _check_vertical_passage(self):
        return None
    def _find_nearest_verti_passage(self):
        return 0
    def _move_to_verti_passage(self, target):
        self.skill_calls.append(("NAVIGATE", target))
        return "MOVE", {"direction": "RIGHT"}


def state():
    return SimpleNamespace(mini_player_loc=(20, 5), player_center_loc=(100, 100),
                           player_hp=75, player_mp=50, roi_BBOX=SimpleNamespace(x1=90, y1=80),
                           mobs=[("monster-template", [{"top_left": (20, 20)}])])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.owner, self.state = Clock(), Owner(), state()
        self.submit = Submit()
    def bridge(self, mode):
        return PolicyBridge(self.owner, mode, client=SimpleNamespace(decide=lambda *a: None),
                            clock=self.clock, submit=self.submit)
    def test_native_pair_shape_and_roi_offset(self):
        snapshot, choices, _ = ObservationBuilder(clock=self.clock).build(self.state, self.owner)
        self.assertEqual(snapshot["mobs"][0]["screen_xy"], (110, 100))
        self.assertIn("COMBAT", choices)
        self.assertNotIn("monster-template", json.dumps(snapshot))
        self.assertIn(0, snapshot["reachable_passages"])
    def test_zero_index_is_valid_platform(self):
        snapshot, choices, _ = ObservationBuilder(clock=self.clock).build(self.state, self.owner)
        self.assertEqual(snapshot["platform_id"], 0)
        self.assertIn("PATROL", choices)
    def test_empty_nested_detections_do_not_enable_combat(self):
        self.state.mobs = [("empty", [])]
        _, choices, _ = ObservationBuilder().build(self.state, self.owner)
        self.assertNotIn("COMBAT", choices)
    def test_unknown_health_is_not_dead(self):
        self.assertIsNone(percentage(None))
        self.state.player_hp = None
        bridge = self.bridge("fsm")
        adapter = AutoControlAdapter(self.owner, bridge)
        self.assertEqual(adapter.run(self.state)[0], "MOVE")
    def test_dead_or_missing_tracking_never_enters_owner(self):
        adapter = AutoControlAdapter(self.owner, self.bridge("fsm"))
        for missing in ("mini_player_loc", "player_center_loc"):
            broken = copy.deepcopy(self.state)
            setattr(broken, missing, None)
            self.assertEqual(adapter.run(broken), stop_command())
        self.state.player_hp = 0
        self.assertEqual(adapter.run(self.state), stop_command())
        self.assertEqual(self.owner.run_calls, 0)
    def test_fsm_adapter_calls_baseline_once(self):
        original = self.owner.bot_state
        bridge = self.bridge("fsm")
        self.assertEqual(bridge.handle(self.state), ("MOVE", {"direction": "LEFT"}))
        self.assertEqual(original.calls, 1)
        self.assertEqual(self.owner.skill_calls, [])
        self.assertEqual(len(self.submit.jobs), 0)
    def test_shadow_never_executes_jev_skills(self):
        original = self.owner.bot_state
        bridge = self.bridge("shadow")
        bridge.handle(self.state)
        self.submit.complete("COMBAT")
        result = bridge.handle(self.state)
        self.assertEqual(result, ("MOVE", {"direction": "LEFT"}))
        self.assertEqual(original.calls, 2)
        self.assertEqual(self.owner.skill_calls, [])
        self.assertEqual(self.owner.current_verti_target, "UP")
    def test_jev_executes_selected_existing_skill_not_baseline(self):
        original = self.owner.bot_state
        bridge = self.bridge("jev")
        self.assertEqual(bridge.handle(self.state), stop_command())
        self.submit.complete("COMBAT")
        self.assertEqual(bridge.handle(self.state)[0], "ATTACK")
        self.assertEqual(original.calls, 0)
        self.assertEqual(self.owner.skill_calls, ["COMBAT"])
    def test_no_hidden_fsm_fallback(self):
        original = self.owner.bot_state
        bridge = self.bridge("jev")
        bridge.handle(self.state)
        self.submit.complete("COMBAT", confidence=0.01)
        self.assertEqual(bridge.handle(self.state), stop_command())
        self.assertEqual(original.calls, 0)
    def test_healing_preempts_jev(self):
        bridge = self.bridge("jev")
        adapter = AutoControlAdapter(self.owner, bridge)
        self.owner.health = ("HEAL_LOW", {"key": "h"})
        self.assertEqual(adapter.run(self.state), self.owner.health)
        self.assertEqual(self.owner.skill_calls, [])
        self.assertEqual(len(self.submit.jobs), 0)
    def test_focus_guard_stops_before_policy(self):
        halts = []
        adapter = AutoControlAdapter(self.owner, self.bridge("jev"), allowed=lambda: False,
                                     halt=lambda: halts.append(1))
        self.assertEqual(adapter.run(self.state), stop_command())
        self.assertEqual(halts, [1])
        self.assertEqual(len(self.submit.jobs), 0)
    def test_stuck_remains_legal_while_waiting_for_jev(self):
        builder = ObservationBuilder(clock=self.clock)
        builder.last_action = "MOVE"
        builder.build(self.state, self.owner)
        self.clock.advance(1.6)
        _, choices, _ = builder.build(self.state, self.owner)
        self.assertIn("RECOVER_STUCK", choices)
        builder.last_action = "IDLE"  # pause while the request is in flight
        _, choices, _ = builder.build(self.state, self.owner)
        self.assertIn("RECOVER_STUCK", choices)
        self.state.mini_player_loc = (25, 5)
        _, choices, _ = builder.build(self.state, self.owner)
        self.assertNotIn("RECOVER_STUCK", choices)
    def test_invalid_geometry_rejected_without_reindexing(self):
        self.owner.platforms.insert(0, {"t_l": (0, 0), "b_r": (-1, 1)})
        with self.assertRaises(ValueError):
            ObservationBuilder().build(self.state, self.owner)
    def test_hp_hold_is_not_replaced_by_an_mp_heal(self):
        potions = [("HEAL_HP", {"key": "h"}), ("HEAL_MP", {"key": "m"})]

        def run(state):
            self.owner.run_calls += 1
            return potions[self.owner.run_calls - 1]

        self.owner.run = run
        adapter = AutoControlAdapter(self.owner, self.bridge("jev"))
        self.assertEqual(adapter.run(self.state), potions[0])
        started = self.owner.run_calls
        self.clock.advance(0.5)
        self.assertEqual(adapter.run(self.state), potions[0])
        self.assertEqual(self.owner.run_calls, started)
        self.clock.advance(HEAL_HOLD_SECONDS)
        self.assertEqual(adapter.run(self.state), potions[1])
        self.assertEqual(self.owner.run_calls, started + 1)

    def test_cut_short_heal_is_replayed_after_a_long_stall(self):
        def run(state):
            self.owner.run_calls += 1
            if self.owner.run_calls == 1:
                return ("HEAL_HP", {"key": "h"})
            return ("ATTACK", {"direction": "LEFT"})

        self.owner.run = run
        lost = {"at": None}

        def consume():
            at = lost["at"]
            lost["at"] = None
            return at

        adapter = AutoControlAdapter(self.owner, self.bridge("jev"), consume_focus_losses=consume)
        self.assertEqual(adapter.run(self.state), ("HEAL_HP", {"key": "h"}))
        self.clock.advance(0.4)
        lost["at"] = self.clock.now
        self.clock.advance(5)
        self.assertEqual(adapter.run(self.state), stop_command())
        self.assertEqual(self.owner.run_calls, 1)
        self.assertEqual(adapter.run(self.state), ("HEAL_HP", {"key": "h"}))
        self.assertEqual(self.owner.run_calls, 1)

    def test_focus_loss_between_frames_suspends_before_cached_decision(self):
        seen = {"n": 1}

        def consume():
            lost = seen["n"]
            seen["n"] = None
            return lost

        halts = []
        adapter = AutoControlAdapter(self.owner, self.bridge("jev"),
                                     halt=lambda: halts.append(1),
                                     consume_focus_losses=consume)
        self.owner.health = ("ATTACK", {"direction": "LEFT", "name": "slime", "distance": 4})
        self.assertEqual(adapter.run(self.state), stop_command())
        self.assertEqual(self.owner.run_calls, 0)
        self.assertEqual(halts, [1])
        self.assertEqual(adapter.run(self.state)[0], "ATTACK")

    def test_missing_action_releases_instead_of_leaving_keys_down(self):
        self.owner.bot_state = SimpleNamespace(handle=lambda state: (None, None))
        bridge = self.bridge("fsm")
        self.assertEqual(bridge.handle(self.state), stop_command())

    def test_nonfinite_coordinates_rejected(self):
        self.assertIsNone(point((float("nan"), 0)))
        self.assertIsNone(point((True, 0)))
        self.assertIsNone(point((0,)))

    def test_heal_hold_is_not_preempted_and_then_expires(self):
        heal = ("HEAL_LOW", {"key": "h"})
        released = []
        bridge = self.bridge("jev")
        bridge.release_inputs = lambda: released.append("release")
        adapter = AutoControlAdapter(self.owner, bridge, halt=lambda: released.append("halt"))
        self.owner.health = heal
        self.assertEqual(adapter.run(self.state), heal)
        self.assertEqual(self.owner.skill_calls, [])
        self.owner.health = None
        released.clear()
        self.clock.advance(1.0)
        # The next frame used to become IDLE and halt the potion key.
        self.assertEqual(adapter.run(self.state), heal)
        self.assertEqual(released, [])
        self.assertEqual(self.owner.skill_calls, [])
        self.clock.advance(HEAL_HOLD_SECONDS)
        result = adapter.run(self.state)
        self.assertFalse(str(result[0]).startswith("HEAL"))

    def test_heal_hold_suppresses_baseline_until_it_completes(self):
        machine = self.owner.bot_state
        bridge = self.bridge("fsm")
        adapter = AutoControlAdapter(self.owner, bridge)
        self.owner.health = ("HEAL_LOW", {"key": "h"})
        adapter.run(self.state)
        self.owner.health = None
        self.clock.advance(0.5)
        adapter.run(self.state)
        self.assertEqual(machine.calls, 0)
        self.clock.advance(HEAL_HOLD_SECONDS)
        adapter.run(self.state)
        self.assertEqual(machine.calls, 1)

    def test_safety_stop_retries_heal_and_death_does_not(self):
        heal = ("HEAL_LOW", {"key": "h"})
        allowed = {"ok": True}
        adapter = AutoControlAdapter(self.owner, self.bridge("jev"), allowed=lambda: allowed["ok"])
        self.owner.health = heal
        self.assertEqual(adapter.run(self.state), heal)
        self.owner.health = None
        self.clock.advance(0.2)
        allowed["ok"] = False
        self.assertEqual(adapter.run(self.state), stop_command())
        allowed["ok"] = True
        self.assertEqual(adapter.run(self.state), heal)
        self.assertEqual(self.owner.skill_calls, [])
        # Focus loss calls suspend() directly and must keep the same retry.
        adapter.suspend()
        self.assertEqual(adapter.run(self.state), heal)
        self.state.player_hp = 0
        self.assertEqual(adapter.run(self.state), stop_command())
        self.state.player_hp = 80
        result = adapter.run(self.state)
        self.assertFalse(str(result[0]).startswith("HEAL"))

    def test_tactic_switch_and_exit_release_held_keys(self):
        keyboard = Keyboard(allowed=lambda: True, clock=self.clock, watchdog=False)
        try:
            keyboard.arm()
            keyboard._key_down("left")
            bridge = self.bridge("jev")
            bridge.release_inputs = keyboard.halt
            bridge._execute("PATROL", self.state)
            self.assertEqual(keyboard._pressed_keys, set())
            self.assertEqual(self.owner.skill_calls, ["PATROL"])
            keyboard.arm()
            keyboard._key_down("right")
            bridge._execute("PATROL", self.state)
            self.assertEqual(keyboard._pressed_keys, {"right"})
            bridge._execute("COMBAT", self.state)
            self.assertEqual(keyboard._pressed_keys, set())
            self.assertEqual(self.owner.skill_calls, ["PATROL", "PATROL", "COMBAT"])
            keyboard.arm()
            keyboard._key_down("left")
            self.assertEqual(bridge._execute("IDLE", self.state), stop_command())
            self.assertEqual(keyboard._pressed_keys, set())
            self.assertIsNone(bridge.last_tactic)
            bridge._execute("NAVIGATE", self.state)
            keyboard.arm()
            keyboard._key_down("up")
            self.assertEqual(bridge.handle(self.state), stop_command())
            self.assertEqual(keyboard._pressed_keys, set())
            self.assertIsNone(bridge.last_tactic)
            bridge._execute("RECOVER_STUCK", self.state)
            keyboard.arm()
            keyboard._key_down("jump")
            bridge.reset()
            self.assertEqual(keyboard._pressed_keys, set())
            self.assertIsNone(bridge.last_tactic)
            held = len(keyboard.events)
            bridge.reset()
            self.assertEqual(len(keyboard.events), held)
        finally:
            keyboard.close()


class KeyboardBase:
    def __init__(self):
        self.events = []
        self._pressed_keys = set()
        self._pressed_keys_lock = threading.Lock()
        self._current_move = None
    def _key_down(self, key):
        self.events.append(("down", key))
        self._pressed_keys.add(key)
    def _key_up(self, key):
        self.events.append(("up", key))
        self._pressed_keys.discard(key)
    def _press_direction_locked(self, direction):
        self._key_down(direction)
        self._current_move = direction
    def _release_direction_locked(self):
        if self._current_move:
            self._key_up(self._current_move)
        self._current_move = None
    def _attack_command(self):
        self._key_down("attack")
    def jump_left_grab_command(self):
        self._key_down("jump")
    def _finish_command(self):
        self._key_up("attack")


class Keyboard(KeyboardLeaseMixin, KeyboardBase):
    pass


class InputTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.allowed = True
        self.kb = Keyboard(allowed=lambda: self.allowed, clock=self.clock, watchdog=False)
    def tearDown(self):
        self.kb.close()
    def test_unarmed_input_is_denied(self):
        self.kb._key_down("left")
        self.assertEqual(self.kb.events, [])
    def test_halt_synchronously_releases_tracked_keys(self):
        self.kb.arm()
        self.kb._key_down("left")
        self.kb.halt()
        self.assertEqual(self.kb.events, [("down", "left"), ("up", "left")])
    def test_expired_lease_denies_new_press(self):
        self.kb.arm()
        self.clock.advance(2)
        self.kb._key_down("left")
        self.assertEqual(self.kb.events, [])
    def test_focus_checked_at_actual_keydown(self):
        self.kb.arm()
        self.allowed = False
        self.kb._key_down("left")
        self.assertEqual(self.kb.events, [])
    def test_pre_pause_worker_cannot_press_after_resume(self):
        self.kb.arm()
        worker = self.kb._attack_command  # target bound before pause
        self.kb.halt()
        self.kb.arm()
        worker()
        self.assertEqual(self.kb.events, [])
    def test_public_named_worker_is_also_generation_bound(self):
        self.kb.arm()
        worker = self.kb.jump_left_grab_command
        self.kb.halt()
        self.kb.arm()
        worker()
        self.assertEqual(self.kb.events, [])
    def test_cancelled_worker_does_not_release_new_key(self):
        self.kb.arm()
        old_finish = self.kb._finish_command
        self.kb.halt()
        self.kb.arm()
        self.kb._key_down("attack")
        old_finish()
        self.assertEqual(self.kb._pressed_keys, {"attack"})
    def test_invalid_action_and_idle_halt(self):
        calls = []
        base = SimpleNamespace(execute_behavior=lambda *args: calls.append(args))
        handler = GuardedActionHandler(base, self.kb)
        handler.execute_behavior("MOVE", {"direction": "RIGHT"})
        self.assertEqual(len(calls), 1)
        self.kb._key_down("right")
        for action, params in [("MOVE", {"direction": "DIAGONAL"}), ([], {}), ("IDLE", {})]:
            handler.execute_behavior(action, params)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.kb._pressed_keys, set())
    def test_close_never_rearms(self):
        self.kb.close()
        self.kb.arm()
        self.kb._key_down("left")
        self.assertEqual(self.kb.events, [])

    def test_expired_lease_cannot_be_renewed(self):
        self.assertTrue(self.kb.arm())
        self.clock.advance(0.999)
        self.assertTrue(self.kb.arm())
        deadline = self.kb._lease_until
        self.clock.now = deadline
        self.assertFalse(self.kb.arm())
        self.assertEqual(self.kb._lease_until, deadline)
        self.kb._key_down("left")
        self.assertEqual(self.kb.events, [])
        self.kb.halt()
        self.assertFalse(self.kb.arm())
        self.assertTrue(self.kb.arm())
        self.kb._key_down("left")
        self.assertEqual(self.kb.events, [("down", "left")])

    def test_guard_does_not_dispatch_on_expired_lease(self):
        calls = []
        handler = GuardedActionHandler(
            SimpleNamespace(execute_behavior=lambda *args: calls.append(args)), self.kb)
        handler.execute_behavior("MOVE", {"direction": "LEFT"})
        self.kb._key_down("left")
        self.assertEqual(calls, [("MOVE", {"direction": "LEFT"})])
        self.clock.advance(1.0)
        calls.clear()
        handler.execute_behavior("MOVE", {"direction": "RIGHT"})
        self.assertEqual(calls, [])
        self.assertEqual(self.kb._pressed_keys, set())
        self.assertEqual(self.kb._lease_until, 0.0)
        handler.execute_behavior("MOVE", {"direction": "RIGHT"})
        self.assertEqual(calls, [("MOVE", {"direction": "RIGHT"})])

    def test_heal_lease_covers_hold_and_idle_does_not_release_it(self):
        calls = []
        handler = GuardedActionHandler(
            SimpleNamespace(execute_behavior=lambda *args: calls.append(args)), self.kb)
        handler.execute_behavior("HEAL_LOW", {"key": "h"})
        self.kb._key_down("h")
        self.assertGreaterEqual(self.kb._lease_until - self.clock.now, HEAL_HOLD_SECONDS)
        self.clock.advance(1.0)
        self.assertTrue(self.kb._permitted())
        handler.execute_behavior("IDLE", {"command": "RELEASE_ALL"})
        handler.execute_behavior("MOVE", {"direction": "LEFT"})
        self.assertEqual(calls, [("HEAL_LOW", {"key": "h"})])
        self.assertEqual(self.kb._pressed_keys, {"h"})
        self.clock.advance(HEAL_HOLD_SECONDS)
        handler.execute_behavior("IDLE", {"command": "RELEASE_ALL"})
        self.assertEqual(self.kb._pressed_keys, set())

    def test_renewal_does_not_shorten_an_active_lease(self):
        self.assertTrue(self.kb.arm(HEAL_HOLD_SECONDS))
        deadline = self.kb._lease_until
        self.clock.advance(0.2)
        self.assertTrue(self.kb.arm())
        self.assertEqual(self.kb._lease_until, deadline)
        self.clock.advance(1.1)
        self.kb._key_down("h")
        self.assertEqual(self.kb.events, [("down", "h")])

    def test_expired_halt_does_not_grant_on_the_same_attempt(self):
        calls = []
        handler = GuardedActionHandler(
            SimpleNamespace(execute_behavior=lambda *args: calls.append(args)), self.kb)
        self.assertTrue(self.kb.arm())
        self.kb._key_down("left")
        self.clock.advance(1.0)
        self.kb.halt()
        handler.execute_behavior("MOVE", {"direction": "RIGHT"})
        self.assertEqual(calls, [])
        self.assertEqual(self.kb._pressed_keys, set())
        handler.execute_behavior("MOVE", {"direction": "RIGHT"})
        self.assertEqual(calls, [("MOVE", {"direction": "RIGHT"})])

    def test_attack_target_jitter_does_not_cancel_the_windup(self):
        calls = []
        handler = GuardedActionHandler(
            SimpleNamespace(execute_behavior=lambda *args: calls.append(args)), self.kb)
        handler.execute_behavior("ATTACK", {"name": "slime", "distance": 12.0, "direction": "LEFT"})
        epoch = self.kb._lease_epoch
        self.kb._key_down("attack")
        handler.execute_behavior("ATTACK", {"name": "slime", "distance": 11.0, "direction": "LEFT"})
        self.assertEqual(self.kb._lease_epoch, epoch)
        self.assertIn("attack", self.kb._pressed_keys)
        self.assertEqual(len(calls), 2)
        handler.execute_behavior("ATTACK", {"name": "slime", "distance": 11.0, "direction": "RIGHT"})
        self.assertNotIn("attack", self.kb._pressed_keys)
        self.assertGreater(self.kb._lease_epoch, epoch)

    def test_unconsumed_focus_loss_rejects_a_new_lease(self):
        allowed = {"ok": True}
        keyboard = Keyboard(allowed=lambda: allowed["ok"], clock=self.clock, watchdog=False)
        try:
            self.assertTrue(keyboard.arm())
            keyboard._key_down("left")
            allowed["ok"] = False
            keyboard._poll_lease()
            allowed["ok"] = True
            self.assertFalse(keyboard.arm())
            keyboard._key_down("left")
            self.assertEqual(keyboard.events, [("down", "left"), ("up", "left")])
            self.assertIsNotNone(keyboard.consume_focus_losses())
            self.assertTrue(keyboard.arm())
            keyboard._key_down("left")
            self.assertEqual(keyboard.events[-1], ("down", "left"))
        finally:
            keyboard.close()

    def test_halt_releases_remaining_keys_when_one_driver_call_fails(self):
        class Base:
            def __init__(self):
                self.events = []
                self._pressed_keys = set()
                self._pressed_keys_lock = threading.Lock()
                self._current_move = None
                self.fail = {"b"}

            def _key_down(self, key):
                self.events.append(("down", key))
                self._pressed_keys.add(key)

            def _key_up(self, key):
                if key in self.fail:
                    raise RuntimeError("driver")
                self.events.append(("up", key))
                self._pressed_keys.discard(key)

        class Flaky(KeyboardLeaseMixin, Base):
            pass

        keyboard = Flaky(allowed=lambda: True, clock=self.clock, watchdog=False)
        try:
            self.assertTrue(keyboard.arm())
            for key in ("a", "b", "c"):
                keyboard._key_down(key)
            keyboard.halt()
            self.assertEqual(keyboard._pressed_keys, {"b"})
            self.assertIn(("up", "a"), keyboard.events)
            self.assertIn(("up", "c"), keyboard.events)
            keyboard.fail.clear()
            keyboard.halt()
            self.assertEqual(keyboard._pressed_keys, set())
            keyboard._poll_lease = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
            keyboard._watch_once()
        finally:
            keyboard.close()

    def test_second_potion_worker_cannot_share_an_open_hold(self):
        started = []

        class ItemKeyboard(Keyboard):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._item_lock = threading.Lock()
                self._status_item = False

            def _item_command(self, key):
                started.append(key)
                self._key_down(key)
                if key == "h":
                    with self._item_lock:
                        self._status_item = False
                    self._item_command("p")
                self._key_up(key)
                with self._item_lock:
                    self._status_item = False

        keyboard = ItemKeyboard(allowed=lambda: True, clock=self.clock, watchdog=False)
        try:
            self.assertTrue(keyboard.arm())
            keyboard._item_command("h")
            self.assertEqual(started, ["h"])
            self.assertIsNone(keyboard._item_owner_token)
            self.assertFalse(keyboard._status_item)
            keyboard._item_command("p")
            self.assertEqual(started, ["h", "p"])
            self.assertNotEqual(keyboard._item_hold_serial, 1)
        finally:
            keyboard.close()

    def test_focus_loss_is_counted_once_per_edge_and_consumed(self):
        allowed = {"ok": True}
        keyboard = Keyboard(allowed=lambda: allowed["ok"], clock=self.clock, watchdog=False)
        try:
            keyboard._poll_lease()
            self.assertFalse(keyboard.consume_focus_losses())
            allowed["ok"] = False
            keyboard._poll_lease()
            keyboard._poll_lease()
            allowed["ok"] = True
            self.assertTrue(keyboard.consume_focus_losses())
            self.assertFalse(keyboard.consume_focus_losses())
        finally:
            keyboard.close()

    def test_different_motor_command_releases_held_keys(self):
        calls = []
        handler = GuardedActionHandler(
            SimpleNamespace(execute_behavior=lambda *args: calls.append(args)), self.kb)
        handler.execute_behavior("MOVE", {"direction": "LEFT"})
        self.kb._key_down("left")
        handler.execute_behavior("MOVE", {"direction": "LEFT"})
        self.assertEqual(self.kb._pressed_keys, {"left"})
        handler.execute_behavior("CLIMB", {"direction": "UP"})
        self.assertEqual(self.kb._pressed_keys, set())
        self.assertEqual(calls[-1], ("CLIMB", {"direction": "UP"}))

    def test_cancelled_heal_worker_cannot_clear_a_retried_hold(self):
        entered = threading.Event()
        second_entered = threading.Event()
        release_old = threading.Event()
        release_new = threading.Event()

        class ItemKeyboard(Keyboard):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._item_lock = threading.Lock()
                self._status_item = False

            def _item_command(self, key):
                with self._item_lock:
                    self._status_item = True
                self._key_down(key)
                try:
                    if key == "h":
                        entered.set()
                        release_old.wait(2)
                    else:
                        second_entered.set()
                        release_new.wait(2)
                finally:
                    self._key_up(key)
                    with self._item_lock:
                        self._status_item = False

        keyboard = ItemKeyboard(allowed=lambda: True, clock=self.clock, watchdog=False)
        old = threading.Thread(target=lambda: None)
        new = threading.Thread(target=lambda: None)
        try:
            self.assertTrue(keyboard.arm(HEAL_HOLD_SECONDS))
            old = threading.Thread(target=keyboard._item_command, args=("h",))
            old.start()
            self.assertTrue(entered.wait(2))
            self.assertTrue(keyboard._status_item)
            self.assertIn("h", keyboard._pressed_keys)
            keyboard.halt()
            self.assertFalse(keyboard._status_item)
            self.assertNotIn("h", keyboard._pressed_keys)
            self.assertTrue(keyboard.arm(HEAL_HOLD_SECONDS))
            new = threading.Thread(target=keyboard._item_command, args=("p",))
            new.start()
            self.assertTrue(second_entered.wait(2))
            self.assertTrue(keyboard._status_item)
            self.assertIn("p", keyboard._pressed_keys)
            release_old.set()
            old.join(2)
            self.assertFalse(old.is_alive())
            self.assertTrue(keyboard._status_item)
            self.assertIn("p", keyboard._pressed_keys)
            self.assertNotIn("h", keyboard._pressed_keys)
        finally:
            release_old.set()
            release_new.set()
            old.join(2)
            new.join(2)
            keyboard.close()
            self.assertFalse(keyboard._status_item)
            self.assertNotIn("p", keyboard._pressed_keys)


if __name__ == "__main__":
    unittest.main()
