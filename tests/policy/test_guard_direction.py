import unittest
from types import SimpleNamespace
from test_policy import Clock, Keyboard
from src.policy.input_guard import DIRECTIONS, GuardedActionHandler

class DirectionShapeTests(unittest.TestCase):
    def test_invalid_direction_stops_held_inputs_without_dispatch(self):
        for action in DIRECTIONS:
            for direction in ([], {}, ['LEFT'], {'value': 'LEFT'}, None, 1, True):
                with self.subTest(action=action, direction=direction):
                    calls = []
                    keyboard = Keyboard(allowed=lambda: True, clock=Clock(), watchdog=False)
                    try:
                        handler = GuardedActionHandler(SimpleNamespace(execute_behavior=lambda *args: calls.append(args)), keyboard)
                        keyboard.arm()
                        keyboard._key_down('left')
                        handler.execute_behavior(action, {'direction': direction})
                        self.assertEqual(keyboard._pressed_keys, set())
                        self.assertEqual(calls, [])
                        self.assertEqual(keyboard._lease_until, 0.0)
                    finally:
                        keyboard.close()

    def test_all_valid_directions_keep_dispatch(self):
        for action, directions in DIRECTIONS.items():
            for direction in directions:
                with self.subTest(action=action, direction=direction):
                    calls = []
                    keyboard = Keyboard(allowed=lambda: True, clock=Clock(), watchdog=False)
                    try:
                        handler = GuardedActionHandler(SimpleNamespace(execute_behavior=lambda *args: calls.append(args)), keyboard)
                        params = {'direction': direction}
                        handler.execute_behavior(action, params)
                        self.assertEqual(calls, [(action, params)])
                        self.assertGreater(keyboard._lease_until, 0)
                    finally:
                        keyboard.close()
