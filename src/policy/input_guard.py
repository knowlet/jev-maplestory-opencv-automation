"""Input lease for the experimental launcher, layered over the original backend.

The original keyboard worker methods end in *_command. Bind their generation
when the target is retrieved, so a pre-pause worker cannot press after resume.
This is a best-effort application guard, not a hard-real-time driver guarantee.
"""
import functools
import threading
import time


class KeyboardLeaseMixin:
    def __init__(self, *, allowed, lease_seconds=1.0, clock=time.monotonic, watchdog=True):
        self._lease_lock = threading.RLock()
        self._lease_clock = clock
        self._lease_seconds = lease_seconds
        self._lease_allowed = allowed
        self._lease_until = 0.0
        self._lease_epoch = 0
        self._lease_local = threading.local()
        self._lease_closed = False
        self._lease_stop = threading.Event()
        super().__init__()
        self._lease_watchdog = None
        if watchdog:
            self._lease_watchdog = threading.Thread(target=self._watch_lease, daemon=True, name="input-lease")
            self._lease_watchdog.start()

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name.endswith("_command") and callable(value):
            epoch = super().__getattribute__("_lease_epoch")
            @functools.wraps(value)
            def generation_bound(*args, **kwargs):
                local = self._lease_local
                previous = getattr(local, "epoch", None)
                local.epoch = epoch
                try:
                    return value(*args, **kwargs)
                finally:
                    if previous is None:
                        del local.epoch
                    else:
                        local.epoch = previous
            return generation_bound
        return value

    def _permitted(self):
        try:
            return (not self._lease_closed and self._lease_clock() < self._lease_until
                    and getattr(self._lease_local, "epoch", self._lease_epoch) == self._lease_epoch
                    and self._lease_allowed())
        except Exception:
            return False

    def arm(self):
        with self._lease_lock:
            if not self._lease_closed:
                self._lease_until = self._lease_clock() + self._lease_seconds

    def _key_down(self, key):
        with self._lease_lock:
            if self._permitted():
                return super()._key_down(key)

    def _key_up(self, key):
        with self._lease_lock:
            # A cancelled worker must not release a key owned by a new generation.
            if getattr(self._lease_local, "epoch", self._lease_epoch) == self._lease_epoch:
                return super()._key_up(key)

    def _press_direction_locked(self, direction):
        with self._lease_lock:
            if self._permitted():
                return super()._press_direction_locked(direction)

    def _release_direction_locked(self):
        with self._lease_lock:
            if getattr(self._lease_local, "epoch", self._lease_epoch) == self._lease_epoch:
                return super()._release_direction_locked()

    def halt(self):
        with self._lease_lock:
            with self._pressed_keys_lock:
                keys = list(self._pressed_keys)
            if self._lease_until or keys:
                self._lease_epoch += 1
            self._lease_until = 0.0
            self._current_move = None
            # Synchronous key-up, bypassing cancelled-worker filtering. No move lock:
            # an upstream timed movement worker may hold it while sleeping.
            for key in keys:
                super()._key_up(key)

    def release_all(self):
        self.halt()

    def _watch_lease(self):
        while not self._lease_stop.wait(0.025):
            with self._lease_lock:
                if not self._permitted():
                    self.halt()

    def close(self):
        with self._lease_lock:
            self._lease_closed = True
            self.halt()
        self._lease_stop.set()
        if self._lease_watchdog and self._lease_watchdog is not threading.current_thread():
            self._lease_watchdog.join(timeout=0.2)


DIRECTIONS = {
    "MOVE": {"LEFT", "RIGHT"}, "ATTACK": {"LEFT", "RIGHT"},
    "JUMP": {"LEFT", "RIGHT", "NONE", "DOWN"},
    "ROPE": {"LEFT_UP", "RIGHT_UP", "LEFT_DOWN", "RIGHT_DOWN", "UP", "DOWN"},
    "CLIMB": {"UP", "DOWN"}, "SMALL_MOVE": {"M2LEFT", "M2RIGHT"},
    "JUMP_GRAB": {"LEFT", "RIGHT", "UP"},
}


class GuardedActionHandler:
    def __init__(self, handler, keyboard):
        self.handler = handler
        self.keyboard = keyboard
        handler.keyboard = keyboard

    def execute_behavior(self, action, params=None):
        if action is None:
            return
        params = params or {}
        if action == "IDLE":
            # For the experiment, every IDLE stops inputs, not just horizontal motion.
            self.keyboard.halt()
            return
        valid = (isinstance(action, str) and isinstance(params, dict) and
                 ((action in DIRECTIONS and params.get("direction") in DIRECTIONS[action])
                  or (isinstance(action, str) and action.startswith("HEAL_")
                      and isinstance(params.get("key"), str) and bool(params["key"]))))
        if not valid:
            self.keyboard.halt()
            return
        self.keyboard.arm()
        try:
            self.handler.execute_behavior(action, params)
        except Exception:
            self.keyboard.halt()
            raise
