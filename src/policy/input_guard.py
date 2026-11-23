"""Input lease for the experimental launcher, layered over the original backend.

The original keyboard worker methods end in *_command. Bind their generation
when the target is retrieved, so a pre-pause worker cannot press after resume.
This is a best-effort application guard, not a hard-real-time driver guarantee.

A lease is granted when none is held and renewed only while it is still in the
future. Renewal after expiry fails until halt() clears it. Healing outranks
tactical commands for the upstream potion-key hold; a safety halt can cancel
that hold, and a later heal must be able to press again.
"""
import functools
import threading
import time

# KeyBoard._item_command holds a potion key for this long. A heal's input lease
# covers the same interval so the watchdog does not release the key early, and
# tactical commands must not preempt the hold.
HEAL_HOLD_SECONDS = 3.0


class KeyboardLeaseMixin:
    def __init__(self, *, allowed, lease_seconds=1.0, clock=time.monotonic, watchdog=True):
        self._lease_lock = threading.RLock()
        self._lease_clock = clock
        self._lease_seconds = lease_seconds
        self._lease_allowed = allowed
        self._lease_until = 0.0
        self._lease_epoch = 0
        self._grant_blocked = False
        self._focus_losses = 0
        self._focus_seen = 0
        self._focus_down = False
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
                    if name == "_item_command":
                        self._claim_item_owner(epoch)
                    return value(*args, **kwargs)
                finally:
                    if name == "_item_command":
                        # Upstream clears _status_item in its own finally, including
                        # after this worker was cancelled. Restore a newer hold.
                        self._finish_item_worker(epoch)
                    if previous is None:
                        del local.epoch
                    else:
                        local.epoch = previous
            return generation_bound
        return value

    def _item_state_lock(self):
        lock = getattr(self, "_item_lock", None)
        if lock is None or not hasattr(self, "_status_item"):
            return None
        return lock

    def _claim_item_owner(self, epoch):
        lock = self._item_state_lock()
        if lock is None:
            return
        with lock:
            # A worker that already lost the lease must not steal a newer heal.
            if epoch == self._lease_epoch:
                current = getattr(self, "_item_owner_epoch", None)
                if current is None or current == epoch:
                    self._item_owner_epoch = epoch

    def _finish_item_worker(self, epoch):
        lock = self._item_state_lock()
        if lock is None:
            return
        with lock:
            owner = getattr(self, "_item_owner_epoch", None)
            if owner is None:
                return
            if owner == epoch:
                self._status_item = False
                self._item_owner_epoch = None
                return
            if owner == self._lease_epoch:
                self._status_item = True

    def _cancel_item_hold(self):
        lock = self._item_state_lock()
        if lock is None:
            return
        with lock:
            self._status_item = False
            self._item_owner_epoch = None

    def _permitted(self):
        try:
            return (not self._lease_closed and self._lease_clock() < self._lease_until
                    and getattr(self._lease_local, "epoch", self._lease_epoch) == self._lease_epoch
                    and self._lease_allowed())
        except Exception:
            return False

    def arm(self, seconds=None):
        """Grant a lease, or extend one that has not expired.

        Returns False when the keyboard is closed, the duration is unusable,
        the current lease has expired, or the previous halt cleared an expired
        lease and that rejection has not been consumed yet. Renewal never
        moves an active deadline earlier.
        """
        with self._lease_lock:
            if self._lease_closed:
                return False
            now = self._lease_clock()
            if self._grant_blocked:
                self._grant_blocked = False
                return False
            if self._lease_until > 0.0 and now >= self._lease_until:
                return False
            duration = self._lease_seconds if seconds is None else seconds
            if type(duration) not in (int, float):
                return False
            duration = float(duration)
            if duration != duration or duration <= 0.0 or duration == float("inf"):
                return False
            proposed = now + duration
            if proposed > self._lease_until:
                self._lease_until = proposed
            return True

    def acknowledge_expired_grant(self):
        """The rejected attempt has finished. The following attempt may grant."""
        with self._lease_lock:
            self._grant_blocked = False

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
            now = self._lease_clock()
            expired = self._lease_until > 0.0 and now >= self._lease_until
            with self._pressed_keys_lock:
                keys = list(self._pressed_keys)
            if self._lease_until or keys:
                self._lease_epoch += 1
            self._lease_until = 0.0
            if expired:
                # Clearing an expired lease must not let the same attempt grant again.
                self._grant_blocked = True
            self._current_move = None
            # Synchronous key-up, bypassing cancelled-worker filtering. No move lock:
            # an upstream timed movement worker may hold it while sleeping.
            for key in keys:
                super()._key_up(key)
            self._cancel_item_hold()

    def release_all(self):
        self.halt()

    def _focus_ok(self):
        try:
            return bool(self._lease_allowed())
        except Exception:
            return False

    def _poll_lease(self):
        with self._lease_lock:
            # One count per focus-loss edge. The game loop consumes it and suspends
            # even if focus has already returned before the next frame.
            if self._focus_ok():
                self._focus_down = False
            elif not self._focus_down:
                self._focus_losses += 1
                self._focus_down = True
            if not self._permitted():
                self.halt()

    def consume_focus_losses(self):
        with self._lease_lock:
            lost = self._focus_losses != self._focus_seen
            self._focus_seen = self._focus_losses
            return lost

    def _watch_lease(self):
        while not self._lease_stop.wait(0.025):
            self._poll_lease()

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


def _is_heal(action, params):
    return (isinstance(action, str) and action.startswith("HEAL_")
            and isinstance(params, dict)
            and isinstance(params.get("key"), str) and bool(params["key"]))


def _command_signature(action, params):
    # Distance and target name change every frame. Including them restarts the
    # attack worker and can suppress a short wind-up indefinitely.
    if isinstance(params, dict) and isinstance(params.get("key"), str) and params.get("key"):
        return (action, params["key"])
    direction = params.get("direction") if isinstance(params, dict) else None
    return (action, direction)


def _accepted(action, params):
    if not isinstance(action, str) or not isinstance(params, dict):
        return False
    if action in DIRECTIONS:
        direction = params.get("direction")
        return isinstance(direction, str) and direction in DIRECTIONS[action]
    return _is_heal(action, params)


class GuardedActionHandler:
    def __init__(self, handler, keyboard):
        self.handler = handler
        self.keyboard = keyboard
        self._heal_until = 0.0
        self._command = None
        handler.keyboard = keyboard

    def _now(self):
        return self.keyboard._lease_clock()

    def _heal_in_progress(self):
        # The hold is live only while its covering lease has not been halted.
        now = self._now()
        return self._heal_until > now and self.keyboard._lease_until > now

    def execute_behavior(self, action, params=None):
        if action is None:
            return
        params = params or {}
        healing = _is_heal(action, params)
        if self._heal_in_progress() and not healing:
            # Tactical and idle commands must not release the potion key.
            self.keyboard.arm(self._heal_until - self._now())
            return
        if action == "IDLE" or not _accepted(action, params):
            # For the experiment, every IDLE stops inputs, not just horizontal motion.
            self._heal_until = 0.0
            self._command = None
            self.keyboard.halt()
            self.keyboard.acknowledge_expired_grant()
            return
        if healing and not self._heal_in_progress():
            duration = HEAL_HOLD_SECONDS
            fresh_heal = True
        elif healing:
            duration = self._heal_until - self._now()
            fresh_heal = False
        else:
            duration = None
            fresh_heal = False
        signature = _command_signature(action, params)
        if self._command is not None and signature != self._command:
            # A new motor command must not inherit keys held by the previous one.
            self.keyboard.halt()
        if not self.keyboard.arm(duration):
            # Expired lease: do not dispatch, and do not extend the deadline.
            self._heal_until = 0.0
            self._command = None
            self.keyboard.halt()
            self.keyboard.acknowledge_expired_grant()
            return
        self._command = signature
        if fresh_heal:
            self._heal_until = self._now() + HEAL_HOLD_SECONDS
        try:
            self.handler.execute_behavior(action, params)
        except Exception:
            self._heal_until = 0.0
            self._command = None
            self.keyboard.halt()
            self.keyboard.acknowledge_expired_grant()
            raise

