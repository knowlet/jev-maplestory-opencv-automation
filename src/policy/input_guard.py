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
        self._focus_lost_at = None
        self._pending_releases = []
        self._pinned_epoch = None
        self._last_grant_epoch = 0
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
            pinned = super().__getattribute__("_pinned_epoch")
            epoch = pinned if pinned is not None else super().__getattribute__("_lease_epoch")
            @functools.wraps(value)
            def generation_bound(*args, **kwargs):
                local = self._lease_local
                previous = getattr(local, "epoch", None)
                local.epoch = epoch
                token = None
                try:
                    if name == "_item_command":
                        token = self._begin_item_hold()
                        # A hold is already owned. Do not run a second potion worker.
                        if token is None and self._item_state_lock() is not None:
                            return None
                    return value(*args, **kwargs)
                finally:
                    if name == "_item_command":
                        self._finish_item_worker(token)
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

    def _begin_item_hold(self):
        """Give this potion press its own token. A busy hold is not shared.

        The worker's captured generation and the live lease are checked under
        the same lock halt() holds, so a cancelled worker cannot claim the
        token and block a retry for the upstream potion sleep.
        """
        lock = self._item_state_lock()
        if lock is None:
            return None
        with self._lease_lock:
            bound = getattr(self._lease_local, "epoch", None)
            if bound != self._lease_epoch or self._lease_closed:
                return None
            if not (self._lease_clock() < self._lease_until):
                return None
            with lock:
                if getattr(self, "_item_owner_token", None) is not None:
                    return None
                token = getattr(self, "_item_hold_serial", 0) + 1
                self._item_hold_serial = token
                self._item_owner_token = token
                self._status_item = True
                return token

    def _finish_item_worker(self, token):
        """One locked update: clear the flag only if this hold still owns it."""
        lock = self._item_state_lock()
        if lock is None or token is None:
            return
        with lock:
            owner = getattr(self, "_item_owner_token", None)
            if owner == token:
                self._status_item = False
                self._item_owner_token = None
                return
            # Upstream just cleared the flag. A newer hold still owns it.
            if owner is not None:
                self._status_item = True

    def _cancel_item_hold(self):
        lock = self._item_state_lock()
        if lock is None:
            return
        with lock:
            self._status_item = False
            self._item_owner_token = None

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
            # A focus loss the game loop has not consumed yet must not be re-armed.
            if self._focus_losses != self._focus_seen:
                return False
            # A key the driver failed to release must not stay down under a new lease.
            if self._pending_releases:
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
            self._last_grant_epoch = self._lease_epoch
            # Same critical section as the grant, so a halt cannot slip in
            # before worker targets are bound to this generation.
            self._pinned_epoch = self._lease_epoch
            return True

    def acknowledge_expired_grant(self):
        """The rejected attempt has finished. The following attempt may grant."""
        with self._lease_lock:
            self._grant_blocked = False

    def pin_granted_dispatch(self):
        """Bind later worker-target lookups to the epoch captured with the grant."""
        with self._lease_lock:
            self._pinned_epoch = self._last_grant_epoch

    def unpin_dispatch(self):
        with self._lease_lock:
            self._pinned_epoch = None

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
            # One driver failure must not skip the rest or escape into the watchdog.
            failed = []
            release_keys = list(keys)
            for key in self._pending_releases:
                if key not in release_keys:
                    release_keys.append(key)
            for key in release_keys:
                try:
                    super()._key_up(key)
                except Exception:
                    failed.append(key)
            self._pending_releases = failed
            if failed:
                with self._pressed_keys_lock:
                    for key in failed:
                        self._pressed_keys.add(key)
            self._cancel_item_hold()

    def _retry_pending_releases(self):
        """Caller holds the lease lock. Retry even when the lease is still valid."""
        if not self._pending_releases:
            return
        still = []
        for key in list(self._pending_releases):
            try:
                super()._key_up(key)
            except Exception:
                still.append(key)
            else:
                with self._pressed_keys_lock:
                    self._pressed_keys.discard(key)
        self._pending_releases = still

    def release_all(self):
        self.halt()

    def _focus_ok(self):
        try:
            return bool(self._lease_allowed())
        except Exception:
            return False

    def _record_focus(self, focused):
        # One count per focus-loss edge. Keep the earliest unconsumed time.
        if focused:
            self._focus_down = False
            return
        if not self._focus_down:
            self._focus_losses += 1
            self._focus_down = True
            if self._focus_lost_at is None:
                self._focus_lost_at = self._lease_clock()

    def _permitted_focused(self, focused):
        try:
            return (not self._lease_closed and self._lease_clock() < self._lease_until
                    and getattr(self._lease_local, "epoch", self._lease_epoch) == self._lease_epoch
                    and focused)
        except Exception:
            return False

    def _poll_lease(self):
        with self._lease_lock:
            # Sample focus once. A second sample can halt without recording the loss.
            focused = self._focus_ok()
            self._record_focus(focused)
            self._retry_pending_releases()
            if not self._permitted_focused(focused):
                self.halt()

    def consume_focus_losses(self):
        """Return the clock time of an unconsumed focus loss, or None."""
        with self._lease_lock:
            if self._focus_losses == self._focus_seen:
                return None
            self._focus_seen = self._focus_losses
            lost_at = self._focus_lost_at
            self._focus_lost_at = None
            return lost_at

    def _watch_once(self):
        try:
            self._poll_lease()
        except Exception:
            return

    def _watch_lease(self):
        while not self._lease_stop.wait(0.025):
            self._watch_once()

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
            self.keyboard.unpin_dispatch()
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
        # Worker targets looked up below stay bound to this grant, even if the
        # watchdog halts before the lookup.
        self.keyboard.pin_granted_dispatch()
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
        finally:
            self.keyboard.unpin_dispatch()

