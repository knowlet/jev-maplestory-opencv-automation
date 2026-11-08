"""Pure-Python policies. Workers receive JSON snapshots, never mutable game objects."""
from concurrent.futures import Future
from dataclasses import dataclass
import threading
import time
from typing import Protocol

from .client import Decision, JevError, Settings


class Policy(Protocol):
    def decide(self, state, choices): ...


class FSMPolicy:
    """Thin baseline adapter: exactly one call, no speculative FSM evaluation."""
    def __init__(self, machine):
        self.machine = machine

    def decide(self, state, choices=None):
        return self.machine.handle(state)


@dataclass
class Pending:
    future: Future
    epoch: int
    started: float
    context: tuple
    choices: tuple[str, ...]
    request_id: int
    expired: bool = False


class JevPolicy:
    """Non-blocking, one-flight policy with observation-aged decisions.

    Only the game-loop thread calls decide/reset/close. HTTP threads can outlive a
    socket timeout (e.g. DNS); their results are ignored and no new thread is
    created until the old one exits. This bounds concurrency, not OS DNS latency.
    """
    def __init__(self, client, settings=None, *, clock=time.monotonic, submit=None, emit=None):
        self.client = client
        self.settings = settings or Settings()
        self.clock = clock
        self.submit = submit or self._submit
        self.emit = emit or (lambda event: None)
        self.epoch = 0
        self.pending = None
        self.cached = None
        self.next_request = 0.0
        self.errors = 0
        self.closed = False
        self.last_status = "not_started"
        self.last_decision = None
        self.request_id = 0
        self.last_observation_id = None

    @staticmethod
    def _submit(fn, *args):
        future = Future()
        def run():
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(fn(*args))
            except Exception as exc:
                future.set_exception(exc)
        threading.Thread(target=run, daemon=True, name="jev-http").start()
        return future

    def reset(self):
        self.epoch += 1
        self.cached = None
        self.last_decision = None
        self.last_status = "reset"
        self.last_observation_id = None
        # Keep the in-flight slot occupied until its worker really exits.

    def close(self):
        self.closed = True
        self.reset()

    def _failure(self, reason, now):
        self.cached = None
        self.errors = min(self.errors + 1, 6)
        self.next_request = max(self.next_request, now + min(30.0, 2 ** (self.errors - 1)))
        self.last_status = reason
        self.emit({"event": "jev_failure", "request_id": self.request_id, "reason": reason})

    def decide(self, state, choices, context=()):
        now = self.clock()
        self.last_decision = None
        if self.closed:
            return None
        choices, context = tuple(choices), tuple(context)
        pending = self.pending
        if pending is not None:
            if pending.future.done():
                self.pending = None
                elapsed = now - pending.started
                if pending.expired or pending.epoch != self.epoch:
                    self.last_status = "discarded_epoch_or_deadline"
                elif elapsed > self.settings.timeout:
                    self._failure("deadline", now)
                else:
                    try:
                        decision = pending.future.result()
                        if not isinstance(decision, Decision):
                            raise JevError("invalid_decision")
                        self.errors = 0
                        accepted = (pending.context == context and decision.choice in choices
                                    and decision.confidence >= self.settings.min_confidence)
                        self.emit({"event": "jev_response", "request_id": pending.request_id, "elapsed_ms": round(elapsed * 1000, 3),
                                   "choice": decision.choice, "confidence": decision.confidence,
                                   "probabilities": decision.probabilities, "model": decision.model,
                                   "input_tokens": decision.input_tokens, "accepted": accepted})
                        self.cached = (decision, pending.started, context, pending.request_id) if accepted else None
                        self.last_status = "accepted" if accepted else "context_or_confidence_rejected"
                    except Exception as exc:
                        # Do not leak tokens/bodies from arbitrary transport exception strings.
                        self._failure(str(exc) if isinstance(exc, JevError) else "client_error", now)
            elif not pending.expired and now - pending.started > self.settings.timeout:
                pending.expired = True
                self._failure("deadline", now)
        if self.cached:
            decision, observed_at, cached_context, request_id = self.cached
            if (now - observed_at > self.settings.max_age or cached_context != context
                    or decision.choice not in choices):
                self.cached = None
                self.last_status = "stale"
            else:
                self.last_decision = decision
                self.last_observation_id = request_id
        if self.pending is None and now >= self.next_request and len(choices) > 1:
            # state is built afresh from scalars/lists and is not modified afterwards.
            future = self.submit(self.client.decide, state, choices)
            self.request_id += 1
            self.pending = Pending(future, self.epoch, now, context, choices, self.request_id)
            self.next_request = now + self.settings.interval
            self.emit({"event": "jev_request", "request_id": self.request_id, "state": state, "choices": choices})
        return self.last_decision
