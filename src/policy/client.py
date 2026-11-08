"""Small, dependency-free client for the documented TypeSafe System One API.

Reference: https://docs.typesafe.ai/api (checked 2026-09-19).
This is not a chat-completions API. Do not log credentials or HTTP error bodies.
"""
from dataclasses import dataclass, field
import json
import math
import os
import urllib.error
import urllib.request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MAX_RESPONSE_BYTES = 256_000
CRITERIA = {
    "IDLE": "Stop all inputs; no useful safe tactical action is apparent.",
    "PATROL": "Search along the current platform using the existing route controller.",
    "COMBAT": "Chase and attack visible nearby enemies using the existing combat skill.",
    "NAVIGATE": "Move to a reachable rope and traverse it to search another platform.",
    "RECOVER_STUCK": "Attempt recovery after intended movement has made no progress.",
}
INSTRUCTIONS = (
    "Choose the next short-horizon tactic for a 2D platform-game experiment. "
    "Prefer useful combat when enemies are present; patrol to search; navigate when "
    "the current platform is unproductive; recover only when motion is stuck. "
    "Only the supplied choices are legal. Timed skills, targeting, healing and "
    "emergency stops are handled by deterministic code, not by you. "
    "Coordinates are pixels with y increasing downwards; minimap and screen "
    "coordinates are separate systems. Missing health is unknown, not zero."
)


class JevError(RuntimeError):
    """A sanitized error safe to put in experiment logs."""


@dataclass(frozen=True)
class Settings:
    model: str = "jev-latest"
    interval: float = 0.5
    timeout: float = 1.0
    max_age: float = 1.5
    min_confidence: float = 0.35
    input_lease: float = 1.0

    def __post_init__(self):
        for name in ("interval", "timeout", "max_age", "input_lease"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.interval < 0.2:
            raise ValueError("interval must be at least 0.2 seconds")
        if not self.max_age > self.timeout:
            raise ValueError("max_age must exceed timeout")
        if isinstance(self.min_confidence, bool) or not math.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if not self.model or len(self.model) > 128:
            raise ValueError("invalid model name")

    @classmethod
    def from_env(cls):
        defaults = cls()
        return cls(
            model=os.environ.get("JEV_MODEL", defaults.model),
            **{name: float(os.environ.get("JEV_" + name.upper(), getattr(defaults, name)))
               for name in ("interval", "timeout", "max_age", "min_confidence", "input_lease")},
        )


@dataclass(frozen=True)
class Decision:
    choice: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    model: str = ""
    input_tokens: int | None = None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a Bearer token to a redirect destination.
        raise JevError("redirect_rejected")


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError("invalid_probability")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise JevError("invalid_probability")
    return float(value)


def parse_response(payload, choices):
    try:
        answer = payload["answers"]["tactic"]
        if answer["type"] != "choice" or answer["choice"] not in choices:
            raise JevError("invalid_choice")
        probabilities = answer["probabilities"]
        if not isinstance(probabilities, dict) or set(probabilities) != set(choices):
            raise JevError("invalid_probability_keys")
        probabilities = {key: _probability(value) for key, value in probabilities.items()}
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.002):
            raise JevError("invalid_probability_sum")
        if probabilities[answer["choice"]] + 1e-6 < max(probabilities.values()):
            raise JevError("choice_not_argmax")
        confidence = _probability(answer["confidence"])
        model = payload["model"]
        if not isinstance(model, str) or not model or len(model) > 128:
            raise JevError("invalid_model")
        tokens = payload.get("usage", {}).get("input_tokens")
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise JevError("invalid_usage")
        return Decision(answer["choice"], confidence, probabilities, model, tokens)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise JevError("invalid_response") from exc


class JevClient:
    def __init__(self, settings=None, *, api_key=None, transport=None):
        self.settings = settings or Settings.from_env()
        self._api_key = (api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")).strip()
        if not self._api_key or "\n" in self._api_key or "\r" in self._api_key:
            raise ValueError("Set TYPESAFE_API_KEY in the environment; never commit it")
        self._transport = transport or urllib.request.build_opener(NoRedirect()).open

    def decide(self, state, choices):
        choices = tuple(choices)
        if not choices or len(set(choices)) != len(choices) or any(c not in CRITERIA for c in choices):
            raise ValueError("invalid tactical choices")
        body = json.dumps({
            "model": self.settings.model,
            "state": state,
            "questions": {"tactic": {"type": "choice", "instructions": INSTRUCTIONS,
                                       "criteria": {c: CRITERIA[c] for c in choices}}},
        }, allow_nan=False, ensure_ascii=False).encode("utf-8")
        if len(body) > 64_000:
            raise JevError("request_too_large")
        req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json", "Accept": "application/json",
        })
        try:
            with self._transport(req, timeout=self.settings.timeout) as response:
                if response.status != 200:
                    raise JevError(f"http_{response.status}")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise JevError("response_too_large")
            return parse_response(json.loads(raw), choices)
        except urllib.error.HTTPError as exc:
            # No inline retries: retry only with a new observation after cooldown.
            raise JevError(f"http_{exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise JevError("transport_error") from None
        except (UnicodeError, json.JSONDecodeError):
            raise JevError("invalid_json") from None
