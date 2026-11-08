# Jev tactical-policy experiment

Experimental branch: `jev-policy`, based on upstream commit
`0df512a300aa3d793a28394054055b10c7e76ad3` (the default branch is **master**).
The original `main.py`, detector, FSM, map configuration and timed skills are not edited.
The opt-in entrypoint is **`jev_main.py`**. Do not merge on the assumption that
headless unit tests prove compatibility with the current game or input driver.

## Scope

```text
OpenCV -> original GameState -> original AutoControl health priority
                                    |
                         PolicyBridge (three modes)
                         /          |             
                       FSM       shadow           Jev
                         \          |             /
                   original deterministic skills / geometry
                                    |
                         guarded ActionHandler -> keyboard
```

Jev selects one of `IDLE`, `PATROL`, `COMBAT`, `NAVIGATE`, `RECOVER_STUCK`.
The legal subset is computed from the current observation. A **single Choice**
selects the tactic; independent action/direction/target answers are not combined
into potentially inconsistent commands. Jev does **not** choose individual
keypress timings, arbitrary Python functions, potion keys or arbitrary coordinates.
Combat targeting and rope selection still use the original local helpers. This is
a tactical-layer replacement, **not a replacement of every decision in the repo**.

The real detector's `mobs` value is a list of `(label, detections)` pairs. The adapter
normalizes that format, converts ROI-relative detections to screen coordinates,
and sends only a bounded structured snapshot. No screenshots, player names,
API keys or local file paths are sent in state. Snapshots and responses are logged
locally; do not upload real-game logs without reviewing their contents.

## Windows setup and first run

Complete the original README's Windows, Python, input-driver, template, map,
resolution and keybind setup first. This integration adds **no pip dependency**;
its HTTP client uses Python's standard library. Python 3.10+ is required.

In an **elevated PowerShell**, from the repository:

```powershell
git fetch origin
git switch jev-policy
python -m pip install -r requirements.txt
python tools/check_policy.py

# Baseline under the same guarded harness, no API key/network required.
python jev_main.py --policy fsm --log artifacts/policy/fsm.jsonl
```

All experimental modes start **paused**: F9 starts/pauses; F12 exits. Keep the
same game configuration and seed for all modes. Stop each run before starting the next.
Then supply the key in the same elevated shell (do not put it in YAML or Git):

```powershell
$secureKey = Read-Host 'TypeSafe API key' -AsSecureString
$env:TYPESAFE_API_KEY = [System.Net.NetworkCredential]::new('', $secureKey).Password
Remove-Variable secureKey
$env:JEV_MODEL = 'jev-latest'

# First inspect decisions while ONLY the original FSM controls the game.
python jev_main.py --policy shadow --log artifacts/policy/shadow.jsonl

# Then give Jev the tactical choice. There is no automatic FSM fallback.
python jev_main.py --policy jev --log artifacts/policy/jev.jsonl

Remove-Item Env:TYPESAFE_API_KEY
```

The key is read at process startup. The launcher deliberately does not auto-elevate:
UAC relaunches can lose the caller's environment. API errors never log the key or
raw HTTP error body. The endpoint is fixed to `https://api.typesafe.ai/v1/systemone`,
uses Bearer authentication and rejects redirects.

## Modes and controls

| Mode | Executed controller | Jev requests |
| --- | --- | --- |
| `fsm` (default) | Original FSM, once per state update | None |
| `shadow` | Original FSM only; no Jev skill execution | Yes |
| `jev` | Jev-selected tactic -> original timed skills | Yes |

Environment configuration (seconds unless stated):

| Variable | Default | Meaning |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | required for shadow/jev | Secret, never committed |
| `JEV_MODEL` | `jev-latest` | Use a pinned model ID when available for reproducibility |
| `JEV_INTERVAL` | `0.5` | Minimum interval between new requests; minimum allowed 0.2 |
| `JEV_TIMEOUT` | `1.0` | Socket timeout and game-loop response deadline |
| `JEV_MAX_AGE` | `1.5` | Maximum cached decision age **from observation**, not response |
| `JEV_MIN_CONFIDENCE` | `0.35` | Initial experiment threshold, **not calibrated on this game** |
| `JEV_INPUT_LEASE` | `1.0` | Motor-input lease renewed only by valid commands |

API requests use a single daemon worker slot. A delayed DNS/socket worker can
outlive a timeout, but is ignored and occupies its slot until it exits: no
unbounded replacement threads. HTTP failures back off, using a **new** snapshot
for the next call rather than retrying stale input. Returned types, legal choices,
finite probabilities, distribution sum, argmax and confidence are validated.
Probability and confidence are recorded separately; confidence is not treated
as a measured probability of successful gameplay.

Missing tracking, zero HP, pause, focus loss, invalid geometry, API deadline,
low confidence or invalid/stale choices stop Jev-controlled movement rather than
silently using the FSM. Normal configured healing remains outside Jev. No extra
rules for bans, evasion, runes or anti-cheat are added by this experiment.

## Shared input guard and baseline caveat

The upstream game loop and ActionHandler create separate keyboard instances.
The experimental launcher binds both to **one actual action keyboard**. It adds
an input lease, synchronous tracked-key release, foreground checks at key-down,
and generation-tagged worker targets so pre-pause workers cannot press after
resume. A watchdog revokes expired input without waiting for HTTP.

The same guard is applied in all three experimental modes. In particular, every
`IDLE` releases all keys rather than only horizontal movement. Thus:

* the **FSM policy implementation is unchanged**;
* the **experimental harness is deliberately safer, not byte-for-byte behavior
  equivalent to running upstream `main.py`**.

Existing motor helpers still include blocking sleeps and native-driver behavior.
The guard is best effort, not a hard-real-time guarantee. Validate foreground,
minimize, F9/F12, disconnect and held-key behavior on the actual Windows machine
before unattended operation. Use only an environment where automation is permitted.

## Tests, CI and evidence

```powershell
# Offline, no game, driver, model credentials or third-party testing packages.
python tools/check_policy.py

# In a complete checkout, also check upstream method/worker naming contracts.
python tools/check_policy.py --upstream

# Optional: exactly one billed request on synthetic data, never sends game inputs.
python tools/check_policy.py --live
```

Reports are written to `artifacts/policy/tests.txt` and `results.json`.
`policy-ci.yml` runs offline tests on Linux/Python 3.10 and Windows/Python 3.12,
checks source compilation and uploads reports using `actions/upload-artifact`.
Its optional `workflow_dispatch` input `live_api=true` requires repository secret
`TYPESAFE_API_KEY`; repository variable `JEV_MODEL` is optional. Push/PR tests
never receive the API secret or make billed model calls. A newly added workflow
may not appear in the dispatch UI until GitHub indexes it/the default branch has it;
the local `--live` command is independent of that UI.

Logs have a `session_id` and request IDs. `jev_request` contains the exact model
snapshot and legal options; shadow requests additionally record the original
motor command **in the log, not in model state**. `jev_response` records choice,
confidence, distribution, model, usage and observation-to-poll latency. Shadow
observations compare different abstractions (tactic vs motor command), not labels
of truth. Request/response IDs should be used to align observations.

Offline tests validate software behavior under fakes, including stale responses,
backoff, shadow isolation and cancelled input workers. A synthetic API smoke test
validates connectivity/schema, not game competence. **EXP/min, kills/min, recovery
success, calibration and superiority to FSM are not measured by this PR.**
For a gameplay A/B test, hold map, class, gear, observation pipeline and guard
settings fixed; randomize run order; record manual interventions and outcomes.
Do not report FSM agreement or a successful HTTP call as gameplay success.

## Official API references

- https://docs.typesafe.ai/api
- https://docs.typesafe.ai/primitives/choice
- https://docs.typesafe.ai/confidence

API contract checked 2026-09-19. The alias `jev-latest` and service performance can change.
