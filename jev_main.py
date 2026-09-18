"""Opt-in experimental launcher; main.py remains the untouched upstream entrypoint."""
import argparse
import ctypes
import os
from pathlib import Path
import random
import sys

from src.policy.client import JevClient, Settings
from src.policy.integration import AutoControlAdapter, JsonlLog, PolicyBridge
from src.policy.input_guard import GuardedActionHandler, KeyboardLeaseMixin


def main(argv=None):
    parser = argparse.ArgumentParser(description="Jev tactical-policy experiment (Windows only)")
    parser.add_argument("--policy", choices=("fsm", "shadow", "jev"), default="fsm")
    parser.add_argument("--log", default="artifacts/policy/policy.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("The live game/input backend needs Windows; headless policy tests do not")
    if not ctypes.windll.shell32.IsUserAnAdmin():
        parser.error("Run from an elevated PowerShell; automatic elevation can lose API-key environment variables")
    try:
        settings = Settings.from_env()
        client = JevClient(settings) if args.policy != "fsm" else None
    except ValueError as exc:
        parser.error(str(exc))
    # Paths in upstream configuration are relative to the repository, not this module.
    os.chdir(Path(__file__).resolve().parent)
    random.seed(args.seed)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()
    # Lazy imports: --help and unit tests never load input drivers or capture devices.
    from src.engine.GameBot import GameBot
    from src.action.KeyBoardController import KeyBoard
    from config.config_loader import config
    import src.utils.logger as logger

    class SafeKeyboard(KeyboardLeaseMixin, KeyBoard):
        pass

    class ExperimentBot(GameBot):
        def active(self):
            try:
                return (self.bot_enabled and GameBot._is_window_valid(self)
                        and GameBot._is_game_window_foreground(self))
            except Exception:
                return False

        def _load_game_resources(self):
            super()._load_game_resources()
            self.policy_log = log
            self.keyboard = SafeKeyboard(allowed=self.active, lease_seconds=settings.input_lease)
            self.action_handler = GuardedActionHandler(self.action_handler, self.keyboard)
            capabilities = {
                "patrol": bool(config.get("auto_control_config.TOGGLE_PATROL_ACTION", True)),
                "combat": bool(config.get("auto_control_config.TOGGLE_COMBAT_ACTION", True)),
                "stuck": bool(config.get("auto_control_config.TOGGLE_Stuck_ACTION", True)),
            }
            self.bridge = PolicyBridge(self.auto_control, args.policy, client=client, settings=settings,
                                       emit=log, capabilities=capabilities)
            self.auto_control = AutoControlAdapter(
                self.auto_control, self.bridge,
                allowed=lambda: self.active() and self.loss_tracking_count == 0,
                halt=self.keyboard.halt,
            )
            # Require F9 to start; do not begin sending keys when a new launcher opens.
            self.bot_enabled = False

        def _toggle_bot(self):
            if hasattr(self, "bridge"):
                self.auto_control.suspend()
            super()._toggle_bot()

        def _is_window_valid(self):
            valid = super()._is_window_valid()
            if not valid and hasattr(self, "bridge"):
                self.auto_control.suspend()
                # Upstream skips hotkey polling while minimized. Preserve emergency exit.
                self.hotkey_manager.poll()
            return valid

        def _is_game_window_foreground(self):
            focused = super()._is_game_window_foreground()
            if not focused and hasattr(self, "bridge"):
                self.auto_control.suspend()
            return focused

    logger.setup_logging()
    log = JsonlLog(args.log)
    log({"event": "session_start", "mode": args.policy, "seed": args.seed,
         "model": settings.model, "interval": settings.interval,
         "timeout": settings.timeout, "max_age": settings.max_age,
         "min_confidence": settings.min_confidence})
    bot = None
    try:
        bot = ExperimentBot()
        bot.run()
    finally:
        if bot is not None and hasattr(bot, "bridge"):
            bot.bridge.close()
        if bot is not None and isinstance(getattr(bot, "keyboard", None), KeyboardLeaseMixin):
            bot.keyboard.close()
        log({"event": "session_end"})
        log.close()


if __name__ == "__main__":
    main()
