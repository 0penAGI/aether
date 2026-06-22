"""Aether configuration: first-run wizard, model selection, and persistence."""

import json
import os
import sys
import shutil
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "aether"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _detect_ollama() -> bool:
    """Check if Ollama is installed and running."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _list_ollama_models() -> list:
    """Query Ollama for available models."""
    import subprocess, json
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        models = []
        for line in result.stdout.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if parts:
                name = parts[0]
                models.append(name)
        return models
    except Exception:
        return []


def _test_ollama_model(model: str) -> bool:
    """Test that a model is available in Ollama."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "show", model],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _test_openai_api(api_base: str, api_key: str = "") -> bool:
    """Test an OpenAI-compatible API endpoint."""
    try:
        import requests
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        }
        r = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers=headers, json=payload, timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


class AetherConfig:
    """Manages Aether configuration persistence and first-run wizard."""

    def __init__(self):
        self.config = self._defaults()
        if CONFIG_FILE.exists():
            self.load()

    @staticmethod
    def _defaults() -> dict:
        return {
            "backend": "ollama",
            "model": "gemma4:12b-mlx",
            "api_base": "http://localhost:11434",
            "api_key": "",
            "skip_confirmation": False,
        }

    def exists(self) -> bool:
        return CONFIG_FILE.exists()

    def load(self) -> dict:
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            self.config = {**self._defaults(), **data}
        except Exception:
            self.config = self._defaults()
        return self.config

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def model(self) -> str:
        return self.config.get("model", self._defaults()["model"])

    @property
    def backend(self) -> str:
        return self.config.get("backend", "ollama")

    def run_wizard(self):
        """Interactive first-run setup."""
        console = _get_console()

        console.print()
        console.print("[bold cyan]╔══════════════════════════════════════╗[/bold cyan]")
        console.print("[bold cyan]║      Aether — First Run Setup        ║[/bold cyan]")
        console.print("[bold cyan]╚══════════════════════════════════════╝[/bold cyan]")
        console.print()

        # Step 1: detect Ollama
        has_ollama = _detect_ollama()
        if has_ollama:
            console.print("[green]✓ Ollama detected[/green]")
        else:
            console.print("[yellow]✗ Ollama not found[/yellow]")

        # Step 2: choose backend
        if has_ollama:
            choice = _ask(
                "Choose backend",
                ["ollama (local)", "custom API (OpenAI-compatible)"],
                default="ollama (local)",
            )
        else:
            console.print("[yellow]Only custom API is available (Ollama not detected)[/yellow]")
            choice = "custom API (OpenAI-compatible)"

        if "ollama" in choice.lower():
            self._wizard_ollama(console)
        else:
            self._wizard_custom_api(console)

        # Step 4: confirm skip behavior
        skip = _ask_yes_no("Skip confirmation for file writes?", default=False)
        self.config["skip_confirmation"] = skip

        self.save()
        console.print()
        console.print(f"[bold green]✓ Config saved to {CONFIG_FILE}[/bold green]")
        console.print(f"  Model: {self.config['model']}")
        console.print(f"  Backend: {self.config['backend']}")
        console.print()

    def _wizard_ollama(self, console):
        """Configure with Ollama."""
        models = _list_ollama_models()
        if not models:
            console.print("[yellow]No models found in Ollama. Pull one first: ollama pull gemma4[/yellow]")
            model = _ask_text("Enter model name manually", default="gemma4:12b-mlx")
            self.config["model"] = model
            self.config["backend"] = "ollama"
            self.config["api_base"] = "http://localhost:11434"
            return

        console.print("\n[bold]Available models:[/bold]")
        for i, m in enumerate(models, 1):
            console.print(f"  {i}. {m}")
        console.print(f"  {len(models) + 1}. Enter custom name")

        choice = _ask_int(
            f"Pick a model (1-{len(models) + 1})",
            default=1,
            min_val=1,
            max_val=len(models) + 1,
        )
        if 1 <= choice <= len(models):
            self.config["model"] = models[choice - 1]
        else:
            model = _ask_text("Enter model name", default="gemma4:12b-mlx")
            self.config["model"] = model

        self.config["backend"] = "ollama"
        self.config["api_base"] = "http://localhost:11434"

    def _wizard_custom_api(self, console):
        """Configure with custom OpenAI-compatible API."""
        console.print("\n[bold]OpenAI-compatible API setup[/bold]")
        console.print("Examples:")
        console.print("  - OpenAI:        https://api.openai.com/v1")
        console.print("  - OpenRouter:    https://openrouter.ai/api/v1")
        console.print("  - Together AI:   https://api.together.xyz/v1")
        console.print("  - Ollama:        http://localhost:11434")
        console.print()

        api_base = _ask_text("API base URL", default="http://localhost:11434/v1")
        api_key = _ask_text("API key (leave blank if not needed)", default="")
        model = _ask_text("Model name (e.g. gpt-4o-mini, gemma4:12b-mlx)", default="gpt-4o-mini")

        console.print("[dim]Testing connection...[/dim]")
        if _test_openai_api(api_base, api_key):
            console.print("[green]✓ Connection successful[/green]")
        else:
            console.print("[yellow]⚠ Connection failed — config saved but may not work[/yellow]")

        self.config["backend"] = "openai"
        self.config["api_base"] = api_base
        self.config["api_key"] = api_key
        self.config["model"] = model


# ─── Console helpers ─────────────────────────────────────────────────────

def _get_console():
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return _FallbackConsole()


class _FallbackConsole:
    def print(self, *args, **kwargs):
        text = " ".join(str(a) for a in args if not isinstance(a, str) or not a.startswith("["))
        print(text)

    def input(self, prompt=""):
        return input(prompt)


def _ask(prompt: str, options: list, default: str = None) -> str:
    print(f"\n{prompt}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    if default:
        print(f"  (default: {default})")
    while True:
        try:
            choice = input("> ").strip()
            if not choice and default:
                return default
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"Enter a number between 1 and {len(options)}")
        except (ValueError, EOFError):
            print("Invalid input")


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        try:
            answer = input(f"{prompt} ({hint}) ").strip().lower()
            if not answer:
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            print("Enter y or n")
        except EOFError:
            return default


def _ask_text(prompt: str, default: str = "") -> str:
    hint = f" (default: {default})" if default else ""
    try:
        value = input(f"{prompt}{hint} ").strip()
        return value if value else default
    except EOFError:
        return default


def _ask_int(prompt: str, default: int, min_val: int, max_val: int) -> int:
    while True:
        try:
            value = input(f"{prompt} (default: {default}) ").strip()
            if not value:
                return default
            n = int(value)
            if min_val <= n <= max_val:
                return n
            print(f"Enter a number between {min_val} and {max_val}")
        except (ValueError, EOFError):
            print("Invalid input")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Aether — autonomous CLI agent")
    parser.add_argument("--init", action="store_true", help="Run first-time setup")
    parser.add_argument("--model", "-m", type=str, help="Model name (overrides config)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmations")
    parser.add_argument("task", nargs="*", help="Task to run (optional)")

    args = parser.parse_args()

    # Init wizard
    cfg = AetherConfig()
    if args.init or not cfg.exists():
        if not args.init:
            print("First run detected — starting setup wizard")
        cfg.run_wizard()

    # Merge config with CLI flags
    if args.model:
        cfg.config["model"] = args.model
    if args.yes:
        cfg.config["skip_confirmation"] = True

    # Launch agent
    from aether import AetherAgent

    agent = AetherAgent(model=cfg.model)
    agent.skip_confirmation = cfg.config.get("skip_confirmation", False)

    if args.task:
        task = " ".join(args.task)
        agent.think(task)
    else:
        agent.run()


if __name__ == "__main__":
    main()
