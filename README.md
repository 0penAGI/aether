# Aether

Aether is an autonomous CLI agent for writing and refactoring code through Ollama.

```bash
aether "Write tests for parser.py"
aether --init          # first run: select an Ollama model
aether --help
```

## Features

- **Native tool calling** via Ollama (`ollama.chat(tools=...)`) — the model decides whether to read a file, write code, run a shell command, or search the codebase
- **Single execution path**: tool calling is primary, text-based fallback used only on API error
- **Diff-first editing**: targeted replacements with `[{old, new}]`; full rewrite only for new files or >50% changes
- **Action confirmation**: prompts before `write_file`, `edit_file`, `run_shell` (disable with `--yes`)
- **Project context**: AST-based parsing of `.py` files for import analysis and structure awareness
- **Simple chat detection**: short messages are treated as conversation without tool invocation
- **Bilingual prompts**: Russian and English
- **Smart memory**: logs actions to `aether_memory.json` with keyword search

## Install

```bash
git clone https://github.com/0penAGI/aether.git
cd aether
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./install.sh
```

After `install.sh`, the `aether` command is available globally (symlinked to `~/.local/bin`).

## Usage

```bash
# First run — pick your model
aether --init

# Run a task
aether "Create hello.py with a greeting function"

# Skip confirmation prompts
aether --yes "Refactor main.py, extract logic into separate functions"

# Override the saved model
aether --model qwen2.5-coder "Write unit tests"
```

### Flags

| Flag | Description |
|------|-------------|
| `--init` | First-time setup (model selection) |
| `--model`, `-m` | Ollama model override |
| `--yes`, `-y` | Skip all confirmations |
| `task` | Task to execute (positional argument) |

## Tests

```bash
# unit tests (52 tests)
pytest test_aether.py -v

# integration tests (require a running Ollama server)
pytest test_aether_integration.py -v

# all tests
pytest test_aether.py test_aether_integration.py -v
```

## Requirements

- [Ollama](https://ollama.ai) — running server with tool calling support
- Python 3.10+
- `ollama`, `rich`, `numpy`, `faiss`, `playwright`, `requests` (see `requirements.txt`)

## Structure

```
├── aether.py               # Core agent (2836 lines, 112 methods, 9 classes)
├── aether_config.py        # CLI entry, config management, first-run wizard
├── bin/aether              # Bash wrapper
├── test_aether.py          # 52 unit tests
├── test_aether_integration.py  # 13 integration tests
├── install.sh              # Install to ~/.local/bin
└── requirements.txt        # Dependencies
```

## License

MIT
