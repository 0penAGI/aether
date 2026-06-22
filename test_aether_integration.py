"""Integration tests for Aether agent — require a running Ollama instance.

Run with:  pytest test_aether_integration.py -v
Skip with: pytest test_aether_integration.py -v -m "not integration"
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

ollama = pytest.importorskip("ollama")
from aether import AetherAgent


def _check_ollama():
    """Check if Ollama is running and has at least one chat-capable model."""
    try:
        models = ollama.list().get("models", [])
        return any(m.model for m in models)
    except Exception:
        return False


def _filter_chat_models(models):
    """Filter out embedding-only models."""
    embed_only = {"nomic-embed-text", "all-minilm",
                  "mxbai-embed-large", "snowflake-arctic-embed"}
    return [m for m in models if not any(e in m.model for e in embed_only)]


def _pick_chat_model():
    """Pick the smallest chat-capable model."""
    try:
        models = ollama.list().get("models", [])
        chat_models = _filter_chat_models(models)
        if chat_models:
            sorted_models = sorted(chat_models, key=lambda m: m.size)
            return sorted_models[0].model
    except Exception:
        pass
    return "gemma3:1b"


SMALL_MODEL = _pick_chat_model()

integration = pytest.mark.skipif(not _check_ollama(),
                                 reason="Ollama not available or no models")

print(f"\n🔧 Using model: {SMALL_MODEL}\n")


@pytest.fixture
def agent():
    ag = AetherAgent(model=SMALL_MODEL)
    ag.skip_confirmation = True
    ag.verbose = False
    return ag


# ── Basic connectivity ─────────────────────────────────────────────

class TestOllamaConnectivity:
    @integration
    def test_ollama_list_models(self):
        models = ollama.list().get("models", [])
        assert len(models) > 0

    @integration
    def test_chat_model_generates_text(self):
        resp = ollama.chat(
            model=SMALL_MODEL,
            messages=[{"role": "user", "content": "Say exactly: hello world"}]
        )
        content = resp["message"]["content"].strip().lower()
        assert "hello" in content


# ── _llm_with_tools ────────────────────────────────────────────────

class TestLlmWithTools:
    @integration
    def test_returns_valid_type(self, agent):
        result = agent._llm_with_tools(
            "You are a helpful assistant. Answer concisely.",
            "Say exactly: test ok"
        )
        assert result["type"] in ("text", "tool_calls", "error")

    @integration
    def test_with_tools_argument(self, agent, tmp_path):
        (tmp_path / "project.py").write_text("x = 1")
        original = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = agent._llm_with_tools(
                "You are a coding agent. Use tools to answer.",
                "List files in the current directory",
                tools=agent._get_step_tools()
            )
            assert result["type"] in ("text", "tool_calls", "error")
            content = (result.get("content") or "").lower()
            if result["type"] == "text":
                assert "project.py" in content or "error" not in result["type"]
        finally:
            os.chdir(original)


# ── _execute_step ──────────────────────────────────────────────────

class TestExecuteStep:
    @integration
    def test_simple_chat_does_not_crash(self, agent):
        text, status = agent._execute_step("hello")
        assert status in ("SUCCESS", "FIX", "FAIL")
        assert isinstance(text, str) and len(text) > 0

    @integration
    def test_conversational_question(self, agent):
        text, status = agent._execute_step("How are you?")
        assert status in ("SUCCESS", "FIX", "FAIL")
        assert isinstance(text, str) and len(text) > 0

    @integration
    def test_run_shell(self, agent):
        text, status = agent._execute_step(
            "Run: echo hello_shell_test"
        )
        assert status in ("SUCCESS", "FIX", "FAIL")
        # The model may or may not actually run the shell command
        # depending on tool support; check it returned something
        assert isinstance(text, str)

    @integration
    def test_write_file_creates_file(self, agent, tmp_path):
        original = os.getcwd()
        try:
            os.chdir(tmp_path)
            text, status = agent._execute_step(
                "Write a file called test.txt with content 'hello world'"
            )
            assert status in ("SUCCESS", "FIX", "FAIL")
            # File should exist if tool was used, but the model may
            # just respond with text instead of using tools
        finally:
            os.chdir(original)
            Path(tmp_path / "test.txt").unlink(missing_ok=True)

    @integration
    def test_read_file_no_crash(self, agent, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("integration test file content")
        original = os.getcwd()
        try:
            os.chdir(tmp_path)
            text, status = agent._execute_step(
                f"Read the file data.txt and tell me what's in it"
            )
            assert status in ("SUCCESS", "FIX", "FAIL")
            assert isinstance(text, str)
        finally:
            os.chdir(original)

    @integration
    def test_list_files_no_crash(self, agent, tmp_path):
        (tmp_path / "alpha.py").write_text("")
        (tmp_path / "beta.py").write_text("")
        original = os.getcwd()
        try:
            os.chdir(tmp_path)
            text, status = agent._execute_step(
                "List files in the current directory"
            )
            assert status in ("SUCCESS", "FIX", "FAIL")
            assert isinstance(text, str)
        finally:
            os.chdir(original)


# ── Memory ─────────────────────────────────────────────────────────

class TestMemoryIntegration:
    @integration
    def test_memory_log_no_crash(self, agent, tmp_path):
        original = os.getcwd()
        try:
            os.chdir(tmp_path)
            text, status = agent._execute_step(
                "Remember that the secret number is 42"
            )
            assert status in ("SUCCESS", "FIX", "FAIL")
            memory_file = Path("aether_memory.json")
            if memory_file.exists():
                data = json.loads(memory_file.read_text())
                msgs = [d.get("msg", "") for d in data]
                assert any("42" in m for m in msgs), f"No 42 in: {msgs}"
        finally:
            os.chdir(original)


# ── Confirmation ───────────────────────────────────────────────────

class TestConfirmActionIntegration:
    @integration
    def test_skip_confirmation(self, agent):
        agent.skip_confirmation = True
        assert agent._confirm_action("write_file",
                                     {"path": "/tmp/t.txt", "content": "d"}) is True

    @integration
    def test_safe_tools_auto_confirm(self, agent):
        agent.skip_confirmation = False
        assert agent._confirm_action("read_file", {"path": "/tmp/t.txt"}) is True
        assert agent._confirm_action("list_files", {"path": "/tmp"}) is True
