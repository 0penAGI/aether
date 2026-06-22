"""Tests for Aether agent core logic."""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

# Patch ollama before import
import unittest.mock
sys.modules["ollama"] = unittest.mock.MagicMock()

from aether import (
    AetherAgent,
    _show_patch_preview,
    _backup_file,
    _pretty_size,
)


@pytest.fixture
def agent():
    """Create a fresh agent with mocked ollama."""
    with unittest.mock.patch("aether.ollama.chat") as mock_chat:
        mock_chat.return_value = {
            "message": {"content": "", "tool_calls": None}
        }
        agent = AetherAgent(model="test-model")
        agent.skip_confirmation = True
        yield agent


class TestPrettySize:
    def test_bytes(self):
        assert _pretty_size(500) == "500.0B"

    def test_kb(self):
        assert _pretty_size(2048) == "2.0KB"

    def test_mb(self):
        assert _pretty_size(2_621_440) == "2.5MB"


class TestBackupFile:
    def test_backup_creates_file(self, tmp_path):
        src = tmp_path / "test.py"
        src.write_text("x = 1")
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            _backup_file(str(src))
            backup_dir = tmp_path / ".aether_backups"
            assert backup_dir.exists()
            backups = list(backup_dir.iterdir())
            assert len(backups) >= 1
        finally:
            os.chdir(original_cwd)

    def test_backup_nonexistent_file(self, tmp_path):
        os.chdir(tmp_path)
        _backup_file("nonexistent.py")  # should not raise


class TestExtractTool:
    def test_extract_tool_simple(self, agent):
        text = "TOOL: read_file\nARGS: {\"path\": \"test.py\"}"
        name, args = agent._extract_tool(text)
        assert name == "read_file"
        assert args == {"path": "test.py"}

    def test_extract_tool_no_match(self, agent):
        name, args = agent._extract_tool("Just a regular response")
        assert name is None
        assert args == {}

    def test_extract_tool_write_file_path_content(self, agent):
        text = "TOOL: write_file\nPATH: test.py\nCONTENT:\nprint('hello')"
        name, args = agent._extract_tool(text)
        assert name == "write_file"
        assert args["path"] == "test.py"
        assert "print" in args["content"]

    def test_extract_tool_write_file_fenced(self, agent):
        text = "TOOL: write_file\nPATH: test.py\n```python\nprint('hello')\n```"
        name, args = agent._extract_tool(text)
        assert name == "write_file"
        assert args["path"] == "test.py"

    def test_extract_tool_with_markdown_fences(self, agent):
        text = "```\nTOOL: read_file\nARGS: {\"path\": \"foo.py\"}\n```"
        name, args = agent._extract_tool(text)
        assert name == "read_file"
        assert args == {"path": "foo.py"}

    def test_extract_tool_trailing_comma(self, agent):
        text = 'TOOL: read_file\nARGS: {"path": "foo.py",}'
        name, args = agent._extract_tool(text)
        assert name == "read_file"
        assert args == {"path": "foo.py"}

    def test_extract_tool_empty_args(self, agent):
        text = "TOOL: list_files"
        name, args = agent._extract_tool(text)
        assert name == "list_files"
        assert args == {}


class TestVerifier:
    def test_verify_success(self, agent):
        assert agent._verify("All good") == "RETRY"

    def test_verify_crash(self, agent):
        assert agent._verify("crashed unexpectedly") == "FIX"

    def test_verify_traceback(self, agent):
        assert agent._verify("Traceback (most recent call last):") == "FIX"

    def test_verify_nameerror(self, agent):
        assert agent._verify("NameError: name 'x' is not defined") == "FIX"

    def test_verify_syntaxerror(self, agent):
        assert agent._verify("SyntaxError: invalid syntax") == "FIX"

    def test_verify_success_markers(self, agent):
        assert agent._verify("synced") == "SUCCESS"

    def test_verify_error_markers(self, agent):
        assert agent._verify("Error: file not found") == "FAIL"

    def test_verify_none(self, agent):
        assert agent._verify(None) == "FAIL"


class TestReadFile:
    def test_read_existing_file(self, agent, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        content = agent.read_file(str(f))
        assert content == "hello world"

    def test_read_nonexistent_file(self, agent):
        content = agent.read_file("/nonexistent/path/file.txt")
        assert content.startswith("Error:")

    def test_read_file_sets_last_read(self, agent, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        agent.read_file(str(f))
        assert agent.last_read_file == str(f)


class TestWriteFile:
    def test_write_new_file(self, agent, tmp_path):
        path = str(tmp_path / "new.txt")
        result = agent.write_file(path, "hello")
        assert "written successfully" in result
        assert Path(path).read_text() == "hello"

    def test_write_creates_dirs(self, agent, tmp_path):
        path = str(tmp_path / "a" / "b" / "deep.txt")
        result = agent.write_file(path, "nested")
        assert "written successfully" in result

    def test_write_strips_fences(self, agent, tmp_path):
        path = str(tmp_path / "test.py")
        agent.write_file(path, "```python\nprint('hi')\n```")
        content = Path(path).read_text()
        assert "print" in content
        assert "```" not in content


class TestEditFile:
    def test_edit_replaces_text(self, agent, tmp_path):
        path = str(tmp_path / "edit.txt")
        Path(path).write_text("hello world")
        result = agent.edit_file(path=path, old="world", new="there")
        assert "Edited" in result
        assert Path(path).read_text() == "hello there"

    def test_edit_pattern_not_found(self, agent, tmp_path):
        path = str(tmp_path / "edit.txt")
        Path(path).write_text("hello world")
        result = agent.edit_file(path=path, old="nope", new="x")
        assert result.startswith("Error")

    def test_edit_missing_args(self, agent):
        result = agent.edit_file()
        assert result.startswith("Error")


class TestRunShell:
    def test_run_shell_success(self, agent):
        result = agent.run_shell("echo hello")
        assert "hello" in result
        assert "RETURN_CODE: 0" in result

    def test_run_shell_failure(self, agent):
        result = agent.run_shell("exit 42")
        assert "NON_ZERO_EXIT" in result
        assert "RETURN_CODE: 42" in result

    def test_run_shell_strips_venv(self, agent):
        stripped = agent.run_shell("source /path/venv/bin/activate && echo hi")
        assert "/activate" not in stripped if "Error" not in stripped else True


class TestListFiles:
    def test_list_files_dir(self, agent, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        listing = agent.list_files(str(tmp_path))
        assert "a.py" in listing
        assert "b.py" in listing


class TestMemoryLog:
    def test_memory_log_basic(self, agent, tmp_path):
        original = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = agent.memory_log(message="test entry")
            assert "logged" in result.lower()
            log_file = Path("aether_memory.json")
            assert log_file.exists()
            data = json.loads(log_file.read_text())
            assert any("test entry" in d.get("msg", "") for d in data)
        finally:
            os.chdir(original)


class TestChangeDir:
    def test_change_dir_valid(self, agent, tmp_path):
        result = agent.change_dir(str(tmp_path))
        assert str(tmp_path) in result
        assert os.getcwd() == str(tmp_path)

    def test_change_dir_invalid(self, agent):
        result = agent.change_dir("/nonexistent_path_12345")
        assert result.startswith("Error")


class TestStateMachine:
    def test_initial_state(self, agent):
        assert agent.state_machine.state.name == "IDLE"

    def test_valid_transition(self, agent):
        from aether import AgentState
        assert agent.state_machine.transition(AgentState.PLAN) is True
        assert agent.state_machine.state == AgentState.PLAN

    def test_invalid_transition(self, agent):
        from aether import AgentState
        assert agent.state_machine.transition(AgentState.DONE) is False

    def test_force_transition(self, agent):
        from aether import AgentState
        agent.state_machine.force(AgentState.SLEEPING)
        assert agent.state_machine.state == AgentState.SLEEPING

    def test_can_transition(self, agent):
        from aether import AgentState
        assert agent.state_machine.can_transition(AgentState.PLAN) is True
        assert agent.state_machine.can_transition(AgentState.DONE) is False


class TestScanProject:
    def test_scan_project_basic(self, agent, tmp_path):
        original = os.getcwd()
        try:
            os.chdir(tmp_path)
            (tmp_path / "main.py").write_text("import os\nimport sys\nx = 1")
            scan = agent._scan_project(str(tmp_path))
            assert "main.py" in scan
            assert "os" in scan
            assert "sys" in scan
        finally:
            os.chdir(original)


class TestValidateCodeResult:
    def test_non_python_file(self, agent, tmp_path):
        path = str(tmp_path / "test.txt")
        result = agent._validate_code_result("write_file", {"path": path}, "ok")
        assert result is None

    def test_python_compile_ok(self, agent, tmp_path):
        path = str(tmp_path / "test.py")
        Path(path).write_text("x = 1")
        result = agent._validate_code_result("write_file", {"path": path}, "ok")
        assert result is None

    def test_python_compile_error(self, agent, tmp_path):
        path = str(tmp_path / "bad.py")
        Path(path).write_text("x = ")
        result = agent._validate_code_result("write_file", {"path": path}, "ok")
        assert result is not None
        assert "FIX" in result


class TestHasCrashSignal:
    def test_no_crash(self, agent):
        assert not agent._has_crash_signal("all good")

    def test_traceback(self, agent):
        assert agent._has_crash_signal("Traceback (most recent call last)")

    def test_nameerror(self, agent):
        assert agent._has_crash_signal("NameError: x not defined")

    def test_non_zero_exit(self, agent):
        assert agent._has_crash_signal("non_zero_exit")

    def test_crashed_unexpectedly(self, agent):
        assert agent._has_crash_signal("crashed unexpectedly")


class TestConfirmAction:
    def test_skip_confirmation(self, agent):
        agent.skip_confirmation = True
        assert agent._confirm_action("write_file", {"path": "x.py", "content": ""}) is True

    def test_non_dangerous_tool_no_confirm(self, agent):
        agent.skip_confirmation = False
        assert agent._confirm_action("read_file", {"path": "x.py"}) is True
