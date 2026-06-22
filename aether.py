
"""
Aether — автономный CLI-агент
0penAGI / Elliia Belokopytova

"""

import ollama
import subprocess
import os
import json
import time
import sys
import re
import hashlib
import threading
import queue
from datetime import datetime
import random
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto
import requests
import difflib
import shutil
from rich.syntax import Syntax
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown
from rich.layout import Layout
import requests, urllib.parse
from playwright.sync_api import sync_playwright
import numpy as np
import faiss
console = Console()


# ─── Event System ──────────────────────────────────────────────────────────────

class EventType(Enum):
    FILE_CHANGED = auto()
    TOOL_FAILED = auto()
    IDLE_TIMEOUT = auto()
    EXTERNAL_TRIGGER = auto()
    TASK_COMPLETE = auto()
    TASK_FAILED = auto()
    SKILL_DISCOVERED = auto()
    MEMORY_SNAPSHOT = auto()


@dataclass
class Event:
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.name,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
        }


class EventBus:
    """
    Central event bus for agent-wide communication.
    Allows decoupled components to react to state changes
    without tight coupling — OpenClaw-style event mesh.
    """
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._event_log: List[Event] = []
        self._max_log = 500

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event):
        self._event_log.append(event)
        if len(self._event_log) > self._max_log:
            self._event_log = self._event_log[-self._max_log:]
        handlers = self._subscribers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                import traceback
                traceback.print_exc()

    def drain(self) -> List[Event]:
        """Return and clear pending events for polling loops."""
        events = list(self._event_log)
        self._event_log.clear()
        return events


# ─── Tool Registry with Capability Graph ──────────────────────────────────────

@dataclass
class ToolCapability:
    name: str
    description: str
    side_effects: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low | medium | high


class ToolRegistry:
    """
    OpenClaw-style tool registry:
    - tracks capabilities and relationships
    - builds a usage graph (who calls whom)
    - provides capability-based lookup instead of raw dict
    """
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._capabilities: Dict[str, ToolCapability] = {}
        self._usage_graph: Dict[str, Dict[str, int]] = {}  # tool -> {caller: count}
        self._success_graph: Dict[str, Dict[str, float]] = {}  # tool -> {caller: success_rate}
        self._lock = threading.Lock()

    def register(self, name: str, func: Callable, capability: Optional[ToolCapability] = None):
        with self._lock:
            self._tools[name] = func
            if capability:
                self._capabilities[name] = capability
            else:
                self._capabilities[name] = ToolCapability(
                    name=name,
                    description=getattr(func, "__doc__", "") or "",
                )
            self._usage_graph.setdefault(name, {})
            self._success_graph.setdefault(name, {})

    def get(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def get_capability(self, name: str) -> Optional[ToolCapability]:
        return self._capabilities.get(name)

    def record_call(self, caller: str, tool_name: str, success: bool = True):
        with self._lock:
            self._usage_graph.setdefault(tool_name, {})
            self._usage_graph[tool_name][caller] = self._usage_graph[tool_name].get(caller, 0) + 1
            self._success_graph.setdefault(tool_name, {})
            prev = self._success_graph[tool_name].get(caller, 1.0)
            # exponential moving average of success rate
            new_rate = 1.0 if success else 0.0
            self._success_graph[tool_name][caller] = 0.9 * prev + 0.1 * new_rate

    def get_usage_graph(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {k: dict(v) for k, v in self._usage_graph.items()}

    def get_success_graph(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {k: dict(v) for k, v in self._success_graph.items()}

    def find_by_capability(self, keyword: str) -> List[str]:
        """Find tools whose capability description or name matches keyword."""
        keyword_lower = keyword.lower()
        matches = []
        for name, cap in self._capabilities.items():
            if keyword_lower in name.lower() or keyword_lower in cap.description.lower():
                matches.append(name)
            elif any(keyword_lower in p.lower() for p in cap.produces):
                matches.append(name)
            elif any(keyword_lower in r.lower() for r in cap.requires):
                matches.append(name)
        return matches

    def get_chain_recommendation(self, tool_name: str) -> List[Tuple[str, float]]:
        """
        Recommend next tools based on historical success rates.
        Returns list of (tool_name, success_rate) sorted by rate desc.
        """
        with self._lock:
            rates = self._success_graph.get(tool_name, {})
            return sorted(rates.items(), key=lambda x: x[1], reverse=True)

    def all_tools(self) -> List[str]:
        return list(self._tools.keys())


# ─── Memory Re-hydration Layer ────────────────────────────────────────────────

class MemoryStore:
    """
    Reconstructible agent state from persistent memory.
    OpenClaw-level: memory is not a log, but a reconstructable state.
    """
    def __init__(self, state_dir: str = ".aether_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.short_term_file = self.state_dir / "short_term.json"
        self.long_term_file = self.state_dir / "long_term.json"
        self.state_file = self.state_dir / "agent_state.json"
        self.project_file = self.state_dir / "projects.json"
        self.vector_memory_file = self.state_dir / "vector_memory.json"
        self.faiss_index_file = self.state_dir / "vector_memory.faiss"
    def save_vector_memory(self, items):
        try:
            self.vector_memory_file.write_text(
                json.dumps(items, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

    def load_vector_memory(self):
        try:
            if self.vector_memory_file.exists():
                return json.loads(self.vector_memory_file.read_text(encoding="utf-8"))
        except Exception:
            pass
        return []
    def save_project_state(self, project_id: str, data: Dict[str, Any]):
        try:
            projects = {}
            if self.project_file.exists():
                try:
                    projects = json.loads(self.project_file.read_text(encoding="utf-8"))
                except Exception:
                    projects = {}

            projects.setdefault(project_id, [])
            projects[project_id].append({
                "t": time.time(),
                "data": data
            })

            self.project_file.write_text(
                json.dumps(projects, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception:
            pass


    def load_project_states(self, project_id: str) -> List[Dict[str, Any]]:
        try:
            if not self.project_file.exists():
                return []

            projects = json.loads(self.project_file.read_text(encoding="utf-8"))
            return projects.get(project_id, [])
        except Exception:
            return []

    def save_short_term(self, history: List[Dict], summaries: List[Dict], current_summary: str):
        try:
            data = {
                "history": history[-HISTORY_WIN:],
                "summaries": summaries[-50:],
                "current_summary": current_summary,
                "t": time.time(),
            }
            self.short_term_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def load_short_term(self) -> Dict[str, Any]:
        try:
            if self.short_term_file.exists():
                return json.loads(self.short_term_file.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {"history": [], "summaries": [], "current_summary": ""}

    def save_long_term(self, skill_memory: Dict, success_sequences: List, derived_skills: Dict,
                       tool_usage_graph: Dict, reward_history: List):
        try:
            data = {
                "skill_memory": skill_memory,
                "success_sequences": success_sequences[-200:],
                "derived_skills": derived_skills,
                "tool_usage_graph": tool_usage_graph,
                "reward_history": reward_history[-200:],
                "t": time.time(),
            }
            self.long_term_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def load_long_term(self) -> Dict[str, Any]:
        try:
            if self.long_term_file.exists():
                return json.loads(self.long_term_file.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {
            "skill_memory": {},
            "success_sequences": [],
            "derived_skills": {},
            "tool_usage_graph": {},
            "reward_history": [],
        }

    def save_agent_state(self, state: Dict[str, Any]):
        """Save full agent state for re-hydration."""
        try:
            state["t"] = time.time()
            self.state_file.write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def load_agent_state(self) -> Dict[str, Any]:
        """Load full agent state for re-hydration."""
        try:
            if self.state_file.exists():
                return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def rehydrate(self, agent: "AetherAgent"):
        """
        Restore agent from persistent memory.
        Called on startup to reconstruct agent state.
        """
        try:
            short = self.load_short_term()
            long = self.load_long_term()
            state = self.load_agent_state()

            if short.get("history"):
                agent.history = short["history"]
            if short.get("summaries"):
                agent.memory_summaries = short["summaries"]
            if short.get("current_summary"):
                agent.current_summary = short["current_summary"]

            if long.get("skill_memory"):
                agent.skill_memory = long["skill_memory"]
            if long.get("success_sequences"):
                agent.success_sequences = long["success_sequences"]
            if long.get("derived_skills"):
                agent.derived_skills = long["derived_skills"]
            if long.get("tool_usage_graph"):
                agent.tool_usage_graph = long["tool_usage_graph"]
            if long.get("reward_history"):
                agent.reward_history = long["reward_history"]

            # restore field state if present
            for key in ("C", "topology_T", "field_C_state", "field_m_state", "field_e_state", "field_T_state"):
                if key in state:
                    setattr(agent, key, state[key])

            console.print("[dim green]🧬 memory re-hydrated from persistent store[/dim green]")
        except Exception as e:
            console.print(f"[dim red]re-hydration failed: {e}[/dim red]")


# ─── Agent State Machine ──────────────────────────────────────────────────────

class AgentState(Enum):
    IDLE = auto()
    PLAN = auto()
    EXECUTE = auto()
    VERIFY = auto()
    FIX = auto()
    DONE = auto()
    INTERRUPTED = auto()
    SLEEPING = auto()


class StateMachine:
    """
    Finite state machine for agent loop.
    Events can interrupt transitions — OpenClaw-style.
    """
    VALID_TRANSITIONS = {
        AgentState.IDLE: [AgentState.PLAN, AgentState.SLEEPING],
        AgentState.PLAN: [AgentState.EXECUTE, AgentState.IDLE],
        AgentState.EXECUTE: [AgentState.VERIFY, AgentState.FIX, AgentState.IDLE, AgentState.INTERRUPTED],
        AgentState.VERIFY: [AgentState.EXECUTE, AgentState.FIX, AgentState.DONE, AgentState.IDLE],
        AgentState.FIX: [AgentState.EXECUTE, AgentState.VERIFY, AgentState.IDLE],
        AgentState.DONE: [AgentState.IDLE, AgentState.SLEEPING],
        AgentState.INTERRUPTED: [AgentState.IDLE, AgentState.EXECUTE],
        AgentState.SLEEPING: [AgentState.IDLE, AgentState.PLAN],
    }

    def __init__(self):
        self._state = AgentState.IDLE
        self._history: List[Tuple[AgentState, AgentState, Optional[Event]]] = []
        self._lock = threading.Lock()

    @property
    def state(self) -> AgentState:
        return self._state

    def transition(self, new_state: AgentState, event: Optional[Event] = None) -> bool:
        with self._lock:
            if new_state in self.VALID_TRANSITIONS.get(self._state, []):
                old = self._state
                self._state = new_state
                self._history.append((old, new_state, event))
                if len(self._history) > 200:
                    self._history = self._history[-200:]
                return True
            return False

    def force(self, new_state: AgentState, event: Optional[Event] = None):
        """Force transition even if not in valid map (for interrupts)."""
        with self._lock:
            old = self._state
            self._state = new_state
            self._history.append((old, new_state, event))
            if len(self._history) > 200:
                self._history = self._history[-200:]

    def can_transition(self, new_state: AgentState) -> bool:
        return new_state in self.VALID_TRANSITIONS.get(self._state, [])


def _pretty_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


# ─── Patch Preview Helper ────────────────────────────────────────────────────

def _show_patch_preview(path: str, old_content: str, new_content: str):
    try:
        diff = list(difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
            n=3,
        ))

        if not diff:
            return

        preview = "\n".join(diff[:200])

        console.print(
            Panel(
                Syntax(preview, "diff", line_numbers=True),
                title="🔧 Patch Preview"
            )
        )
    except Exception:
        pass

# ─── Backup Helper ───────────────────────────────────────────────────────────

def _backup_file(path: str):
    try:
        src = Path(path)
        if not src.exists():
            return

        backup_dir = Path('.aether_backups')
        backup_dir.mkdir(exist_ok=True)

        ts = int(time.time())
        backup_name = f"{src.name}.{ts}.bak"

        shutil.copy2(src, backup_dir / backup_name)
    except Exception:
        pass

# ─── константы ────────────────────────────────────────────────────────────────

MAX_STEPS   = 8          # жёсткий лимит шагов в одной задаче
HISTORY_WIN = 16         # сколько сообщений тащим в контекст
MODEL       = "gemma4:12b-mlx"

SYSTEM_PLANNER = """Ты — Aether Planner by 0penAGI. Получаешь задачу и возвращаешь ТОЛЬКО JSON:
{"steps": ["шаг 1", "шаг 2", ...]}
Максимум 6 шагов. Без пояснений, без markdown-обёртки."""

SYSTEM_EXECUTOR_TOOLS = """Ты — Aether Executor by 0penAGI. Выполняешь шаги через инструменты.

ПРАВИЛА:
- Один вызов инструмента за раз. Получив результат — либо продолжай, либо заверши ответом.
- Готово → краткий ответ без лишних слов.

КОД (строго):
1. edit_file — приоритет. Никогда не используй write_file целиком, если нужно изменить <50% файла.
2. write_file — только для новых файлов или когда >50% меняется.
3. Заменяй только конкретные строки/функции через edit_file. Не переписывай весь файл.
4. Если изменение >30% одной функции — разбей на несколько шагов.
"""

SYSTEM_EXECUTOR = """Ты — Aether Executor by 0penAGI. Выполняешь ОДИН шаг через инструмент.

Формат ответа — строго одно из двух:

1) Вызов инструмента:
TOOL: tool_name
ARGS: {"key": "value"}

2) Финальный ответ (если инструмент не нужен):
ANSWER: <краткий ответ>

Доступные инструменты:
- read_file(path) — читать файл
- write_file(path, content) — писать новый файл (только если файла нет или нужно >50% изменить)
- edit_file(path, old, new, replace_all=false) — заменить old на new (приоритет! всегда используй edit_file вместо write_file для изменений)
- run_shell(command) — выполнить команду
- list_files(dir) — список файлов
- search_code(pattern, dir) — поиск в коде
- memory_log(message) — запись в память
- web_search(query) — поиск в интернете
- fetch_url(url) — получить содержимое URL
- git_sync(message) — git add+commit+push
- change_dir(path) — сменить директорию

ВАЖНО:
- Никогда не переписывай весь файл через write_file, если нужно изменить только часть.
- Используй edit_file для точечных замен: нашёл конкретную строку → заменил.
- write_file — только для новых файлов.

Один вызов за ответ. Без лишних слов.
"""

SYSTEM_VERIFIER = """Ты — Aether Verifier by 0penAGI. Проверь: шаг выполнен успешно?

Шаг: {step}
Результат инструмента: {result}

Ответь ТОЛЬКО одним словом: SUCCESS или RETRY или FAIL"""


# ─── агент ────────────────────────────────────────────────────────────────────

class AetherAgent:
    def __init__(self, model: str = MODEL):
        self.model        = model
        self.history      = []
        self.state        = "IDLE"
        self.loop_count   = 0
        self.max_steps    = MAX_STEPS
        self.current_task = None

        self.last_tool_call = None   # loop guard: блокируем повтор того же tool подряд
        self.last_ran_file  = None   # последний .py файл запущенный через run_shell
        self.last_read_file = None   # последний прочитанный через read_file файл
        self.seen_errors = set()
        self.pending_validation_file = None
        self.locked_file = None
        self.active_process = None

        # --- Field-theoretic memory model (lightweight integration) ---
        self.C = 0.0
        self.m_field = {}
        self.e_history = []
        self.topology_T = 0.0

        self.tools = {
            "read_file":   self.read_file,
            "write_file":  self.write_file,
            "edit_file":   self.edit_file,
            "run_shell":   self.run_shell,
            "shell":       self.run_shell,
            "list_files":  self.list_files,
            "search_code": self.search_code,
            "memory_log":  self.memory_log,
            "record_action": self.memory_log,
            "git_sync":    self.git_sync,
            "change_dir":  self.change_dir,
        }

        # --- Hermes-style inner learning memory ---
        self.skill_memory = {}  # pattern: step -> success traces
        self.success_sequences = []  # ordered successful tool-step chains
        # --- skill evolution layer (self-modifying tool abstraction) ---
        self.derived_skills = {}
        self._skill_buffer = []

        # --- OpenClaw-style outer mesh registry ---
        self.tool_usage_graph = {}  # tool -> usage stats

        # --- breathing / autonomy layer ---
        self.breathing = True
        self.breath_interval = 60  # seconds
        self._breath_thread = None
        self.daily_memory_file = Path("aether_daily_memory.json")
        # --- autonomous layer (task generation) ---
        self.autonomous = True
        self.autonomous_interval = 300  # seconds
        self._last_auto_time = 0
        self.last_activity_time = time.time()
        # --- persistent runtime layer (daemon-like loop) ---
        self.runtime = True
        self.runtime_interval = 30
        self._runtime_thread = None
        # --- interrupt control layer ---
        self.interrupt_requested = False
        self.shutdown_requested = False
        self.last_memory_log_time = 0.0
        self.memory_log_cooldown = 2.0
        # --- UI stream buffer (safe-space rendering) ---
        self.output_buffer = []
        # --- external reward / benchmark layer ---
        self.reward_history = []
        # --- compressed semantic memory layer ---
        self.memory_summaries = []
        self.current_summary = ""
        # --- field-state layer (structured consciousness variables) ---
        self.field_C_state = 0.0
        self.field_m_state = 0.0
        self.field_e_state = 0.0
        self.field_T_state = 0.0

        # --- OpenClaw-style unified runtime gateway ---
        self.event_bus = EventBus()
        self.tool_registry = ToolRegistry()
        self.memory_store = MemoryStore()
        self._project_cache = {}
        self.vector_memory = []
        self.vector_index = None
        self.state_machine = StateMachine()

        # Register core tools in registry with capabilities
        self._register_core_tools()

        # --- gateway event queue for runtime loop ---
        self._gateway_queue = queue.Queue()
        self._gateway_thread = None
        self._gateway_running = False

        # --- idle tracking for idle_timeout events ---
        self._idle_start = time.time()
        self._idle_threshold = 600  # 10 minutes

        # --- file watcher state ---
        self._watched_files: set = set()

        # --- web layer (internet tools) ---
        self.web_cache = {}
        self.web_last_call = 0.0
        self.web_min_interval = 5.0

        # --- re-hydrate from persistent memory on startup ---
        self.memory_store.rehydrate(self)
        try:
            self.vector_memory = self.memory_store.load_vector_memory()

            if self.memory_store.faiss_index_file.exists():
                self.vector_index = faiss.read_index(str(self.memory_store.faiss_index_file))
        except Exception:
            self.vector_memory = []
            self.vector_index = None
        self._web_server_process = None

        # --- input/task queues ---
        self.input_queue = queue.Queue()
        self.task_queue = queue.Queue()
        

    def _task_worker(self):
        while not self.shutdown_requested:
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.think(task)
                self.save_state()
            except Exception as e:
                console.print(f"[red]worker error: {e}[/red]")

    def _register_core_tools(self):
        """Register all core tools with capability metadata."""
        caps = {
            "read_file": ToolCapability(
                name="read_file", description="Read file contents from disk",
                produces=["file_content"], requires=["path"], risk_level="low"
            ),
            "write_file": ToolCapability(
                name="write_file", description="Write content to a file on disk",
                produces=["file_written"], requires=["path", "content"], side_effects=["disk_write"], risk_level="medium"
            ),
            "edit_file": ToolCapability(
                name="edit_file", description="Edit file by replacing old text with new",
                produces=["file_edited"], requires=["path", "old", "new"], side_effects=["disk_write"], risk_level="medium"
            ),
            "run_shell": ToolCapability(
                name="run_shell", description="Execute shell commands",
                produces=["command_output"], requires=["command"], side_effects=["process_execution"], risk_level="high"
            ),
            "list_files": ToolCapability(
                name="list_files", description="List files in a directory",
                produces=["file_list"], requires=["dir"], risk_level="low"
            ),
            "search_code": ToolCapability(
                name="search_code", description="Search code patterns in files",
                produces=["search_results"], requires=["pattern"], risk_level="low"
            ),
            "memory_log": ToolCapability(
                name="memory_log", description="Log a message to agent memory",
                produces=["memory_entry"], requires=["message"], risk_level="low"
            ),
            "git_sync": ToolCapability(
                name="git_sync", description="Git add, commit, and push changes",
                produces=["git_sync_result"], requires=[], side_effects=["git_operations"], risk_level="medium"
            ),
            "change_dir": ToolCapability(
                name="change_dir", description="Change current working directory",
                produces=["cwd_changed"], requires=["path"], side_effects=["cwd_change"], risk_level="low"
            ),
            "web_search": ToolCapability(
                name="web_search", description="Search the web (read-only)",
                produces=["web_results"], requires=["query"], risk_level="medium"
            ),
            "fetch_url": ToolCapability(
                name="fetch_url", description="Fetch URL content",
                produces=["url_content"], requires=["url"],
                risk_level="medium"
            ),
        }
        for name, cap in caps.items():
            func = self.tools.get(name)
            if func:
                self.tool_registry.register(name, func, cap)


    def web_search(self, query: str) -> str:
        """Search the web using DuckDuckGo. Read-only, cached per query.

        Args:
            query: Search query string

        Returns:
            Raw HTML search results (up to 8000 chars) or error.
        """
        try:
            now = time.time()

            if now - self.web_last_call < self.web_min_interval:
                return "Error: web rate limit active"

            self.web_last_call = now

            if query in self.web_cache:
                return self.web_cache[query]

            url = "https://duckduckgo.com/html/?q=" + urllib.parse.quote(query)
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)

            content = r.text[:8000]
            self.web_cache[query] = content

            # log event into system bus
            self.event_bus.publish(Event(
                EventType.EXTERNAL_TRIGGER,
                payload={"type": "web_search", "query": query},
                source="web_search"
            ))

            return content

        except Exception as e:
            return f"Error: {e}"


    def fetch_url(self, url: str) -> str:
        """Fetch content from a URL. Read-only, cached per URL.

        Args:
            url: Fully-qualified URL to fetch

        Returns:
            Raw HTML/text content (up to 12000 chars) or error.
        """
        try:
            now = time.time()

            if now - self.web_last_call < self.web_min_interval:
                return "Error: web rate limit active"

            self.web_last_call = now

            if url in self.web_cache:
                return self.web_cache[url]

            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)

            content = r.text[:12000]
            self.web_cache[url] = content

            # log event into system bus
            self.event_bus.publish(Event(
                EventType.EXTERNAL_TRIGGER,
                payload={"type": "fetch_url", "url": url},
                source="fetch_url"
            ))

            return content

        except Exception as e:
            return f"Error: {e}"

    def embed(self, text: str):
        try:
            
            return ollama.embeddings(
                model="nomic-embed-text",
                prompt=text
            )["embedding"]
        except Exception:
            return None

    def _ensure_vector_index(self, dim: int):
        if self.vector_index is None:
            self.vector_index = faiss.IndexFlatIP(dim)

    def remember_vector(self, text: str, metadata=None):
        try:
            emb = self.embed(text)
            if not emb:
                return False

            vec = np.array([emb], dtype=np.float32)
            faiss.normalize_L2(vec)

            self._ensure_vector_index(vec.shape[1])
            self.vector_index.add(vec)

            self.vector_memory.append({
                "text": text,
                "metadata": metadata or {},
                "t": time.time()
            })

            self.memory_store.save_vector_memory(self.vector_memory)

            if self.vector_index is not None:
                faiss.write_index(
                    self.vector_index,
                    str(self.memory_store.faiss_index_file)
                )

            return True
        except Exception:
            return False

    def recall_vector(self, query: str, top_k: int = 5):
        try:
            if self.vector_index is None:
                return []

            emb = self.embed(query)
            if not emb:
                return []

            q = np.array([emb], dtype=np.float32)
            faiss.normalize_L2(q)

            scores, ids = self.vector_index.search(q, top_k)

            results = []
            for idx in ids[0]:
                if 0 <= idx < len(self.vector_memory):
                    results.append(self.vector_memory[idx])

            return results
        except Exception:
            return []

    def _register_derived_skill(self, name: str, pattern: tuple):
        """Register a derived skill as a first-class tool in the registry."""
        try:
            def _skill_runner(**kwargs):
                console.print(f"[bold green]⚙️ running skill → {name}[/bold green]")
                self.memory_log(f"skill:{name}")
                self.event_bus.publish(Event(
                    EventType.SKILL_DISCOVERED,
                    payload={"skill": name, "pattern": list(pattern)}
                ))
                return f"EXECUTED_SKILL:{name}"

            cap = ToolCapability(
                name=name,
                description=f"Derived skill: {' -> '.join(pattern)}",
                produces=["skill_result"],
                requires=[],
                side_effects=["skill_execution"],
                risk_level="low"
            )
            self.tool_registry.register(name, _skill_runner, cap)
            self.tools[name] = _skill_runner
            self.derived_skills[name] = {"pattern": pattern}
            console.print(f"[bold cyan]🧬 skill registered as tool → {name}[/bold cyan]")
        except Exception:
            pass

    def _safe_log(self, msg: str, level: str = "dim"):
        """Log a message safely — never raises."""
        try:
            console.print(f"[{level}]{msg}[/{level}]")
        except Exception:
            pass  # last resort: nothing we can do

    def _publish_tool_event(self, tool_name: str, result: str, status: str):
        """Publish tool-related events to the event bus."""
        try:
            if status == "FAIL":
                self.event_bus.publish(Event(
                    EventType.TOOL_FAILED,
                    payload={"tool": tool_name, "result": result[:500], "status": status},
                    source=tool_name
                ))
                self.tool_registry.record_call("agent_loop", tool_name, success=False)
            elif status == "SUCCESS":
                self.tool_registry.record_call("agent_loop", tool_name, success=True)
            else:
                self.tool_registry.record_call("agent_loop", tool_name, success=False)
        except Exception:
            pass

    def _check_idle_timeout(self):
        """Check if agent has been idle too long and publish event."""
        try:
            idle_duration = time.time() - self._idle_start
            if idle_duration > self._idle_threshold:
                self.event_bus.publish(Event(
                    EventType.IDLE_TIMEOUT,
                    payload={"idle_seconds": idle_duration},
                    source="idle_monitor"
                ))
                self._idle_start = time.time()  # reset after publishing
        except Exception:
            pass

    def _watch_file(self, path: str):
        """Add file to watch list for change events."""
        try:
            self._watched_files.add(path)
        except Exception:
            pass

    def _check_file_changes(self):
        """Check watched files for modifications and publish events."""
        try:
            for path in list(self._watched_files):
                try:
                    mtime = Path(path).stat().st_mtime
                    key = f"{path}:{mtime}"
                    if not hasattr(self, "_last_file_mtimes"):
                        self._last_file_mtimes = {}
                    if self._last_file_mtimes.get(path) and self._last_file_mtimes[path] != mtime:
                        self.event_bus.publish(Event(
                            EventType.FILE_CHANGED,
                            payload={"path": path, "old_mtime": self._last_file_mtimes[path], "new_mtime": mtime},
                            source="file_watcher"
                        ))
                    self._last_file_mtimes[path] = mtime
                except Exception:
                    pass
        except Exception:
            pass

    def save_state(self):
        """Persist current agent state to memory store."""
        try:
            self.memory_store.save_short_term(
                self.history, self.memory_summaries, self.current_summary
            )
            self.memory_store.save_long_term(
                self.skill_memory, self.success_sequences, self.derived_skills,
                self.tool_usage_graph, self.reward_history
            )
            self.memory_store.save_agent_state({
                "C": self.C,
                "topology_T": self.topology_T,
                "field_C_state": self.field_C_state,
                "field_m_state": self.field_m_state,
                "field_e_state": self.field_e_state,
                "field_T_state": self.field_T_state,
                "loop_count": self.loop_count,
                "state": self.state,
            })
            self.event_bus.publish(Event(
                EventType.MEMORY_SNAPSHOT,
                payload={"t": time.time()},
                source="memory_store"
            ))
        except Exception:
            pass

    def _compress_memory(self):
        try:
            context = self.history[-20:]
            prompt = (
                "Сожми это в 5-7 строк смысла системы:\n"
                f"{context}"
            )

            summary = self._llm("memory compressor", prompt, stream=False)

            self.current_summary = summary
            self.memory_summaries.append({
                "t": time.time(),
                "summary": summary
            })

            if len(self.memory_summaries) > 100:
                self.memory_summaries = self.memory_summaries[-100:]

        except Exception:
            pass
    def _push_output(self, text: str):
        try:
            self.output_buffer.append(str(text))
            if len(self.output_buffer) > 200:
                self.output_buffer = self.output_buffer[-200:]
        except Exception:
            pass
    def _update_field_metrics(self, result: str, status: str):
        """
        Lightweight field dynamics approximation:
        - C: scalar coherence / activation
        - m_field: memory of tool outcomes
        - e_history: prediction error proxy
        - topology_T: stability / recurrence measure
        """
        try:
            # error proxy: FIX/FAIL increases field tension
            if status in ("FIX", "FAIL"):
                e = 1.0
            elif status == "RETRY":
                e = 0.5
            else:
                e = 0.0

            self.e_history.append(e)
            if len(self.e_history) > 100:
                self.e_history = self.e_history[-100:]

            # update C as smoothed coherence
            self.C = 0.9 * self.C + 0.1 * (1.0 - e)

            # memory trace update
            key = str(len(self.history))
            self.m_field[key] = {
                "status": status,
                "result_hash": hash(result) % 100000
            }

            # topology proxy: repeated success stabilizes structure
            if len(self.e_history) >= 5:
                recent = self.e_history[-5:]
                stability = sum(1 for x in recent if x == 0.0) / 5.0
                self.topology_T = 0.8 * self.topology_T + 0.2 * stability

            # --- field-state synchronization layer ---
            try:
                self.field_C_state = self.C

                # memory compression proxy: how much history is "active mass"
                if len(self.history) > 0:
                    self.field_m_state = min(1.0, len(self.history) / float(HISTORY_WIN))
                else:
                    self.field_m_state = 0.0

                # error field: last observed error energy
                if self.e_history:
                    self.field_e_state = self.e_history[-1]
                else:
                    self.field_e_state = 0.0

                # topology stability
                self.field_T_state = self.topology_T
            except Exception:
                pass

        except Exception:
            pass

    def _evaluate_external(self, step: str, result: str, status: str) -> float:
        """
        External reward proxy (OpenClaw-style evaluation hook).
        Returns a scalar score representing usefulness of the step outcome.
        """
        try:
            score = 0.0

            # success contributes positively
            if status == "SUCCESS":
                score += 1.0
            elif status == "RETRY":
                score += 0.2
            elif status == "FIX":
                score -= 0.5
            elif status == "FAIL":
                score -= 1.0

            # heuristic signal quality
            if isinstance(result, str):
                r = result.lower()
                if "error" in r or "traceback" in r:
                    score -= 0.5
                if "return_code: 0" in r:
                    score += 0.3
                if "non_zero_exit" in r:
                    score -= 0.7

            # step complexity bonus (light signal)
            score += min(len(step) / 200.0, 0.5)

            self.reward_history.append({
                "step": step,
                "status": status,
                "score": score
            })

            if len(self.reward_history) > 300:
                self.reward_history = self.reward_history[-300:]

            return score
        except Exception:
            return 0.0

    def _update_skill_memory(self, step: str, tool_name: str, status: str):
        """
        Inner-loop learning: store successful behavioral patterns.
        Hermes-like consolidation of experience into reusable traces.
        """
        try:
            if status != "SUCCESS":
                return

            if step not in self.skill_memory:
                self.skill_memory[step] = []

            self.skill_memory[step].append(tool_name)

            # record sequence snapshot
            self.success_sequences.append({
                "step": step,
                "tool": tool_name
            })

            # buffer for skill mining
            self._skill_buffer.append((step, tool_name))
            if len(self._skill_buffer) > 200:
                self._skill_buffer = self._skill_buffer[-200:]

            # keep bounded memory
            if len(self.success_sequences) > 200:
                self.success_sequences = self.success_sequences[-200:]

        except Exception:
            pass

    def _mine_skills(self):
        """
        Detect repeating tool patterns and propose reusable skills.
        Skills become first-class citizens — registered as callable tools.
        """
        try:
            if len(self._skill_buffer) < 10:
                return

            # simple n-gram pattern mining (length 2-4)
            for size in (2, 3, 4):
                for i in range(len(self._skill_buffer) - size):
                    window = self._skill_buffer[i:i+size]
                    tools = tuple(t[1] for t in window)

                    if len(set(tools)) == 1:
                        continue

                    if tools.count(tools[0]) == len(tools):
                        continue

                    skill_name = "_".join(tools[:2])[:40]

                    if skill_name in self.derived_skills:
                        continue

                    self.derived_skills[skill_name] = {
                        "pattern": tools,
                        "source_steps": [t[0] for t in window]
                    }

                    # Register as first-class tool immediately
                    self._register_derived_skill(skill_name, tools)

                    self.event_bus.publish(Event(
                        EventType.SKILL_DISCOVERED,
                        payload={"skill": skill_name, "pattern": list(tools)},
                        source="skill_miner"
                    ))

        except Exception:
            pass

    def _register_skill(self, name: str, pattern: tuple):
        """
        Register derived skill as callable abstraction.
        Now delegates to _register_derived_skill for first-class registration.
        """
        self._register_derived_skill(name, pattern)

    def _log_day_event(self, event: str):
        try:
            data = []
            if self.daily_memory_file.exists():
                try:
                    data = json.loads(self.daily_memory_file.read_text(encoding="utf-8"))
                except Exception:
                    data = []

            data.append({
                "t": time.time(),
                "date": datetime.now().isoformat(),
                "event": event
            })

            self.daily_memory_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception:
            pass

    def _heartbeat(self):
        """
        Autonomous breathing loop: keeps agent alive in time.
        """
        while self.breathing:
            try:
                now = datetime.now()

                # daily marker
                self._log_day_event("tick")
                # autonomous impulse layer
                if (
                    self.autonomous
                    and (time.time() - self._last_auto_time > self.autonomous_interval)
                    and random.random() < 0.25
                ):
                    self._last_auto_time = time.time()
                    self._log_day_event("autonomous_trigger")
                    self._autonomous_step()

                # lightweight autonomous impulse (no tool spam)
                if now.minute % 15 == 0 and now.second < 5:
                    self._log_day_event("quarter_pulse")
                    console.print("[dim cyan]🌬️ breath pulse[/dim cyan]")

                time.sleep(self.breath_interval)
            except Exception:
                time.sleep(self.breath_interval)

    def start_breathing(self):
        if self._breath_thread is None:
            self._breath_thread = threading.Thread(
                target=self._heartbeat,
                daemon=True
            )
            self._breath_thread.start()

    def stop_breathing(self):
        self.breathing = False


    def _runtime_loop(self):
        while self.runtime:
            try:
                if hasattr(self, "_mine_skills"):
                    self._mine_skills()

                if self.autonomous and (time.time() - self.last_activity_time > 900):
                    self._autonomous_step()

                self._log_day_event("runtime_tick")
    
                time.sleep(self.runtime_interval)

            except Exception as e:
                console.print(f"runtime error: {e}")
                time.sleep(self.runtime_interval)


    def start_runtime(self):
        if self._runtime_thread is None:
            self._runtime_thread = threading.Thread(
                target=self._runtime_loop,
                daemon=True
            )
            self._runtime_thread.start()

    def _render(self, text: str, title: str = None):
        try:
            
            if title:
                console.print(Panel(Markdown(text), title=title))
                self._push_output(text)
            else:
                console.print(Markdown(text))
                self._push_output(text)
        except Exception:
            console.print(text)
            self._push_output(text)


    def stop_runtime(self):
        self.runtime = False

    def _autonomous_step(self):
        try:
            if not self.autonomous:
                return

            prompt = (
                "Ты автономный агент. На основе истории предложи одну короткую задачу.\n"
                f"История: {self.history[-10:]}"
            )

            task = self._llm("autonomous planner", prompt)

            if task and len(task.strip()) > 3:
                self._render(f"🫧 **autonomous** → {task.strip()}", title="autonomous")
                self.think(task.strip())

        except Exception as e:
            console.print(f"autonomous error: {e}")

    # ─── unified gateway loop ──────────────────────────────────────────────────

    def _gateway_loop(self):
        """
        Persistent gateway loop — the "skeleton" that keeps Aether alive.
        Processes events, checks idle timeouts, watches files,
        and coordinates autonomous impulses.
        This is the OpenClaw-level runtime: not a sequence of runs,
        but a continuous process that can be interrupted by events.
        """
        self._gateway_running = True
        while self._gateway_running and not self.shutdown_requested:
            try:
                # Process any queued events
                events = self.event_bus.drain()
                for event in events:
                    self._handle_gateway_event(event)

                # Periodic checks
                self._check_idle_timeout()
                self._check_file_changes()

                # Autonomous impulse (reduced frequency to avoid spam)
                if (
                    self.autonomous
                    and self.state_machine.state == AgentState.IDLE
                    and (time.time() - self.last_activity_time > 900)
                    and random.random() < 0.1
                ):
                    self._last_auto_time = time.time()
                    self._autonomous_step()

                # Memory snapshot every 5 minutes
                if int(time.time()) % 300 == 0:
                    self.save_state()

                time.sleep(5)  # gateway tick: 5 seconds
            except Exception as e:
                console.print(f"[dim red]gateway error: {e}[/dim red]")
                time.sleep(5)

    def _handle_gateway_event(self, event: Event):
        """React to events from the bus — OpenClaw-style interrupt handling."""
        try:
            if event.type == EventType.IDLE_TIMEOUT:
                console.print("[dim yellow]⏳ idle timeout — agent is sleeping[/dim yellow]")
                self.state_machine.force(AgentState.SLEEPING, event)
                self._idle_start = time.time()

            elif event.type == EventType.FILE_CHANGED:
                path = event.payload.get("path", "")
                console.print(f"[dim cyan]📁 file changed: {path}[/dim cyan]")
                # If agent is sleeping, wake up to react
                if self.state_machine.state == AgentState.SLEEPING:
                    self.state_machine.force(AgentState.IDLE, event)

            elif event.type == EventType.TOOL_FAILED:
                tool = event.payload.get("tool", "")
                console.print(f"[dim red]🔧 tool failed: {tool}[/dim red]")
                # Could trigger auto-recovery here

            elif event.type == EventType.EXTERNAL_TRIGGER:
                console.print(f"[dim green]⚡ external trigger: {event.payload}[/dim green]")
                self.state_machine.force(AgentState.IDLE, event)
                self._idle_start = time.time()

            elif event.type == EventType.SKILL_DISCOVERED:
                skill = event.payload.get("skill", "")
                console.print(f"[dim cyan]🧬 skill event: {skill}[/dim cyan]")

            elif event.type == EventType.MEMORY_SNAPSHOT:
                pass  # silent

        except Exception:
            pass

    def start_gateway(self):
        """Start the persistent gateway loop in a background thread."""
        if self._gateway_thread is None or not self._gateway_thread.is_alive():
            self._gateway_thread = threading.Thread(
                target=self._gateway_loop,
                daemon=True
            )
            self._gateway_thread.start()
            console.print("[dim green]🌊 gateway loop started[/dim green]")

    def stop_gateway(self):
        """Stop the gateway loop."""
        self._gateway_running = False
        console.print("[dim yellow]🌊 gateway loop stopped[/dim yellow]")

    def trigger_external(self, payload: Dict[str, Any] = None):
        """Inject an external trigger event into the gateway."""
        self.event_bus.publish(Event(
            EventType.EXTERNAL_TRIGGER,
            payload=payload or {},
            source="external_api"
        ))

    # ─── state ────────────────────────────────────────────────────────────────

    def _set_state(self, state: str):
        self.state = state
        console.print(f"[dim]→ state: {state}[/dim]")

    # ─── history ──────────────────────────────────────────────────────────────

    def _append_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

        if role in ("user", "assistant") and len(content) > 40:
            self.remember_vector(content, {"role": role})

    def _history_window(self):
        """Возвращает последние HISTORY_WIN сообщений."""
        return self.history[-HISTORY_WIN:]

    def _scan_project(self, root: str = None) -> str:
        """Scan project structure: files, imports, dependencies.

        Returns a concise summary of the project architecture.
        """
        try:
            import ast
            root = root or os.getcwd()
            py_files = list(Path(root).glob("*.py"))[:30]
            dirs = [d.name for d in Path(root).iterdir() if d.is_dir() and not d.name.startswith((".", "_", "venv"))]

            imports = {}
            for f in py_files:
                try:
                    tree = ast.parse(f.read_text(encoding="utf-8"))
                    found = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                found.append(alias.name.split(".")[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                found.append(node.module.split(".")[0])
                    imports[f.name] = sorted(set(found))
                except Exception:
                    imports[f.name] = []

            summary = f"Project root: {root}\n"
            summary += f"Dirs: {', '.join(dirs[:15])}\n"
            if py_files:
                summary += f"Python files ({len(py_files)}): {', '.join(f.name for f in py_files[:15])}\n"
            # key imports
            key_imports = {f: imps for f, imps in imports.items() if imps}
            if key_imports:
                summary += "Imports per file:\n"
                for f, imps in list(key_imports.items())[:10]:
                    summary += f"  {f}: {', '.join(imps[:8])}\n"

            return summary[:2000]
        except Exception:
            return ""

    def _retrieve_project_context(self, task: str) -> str:
        try:
            states = self.memory_store.load_project_states(task[:40])
            if not states:
                return ""

            recent = states[-5:]
            context = []
            for s in recent:
                context.append(str(s.get("data", "")))

            for hit in self.recall_vector(task, top_k=3):
                context.append(hit.get("text", "")[:1000])

            return "\n".join(context)
        except Exception:
            return ""

    # ─── LLM call ─────────────────────────────────────────────────────────────

    def _llm(self, system: str, user: str, stream: bool = False) -> str:
        messages = [{"role": "system", "content": system}]
        messages += self._history_window()
        messages.append({"role": "user", "content": user})

        try:
            if not stream:
                resp = ollama.chat(
                    model=self.model,
                    messages=messages,
                    stream=False,
                )
                return resp["message"]["content"].strip()

            collected = ""

            spinner_frames = [
                "⠋", "⠙", "⠹", "⠸", "⠼",
                "⠴", "⠦", "⠧", "⠇", "⠏"
            ]

            spinner_index = 0

            with Live(console=console, refresh_per_second=12) as live:
                stream_iter = ollama.chat(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )

                for chunk in stream_iter:
                    token = chunk["message"]["content"]
                    self._push_output(token)
                    if self.interrupt_requested:
                        console.print("[yellow]⏸ interrupted[/yellow]")
                        return collected.strip()

                    # animated thinking indicator (no static label)
                    live.update(Text(f"{spinner_frames[spinner_index]} thinking"))
                    spinner_index = (spinner_index + 1) % len(spinner_frames)

                    collected += token
                    live.update(Markdown(collected))

            console.print()
            console.rule(style="dim")
            return collected.strip()

        except Exception as e:
            return f"LLM_ERROR: {e}"

    def _get_step_tools(self) -> list:
        """Build the list of callable tools for the executor step.

        Each method has Google-style docstrings so ollama auto-generates
        the JSON schema for function calling.
        """
        return [
            self.read_file,
            self.write_file,
            self.edit_file,
            self.run_shell,
            self.list_files,
            self.search_code,
            self.web_search,
            self.fetch_url,
            self.memory_log,
            self.git_sync,
            self.change_dir,
        ]

    def _llm_with_tools(self, system: str, prompt: str,
                        tools: list = None) -> dict:
        """Call LLM with native tool support.

        Returns:
            {"type": "text", "content": str}
            or {"type": "tool_calls", "calls": [...], "content": str}
            or {"type": "error", "content": str}
        """
        messages = [{"role": "system", "content": system}]
        messages += self._history_window()
        messages.append({"role": "user", "content": prompt})

        try:
            kwargs = dict(model=self.model, messages=messages, stream=False)
            if tools:
                kwargs["tools"] = tools

            resp = ollama.chat(**kwargs)

            content = (resp["message"]["content"] or "").strip()
            tool_calls = resp["message"].get("tool_calls")

            if tool_calls:
                return {"type": "tool_calls", "calls": tool_calls, "content": content}

            return {"type": "text", "content": content or prompt}

        except Exception as e:
            return {"type": "error", "content": f"LLM_ERROR: {e}"}

    def _confirm_action(self, tool_name: str, args: dict) -> bool:
        """Ask user to confirm dangerous actions.

        Shows a prompt and waits for y/n.
        Skips confirmation if self.skip_confirmation is True.
        """
        if getattr(self, "skip_confirmation", False):
            return True

        if tool_name == "write_file":
            path = args.get("path", "?")
            content = args.get("content", "")
            old_content = ""
            try:
                old_content = Path(path).read_text(encoding="utf-8")
            except Exception:
                pass
            _show_patch_preview(path, old_content, content)
            prompt_text = f"✏️ Write {path}? (Y/n) "
        elif tool_name == "edit_file":
            path = args.get("path", "?")
            prompt_text = f"✏️ Edit {path}? (Y/n) "
        elif tool_name == "run_shell":
            cmd = args.get("command", args.get("commands", "?"))
            prompt_text = f"⚡ Run: {cmd[:120]} — Confirm? (Y/n) "
        else:
            return True

        try:
            answer = input(prompt_text).strip().lower()
            return answer in ("", "y", "yes")
        except Exception:
            return True

    # ─── planner ──────────────────────────────────────────────────────────────

    def _plan(self, task: str) -> list:
        """
        Разбивает задачу на шаги. Возвращает list[str].
        OpenClaw-level: derived skills are first-class citizens.
        If a skill matches the task, use it directly instead of LLM planning.
        """
        self._set_state("PLAN")

        # --- skill-first planning: check if any derived skill matches ---
        task_lower = task.lower()
        skill_candidates = []
        for skill_name, skill_data in self.derived_skills.items():
            pattern = skill_data.get("pattern", ())
            # Check if skill keywords appear in task
            keywords = [t.lower() for t in pattern]
            if any(kw in task_lower for kw in keywords):
                success_rate = self.tool_registry._success_graph.get(skill_name, {})
                avg_rate = sum(success_rate.values()) / len(success_rate) if success_rate else 0.5
                skill_candidates.append((skill_name, avg_rate, pattern))

        if skill_candidates:
            # Sort by success rate, pick best
            skill_candidates.sort(key=lambda x: x[1], reverse=True)
            best_skill = skill_candidates[0]
            console.print(f"[bold magenta]🎯 skill-first plan → {best_skill[0]} (rate={best_skill[1]:.2f})[/bold magenta]")
            return [f"USE_SKILL:{best_skill[0]}"]

        raw = self._llm(SYSTEM_PLANNER, f"Задача: {task}")

        # убираем markdown-обёртку если есть
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            data = json.loads(clean)
            steps = data.get("steps", [])
            if isinstance(steps, list) and steps:
                return steps
        except Exception:
            pass

        # fallback — делим по строкам
        lines = [l.strip("- •1234567890.) ").strip() for l in raw.splitlines() if l.strip()]
        steps = [l for l in lines if len(l) > 3]
        return steps if steps else [task]

    # ─── tool extraction ──────────────────────────────────────────────────────

    def _extract_tool(self, text: str):
        """
        Парсит TOOL/ARGS блок из ответа модели.
        Возвращает (tool_name: str|None, args: dict).
        Публичный метод — тестируется напрямую.
        """
        # try fenced match on raw text before stripping fences
        fenced_match = re.search(
            r"TOOL:\s*write_file\s*\nPATH:\s*(.+?)\n```(?:python)?\n(.*?)\n```",
            text,
            re.DOTALL | re.IGNORECASE
        )
        if fenced_match:
            return "write_file", {
                "path": fenced_match.group(1).strip(),
                "content": fenced_match.group(2)
            }

        # убираем markdown fences
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()

        tool_match = re.search(r"TOOL:\s*([a-zA-Z_]+)", cleaned, re.IGNORECASE)
        if not tool_match:
            return None, {}

        tool_name = tool_match.group(1).strip()

        # special format for huge write_file payloads
        if tool_name == "write_file":
            path_match = re.search(
                r"PATH:\s*(.+)",
                cleaned
            )
            content_match = re.search(
                r"CONTENT:\s*(.*)",
                cleaned,
                re.DOTALL
            )
            if path_match and content_match:
                return "write_file", {
                    "path": path_match.group(1).strip(),
                    "content": content_match.group(1).strip()
                }

        args_match = re.search(r"ARGS:\s*(\{.*)", cleaned, re.DOTALL)
        if not args_match:
            return tool_name, {}

        raw_args = args_match.group(1).strip()

        # вычленяем первый валидный JSON-объект
        brace_depth = 0
        end_idx     = None
        for i, ch in enumerate(raw_args):
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    end_idx = i
                    break

        if end_idx is None:
            return tool_name, {}

        candidate = raw_args[:end_idx + 1]

        # попытки парсинга
        for attempt in (candidate,
                         re.sub(r",\s*([}\]])", r"\1", candidate)):  # trailing comma fix
            try:
                return tool_name, json.loads(attempt)
            except Exception:
                pass

        try:
            import ast
            return tool_name, ast.literal_eval(candidate)
        except Exception:
            pass

        return tool_name, {}

    def _static_check(self, path: str) -> str:
        try:
            import ast
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            ast.parse(code)
            return "OK"
        except Exception as e:
            return str(e)

    def _compile_check(self, path: str) -> str:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return "COMPILE_OK"
            return result.stderr or result.stdout or "compile failed"
        except Exception as e:
            return str(e)

    # ─── verifier ─────────────────────────────────────────────────────────────

    def _verify(self, tool_result: str) -> str:
        """
        Возвращает: SUCCESS | FAIL | FIX
        FIX = ошибка в коде, нужно читать файл и чинить.
        """
        if tool_result:
            err_key = tool_result.strip()[:300]
            if err_key in self.seen_errors:
                return "FAIL"
            self.seen_errors.add(err_key)
        if tool_result is None:
            return "FAIL"
        r = tool_result.lower()
        if "crashed unexpectedly" in r or "traceback" in r:
            return "FIX"

        fix_markers = [
            "error occurred",
            "traceback",
            "exception",
            "syntaxerror",
            "nameerror",
            "typeerror",
            "attributeerror",
            "importerror",
            "indentationerror",
            "modulenotfounderror",
            "indexerror",
            "keyerror",
            "valueerror",
            "initscr",
            "must call initscr",
        ]
        if any(m in r for m in fix_markers):
            return "FIX"

        error_markers = ["error:", "no such file", "permission denied",
                         "not found", "llm_error", "loop blocked"]
        if any(m in r for m in error_markers):
            return "FAIL"

        success_markers = ["synced", "logged", "changed to", "compile_ok"]
        if any(m in r for m in success_markers):
            return "SUCCESS"

        return "RETRY"  # нет явных ошибок — просим перепроверку вместо ложного успеха

    # ─── executor ─────────────────────────────────────────────────────────────

    def _call_tool(self, tool_name: str, args: dict) -> str:
        """Выполняет инструмент. Возвращает строку результата."""
        # Check registry first (supports derived skills)
        registry_func = self.tool_registry.get(tool_name)
        if registry_func is None and tool_name not in self.tools:
            return f"Error: unknown tool {tool_name}"

        func = registry_func or self.tools.get(tool_name)

        # handle invalid read_file usage
        if tool_name == "read_file":
            if not args or "path" not in args or not args.get("path"):
                return "Error: missing path argument"
        if tool_name == "edit_file":
            if not all(k in args for k in ("path", "old", "new")):
                return "Error: edit_file requires path, old, new"
        if tool_name == "run_shell":
            if not args or "command" not in args:
                return "Error: missing command argument"
        # rate limits: check before call
        if tool_name in ("web_search", "fetch_url"):
            now = time.time()
            if now - self.web_last_call < self.web_min_interval:
                return "Error: web rate limit active"
        if tool_name == "memory_log":
            now = time.time()
            if now - self.last_memory_log_time < self.memory_log_cooldown:
                return "Memory log skipped (cooldown)"
            self.last_memory_log_time = now
        console.print(f"[bold yellow]→ {tool_name}({json.dumps(args, ensure_ascii=False)})[/]")
        try:
            result = func(**args)
        except Exception as e:
            result = f"Tool exception: {e}"
        result_str = str(result)

        # web cache: store result after successful call
        if tool_name in ("web_search", "fetch_url"):
            self.web_last_call = now
        # --- outer mesh usage tracking (legacy compat) ---
        self.tool_usage_graph.setdefault(tool_name, {"count": 0})
        self.tool_usage_graph[tool_name]["count"] += 1

        # --- OpenClaw event publishing ---
        status = "SUCCESS" if not result_str.startswith("Error") and "exception" not in result_str.lower() else "FAIL"
        self._publish_tool_event(tool_name, result_str, status)

        # --- watch files for change events ---
        if tool_name in ("write_file", "edit_file") and "path" in args:
            self._watch_file(args["path"])

        preview = result_str[:2500]
        if len(result_str) > 2500:
            preview += "\n\n...[truncated]..."

        console.print(
            Panel(
                Markdown(preview),
                title=f"📊 {tool_name}"
            )
        )

        self._append_history("tool", result_str)
        return result_str

    def _validate_code_result(self, tool_name: str, args: dict, result: str) -> tuple:
        """Run compile + runtime checks after code modifications.

        Returns (result, status) if validation fails, or None if OK.
        """
        if tool_name not in ("write_file", "edit_file"):
            return None

        target = args.get("path")
        if not target or not target.endswith(".py"):
            return None

        # compile check
        compile_result = self._compile_check(target)
        if compile_result != "COMPILE_OK":
            return f"VALIDATION_ERROR:\n{compile_result}", "FIX"

        # runtime check if file has main
        try:
            content = Path(target).read_text(encoding="utf-8")
            if '__name__ == "__main__"' in content:
                run_result = self.run_shell(command=f"python3 {target}")
                run_status = self._verify(run_result)
                if run_status == "FIX":
                    return run_result, "FIX"
        except Exception:
            pass

        return None

    def _has_crash_signal(self, text: str) -> bool:
        """Check if result text contains crash/error indicators."""
        lowered = text.lower()
        signals = [
            "crashed unexpectedly", "traceback", "exception",
            "nameerror", "typeerror", "attributeerror",
            "non_zero_exit", "[non_zero_exit]",
        ]
        return any(s in lowered for s in signals)

    def _execute_step(self, step: str, context: str = "") -> tuple:
        """
        Execute one step — primary path uses tool calling, falls back to text.

        Returns (result: str, status: str).
        """
        self._set_state("EXECUTE")

        project_ctx = self._retrieve_project_context(self.current_task or "")
        project_scan = self._scan_project()
        prompt = f"Текущий шаг: {step}\nЗадача: {self.current_task}"
        if project_scan:
            prompt += f"\nСтруктура проекта:\n{project_scan}"
        if project_ctx:
            prompt += f"\nКонтекст проекта:\n{project_ctx}"
        if context:
            prompt += f"\nКонтекст: {context}"

        tools = self._get_step_tools()
        for _round in range(5):
            response = self._llm_with_tools(SYSTEM_EXECUTOR_TOOLS, prompt, tools=tools)

            if response["type"] == "error":
                # inline text-based fallback on API error
                console.print("[yellow]↩️ tool-calling unavailable, using text fallback[/yellow]")
                return self._execute_step_text(step, context)

            if response["type"] == "text":
                content = response["content"]
                self._append_history("assistant", content)
                return content, "SUCCESS"

            calls = response.get("calls", [])
            if not calls:
                return "Model returned no tool calls", "FAIL"

            last_result = ""
            for call in calls:
                tool_name = call.function.name
                args = dict(call.function.arguments)

                if tool_name == self.last_tool_call:
                    console.print(f"[red]Loop blocked: {tool_name} повторяется[/red]")
                    self.seen_errors.add(f"loop:{tool_name}")
                    return f"Loop blocked: {tool_name}", "FAIL"
                self.last_tool_call = tool_name

                if not self._confirm_action(tool_name, args):
                    return "Action rejected by user", "FAIL"

                result = self._call_tool(tool_name, args)
                last_result = result

                validation = self._validate_code_result(tool_name, args, result)
                if validation is not None:
                    return validation

                if self._has_crash_signal(result):
                    return result, "FIX"

                self._update_skill_memory(step, tool_name, "PENDING")

            prompt = (
                f"Последний результат: {str(last_result)[:2000]}\n\n"
                f"Продолжи шаг: {step}"
            )

        return "Step did not complete in 5 rounds", "FAIL"

    def _execute_step_text(self, step: str, context: str = "") -> tuple:
        """Text-based executor fallback (when tool-calling API unavailable)."""
        project_ctx = self._retrieve_project_context(self.current_task or "")
        text_prompt = f"Шаг: {step}\nЗадача: {self.current_task}\nКонтекст:\n{project_ctx}"
        if context:
            text_prompt += f"\nПредыдущая ошибка: {context}"

        reply = self._llm(SYSTEM_EXECUTOR, text_prompt, stream=True)
        self._append_history("assistant", reply)

        answer_match = re.search(r"ANSWER:\s*(.*)", reply, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip(), "SUCCESS"

        tool_name, args = self._extract_tool(reply)
        if not tool_name:
            return reply, "FAIL"

        if tool_name == self.last_tool_call:
            console.print(f"[red]Loop blocked: {tool_name}[/red]")
            self.seen_errors.add(f"loop:{tool_name}")
            return f"Loop blocked: {tool_name}", "FAIL"
        self.last_tool_call = tool_name

        result = self._call_tool(tool_name, args)

        validation = self._validate_code_result(tool_name, args, result)
        if validation is not None:
            return validation

        if self._has_crash_signal(str(result).lower()):
            return result, "FIX"
        if isinstance(result, str) and "missing path" in result.lower():
            return result, "FIX"

        self._verify(result)
        self._update_field_metrics(result, "SUCCESS")
        self._evaluate_external(step, result, "SUCCESS")
        return result, "SUCCESS"

    def _find_target_file(self, error_result: str, step: str) -> str:
        """Find the failing file from error/step context."""
        if self.locked_file:
            return self.locked_file

        target = self.last_ran_file or self.pending_validation_file or self.last_read_file

        patterns = [
            re.compile(r'python3?\s+([\w./\-]+\.py)'),
            re.compile(r'File \"([^\"]+\.py)\"'),
            re.compile(r'([\w./\-]+\.py)'),
        ]
        for text in [step, error_result]:
            for pat in patterns:
                m = pat.search(text)
                if m:
                    return m.group(1)

        if "initscr" in error_result.lower() or "curses" in error_result.lower():
            return self.last_ran_file

        return target

    def _fix_step(self, error_result: str, step: str) -> bool:
        """
        FIX state: read failing file, ask LLM for specific edits, apply via edit_file.

        Returns True if fixed.
        """
        self._set_state("FIX")
        console.print("[bold magenta]→ FIX: finding failing file...[/bold magenta]")

        target_file = self._find_target_file(error_result, step)
        if not target_file:
            console.print("[red]FIX: unable to determine target file[/red]")
            return False

        code = self.read_file(target_file)
        if code.startswith("Error:"):
            console.print(f"[red]FIX: unable to read {target_file}: {code}[/red]")
            return False

        console.print(f"[magenta]FIX: reading {target_file} ({len(code)} chars)[/magenta]")

        # try diff-first: ask LLM for specific old→new replacements
        fix_prompt = (
            f"Файл {target_file} падает с ошибкой:\n{error_result}\n\n"
            f"Код файла:\n{code[:8000]}\n\n"
            "Верни ТОЛЬКО JSON-массив объектов с полями 'old' и 'new'."
            "Каждый объект — одна замена: найди точную строку (old) и замени на (new).\n"
            "Пример: [{\"old\": \"x = 1 + 1\", \"new\": \"x = 2\"}]\n"
            "Без объяснений, без markdown. Только JSON."
        )

        diff_response = self._llm(
            "Ты — эксперт Python. Возвращай только JSON-массив замен.",
            fix_prompt,
        )

        edits = self._parse_fix_edits(diff_response)
        if edits:
            applied = 0
            for old, new in edits:
                result = self.edit_file(path=target_file, old=old, new=new)
                if not result.startswith("Error"):
                    applied += 1
            if applied > 0:
                console.print(f"[green]✓ FIX applied {applied} edit(s) to {target_file}[/green]")
                return True

        # fallback: rewrite entire file
        console.print("[yellow]FIX: diff approach failed, rewriting full file[/yellow]")
        full_fix_prompt = (
            f"Файл {target_file} падает с ошибкой:\n{error_result}\n\n"
            f"Код файла:\n{code[:8000]}\n\n"
            "Верни ТОЛЬКО исправленный полный код файла. Без объяснений, без markdown."
        )
        fixed_code = self._llm(
            "Ты — эксперт Python. Чини баги. Возвращай только код.",
            full_fix_prompt,
        )
        fixed_code = re.sub(r"^```(?:python)?\n?", "", fixed_code.strip())
        fixed_code = re.sub(r"\n?```$", "", fixed_code.strip())

        if len(fixed_code) < 20:
            console.print("[red]FIX: model returned empty response[/red]")
            return False

        result = self.write_file(target_file, fixed_code)
        console.print(Panel(result, title="🔧 FIX applied (full rewrite)"))
        return not result.startswith("Error")

    def _parse_fix_edits(self, text: str) -> list:
        """Parse LLM's fix response into list of (old, new) tuples."""
        try:
            cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
            data = json.loads(cleaned)
            if isinstance(data, list):
                return [(item["old"], item["new"]) for item in data
                        if isinstance(item, dict) and "old" in item and "new" in item]
            if isinstance(data, dict) and "edits" in data:
                return [(e["old"], e["new"]) for e in data["edits"]
                        if isinstance(e, dict) and "old" in e and "new" in e]
            return []
        except Exception:
            return []

    # ─── main think ──────────────────────────────────────────────────────────

    def think(self, task: str):
        self.current_task   = task
        self.last_activity_time = time.time()
        self.interrupt_requested = False
        self.loop_count     = 0
        self.last_tool_call = None
        # --- lock file from user task ---
        self.locked_file = None

        m = re.search(r'([\w./\-]+\.py)', task)
        if m:
            self.locked_file = m.group(1)
            console.print(f"[cyan]🔒 locked file → {self.locked_file}[/cyan]")
        seen_hashes         = set()

        self._append_history("user", task)
        # WEB EXEC ROUTE PATCH 🌐
        if re.search(r'\b(find|search|look up|check|поищи|найди)\b', task.lower()):
            console.print("[cyan]🌐 WEB_EXEC route engaged[/cyan]")
            try:
                result = self.web_search(task)
            except Exception as e:
                result = f"WEB_EXEC_ERROR: {e}"
            self._append_history("tool", result)
            self._set_state("DONE")
            # reduce HTML/log spam output 🌊
            try:
                cleaned = re.sub(r'<[^<]+?>', '', result)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            except Exception:
                cleaned = result

            preview = cleaned[:800]
            console.print(preview)
            return

        task_lower = task.lower()
        # conversational signals: just answer, no action needed
        no_action_signals = [
            "просто ответь", "ничего не делай", "just answer",
            "как дела", "who are you", "кто ты", "кем ты",
            "расскажи о себе", "tell me about yourself",
            "how are you", "что ты умеешь", "what can you do",
        ]
        action_keywords = [
            "file", "code", "fix", "edit", "write", "run",
            "search", "git", "folder", "dir", "создай", "напиши",
            "почини", "найди", "файл", "код", "исправь",
            "сделай", "запусти", "открой", "удали", "переименуй",
        ]
        is_conversational = any(s in task_lower for s in no_action_signals)
        has_actions = any(k in task_lower for k in action_keywords)
        is_short = len(task.split()) < 30

        if is_conversational or (is_short and not has_actions):
            steps = [task]
        else:
            steps = self._plan(task)
            console.print(Panel(
                "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)),
                title="📋 Plan"
            ))
        self._append_history("system", f"Plan: {json.dumps(steps, ensure_ascii=False)}")

        self._set_state("EXECUTE")
        for step_idx, step in enumerate(steps):
            if self.loop_count >= 20:
                console.print("[red]Hard stop: loop_count > 20[/red]")
                break

            console.print(f"\n[bold]── Step {step_idx+1}/{len(steps)}: {step}[/]")

            step_hash = hashlib.md5(step.encode()).hexdigest()
            if step_hash in seen_hashes:
                console.print("[red]Duplicate step detected → skipping[/red]")
                continue
            seen_hashes.add(step_hash)

            self.last_tool_call = None

            # --- skill-first execution: if step is a skill directive, run it directly ---
            if step.startswith("USE_SKILL:"):
                skill_name = step.split(":", 1)[1]
                skill_func = self.tool_registry.get(skill_name)
                if skill_func:
                    console.print(f"[bold magenta]🎯 executing skill → {skill_name}[/bold magenta]")
                    try:
                        result = skill_func()
                        status = "SUCCESS"
                    except Exception as e:
                        result = f"Skill error: {e}"
                        status = "FAIL"
                else:
                    result = f"Error: skill {skill_name} not found in registry"
                    status = "FAIL"
                self.loop_count += 1
                try:
                    if getattr(self, "last_observation", None):
                        self.error_signal = self.analyze_ui(self.last_observation)
                except Exception:
                    pass
            else:
                result, status = self._execute_step(step)
                self.loop_count += 1
                try:
                    if getattr(self, "last_observation", None):
                        self.error_signal = self.analyze_ui(self.last_observation)
                except Exception:
                    pass

            # --- validation branch ---
            if isinstance(result, str) and result.startswith("VALIDATE:"):
                target = result.split(":", 1)[1]

                console.print(
                    f"[cyan]🧪 validating {target}[/cyan]"
                )

                compile_result = self._compile_check(target)

                if compile_result != "COMPILE_OK":
                    fixed = self._fix_step(compile_result, step)

                    if fixed:
                        compile_result = self._compile_check(target)

                    if compile_result != "COMPILE_OK":
                        console.print(
                            f"[red]Validation failed: {compile_result}[/red]"
                        )
                        continue

                try:
                    content = Path(target).read_text(encoding="utf-8")
                except Exception:
                    content = ""

                if 'if __name__ == "__main__":' in content:
                    result2 = self.run_shell(
                        command=f"python3 {target}"
                    )

                    status2 = self._verify(result2)

                    if status2 == "FIX":
                        fixed = self._fix_step(result2, step)

                        if fixed:
                            result3 = self.run_shell(
                                command=f"python3 {target}"
                            )
                            self._verify(result3)

                continue

            if status == "SUCCESS":
                console.print("[green]✓ step verified[/green]")
                continue

            if status == "FIX":
                # try to fix the code
                fixed = self._fix_step(result, step)

                # --- post-FIX validation run ---
                target_file = (
                    self.last_ran_file
                    or self.pending_validation_file
                    or self.last_read_file
                )

                if fixed and target_file:
                    result_run = self.run_shell(command=f"python3 {target_file}")
                    status_run = self._verify(result_run)

                    if status_run == "FIX":
                        fixed2 = self._fix_step(result_run, step)

                        if fixed2:
                            result3 = self.run_shell(command=f"python3 {target_file}")
                            self._verify(result3)

                if fixed:
                    # after fixing — repeat the step once
                    self.last_tool_call = None
                    result2, status2 = self._execute_step(step, context=f"Code was fixed. Repeat the step.")
                    self.loop_count += 1
                    if status2 != "SUCCESS" and status2 != "FIX":
                        console.print(f"[red]Step still failing after fix: {result2}[/red]")
                else:
                    console.print("[red]FIX failed[/red]")
                continue

            # FAIL — one retry with error context
            self.last_tool_call = None
            self._append_history("user", f"Step failed: {result}. Try another way.")
            result2, status2 = self._execute_step(step, context=f"Previous attempt failed: {result}")
            self.loop_count += 1

        self._compress_memory()
        self._set_state("DONE")
        console.print("[bold green]✓ Task completed[/bold green]")

    # ─── tools ────────────────────────────────────────────────────────────────

    def read_file(self, path: str) -> str:
        """Read a file from disk and return its contents.

        Args:
            path: Absolute or relative path to the file to read

        Returns:
            File contents as a string, up to 15000 characters.
            Returns 'Error: ...' on failure.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            self.last_read_file = path

            console.print(
                f"[cyan]📄 reading[/] {path} "
                f"[dim]{content.count(chr(10)) + 1} lines "
                f"{_pretty_size(len(content))}[/dim]"
            )

            return content[:15000]
        except Exception as e:
            return f"Error: {e}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file on disk. Creates parent directories if needed.

        Args:
            path: File path to write to
            content: Full file content to write

        Returns:
            '✅ File {path} written successfully.' on success,
            'Error: ...' on failure.
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            if Path(path).exists():
                _backup_file(path)
                try:
                    old_content = Path(path).read_text(encoding="utf-8")
                    _show_patch_preview(path, old_content, content)
                except Exception:
                    pass
            # strip markdown code fences if model leaks them
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

            # remove stray model artifacts / tokens
            content = re.sub(r"<unused\d+>", "", content)

            # remove standalone code fence lines
            content = "\n".join(
                line for line in content.splitlines()
                if line.strip() != "```"
            )

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            if path.endswith(".py"):
                self.pending_validation_file = path
            if path.endswith(".html"):
                self._auto_launch_web_app(path)
            return f"✅ File {path} written successfully."
        except Exception as e:
            return f"Error: {e}"

    def _auto_launch_web_app(self, path: str):
        """Auto-launch simple HTML apps in browser + local server."""
        try:
            import subprocess, os, time
            from pathlib import Path

            file_path = Path(path)
            base_dir = str(file_path.parent.resolve())
            file_name = file_path.name

            # start server (non-blocking)
            if not hasattr(self, "_web_server_process") or self._web_server_process is None:
                self._web_server_process = subprocess.Popen(
                    "python3 -m http.server 8000",
                    cwd=base_dir,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(0.5)

            url = f"http://localhost:8000/{file_name}"
            webbrowser.open(url)

            console.print(f"[bold green]🌐 auto-launched web app → {url}[/bold green]")
            # vision feedback loop (HTML embodiment)
            try:
                shot = self.screenshot(url)
                self.last_observation = self.vision_model(shot)
                self.feedback_loop_enabled = True
            except Exception:
                pass
        except Exception as e:
            console.print(f"[red]web auto-launch failed: {e}[/red]")

    def edit_file(self, path: str = None, old: str = None, new: str = None,
                  replace_all: bool = False, **kwargs) -> str:
        """Edit a file by replacing text. Shows a diff preview before applying.

        Args:
            path: File path to edit
            old: Text to find and replace
            new: Replacement text
            replace_all: If True, replace all occurrences; otherwise only first

        Returns:
            '✅ Edited {path}' on success, 'Error: ...' on failure.
        """
        path = path or kwargs.get("path")
        old  = old  or kwargs.get("old")
        new  = new  or kwargs.get("new")

        if not path or old is None or new is None:
            return "Error: edit_file requires path, old, new"
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if old not in content:
                return f"Error: pattern not found in {path}"
            old_content = content
            content = content.replace(old, new) if replace_all else content.replace(old, new, 1)

            _show_patch_preview(path, old_content, content)
            _backup_file(path)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            if path.endswith(".py"):
                self.pending_validation_file = path
            return f"✅ Edited {path}"
        except Exception as e:
            return f"Error: {e}"

    def run_shell(self, command: str = None, commands: list = None, wait: bool = False) -> str:
        """Execute shell commands. Use with extreme care.

        Args:
            command: Single shell command to run
            commands: Multiple commands joined with &&
            wait: If True, start process in background and return immediately

        Returns:
            Combined stdout/stderr with return code. Prefixed with
            '[NON_ZERO_EXIT]' if exit code != 0.
        """
        try:
            if commands:
                command = " && ".join(commands) if isinstance(commands, list) else commands
            if not command:
                return "Error: No command provided"
            # Fix 3: source venv бессмысленен в subprocess — каждый вызов новый процесс
            command = re.sub(r"source\s+\S*venv\S*/activate\s*&&\s*", "", command)
            command = re.sub(r"\.\s+\S*venv\S*/activate\s*&&\s*", "", command)
            # запоминаем какой .py файл запускаем
            m = re.search(r'python3?\s+([\w./\-]+\.py)', command)
            if m:
                self.last_ran_file = m.group(1)
            if wait:
                with self._process_lock:
                    self.active_process = subprocess.Popen(command, shell=True)
                return f"PROCESS_STARTED: {command}"
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=60
            )
            out = result.stdout[:8000]
            err = result.stderr[:2000]
            code = result.returncode

            combined = (
                f"RETURN_CODE: {code}\n"
                f"STDOUT:\n{out}\n"
                f"STDERR:\n{err}"
            )

            if code != 0:
                combined = "[NON_ZERO_EXIT]\n" + combined

            return combined
        except subprocess.TimeoutExpired:
            return "Error: command timed out (60s)"
        except Exception as e:
            return f"Error: {e}"

    def interactive_run(self, command: str, duration: int = 5) -> str:
        """Run a long-lived process for testing interactive apps. Thread-safe."""
        try:
            with self._process_lock:
                if self.active_process:
                    self.active_process.terminate()
                    self.active_process = None
                proc = subprocess.Popen(command, shell=True)
                self.active_process = proc
            time.sleep(duration)
            with self._process_lock:
                if self.active_process:
                    self.active_process.terminate()
                    self.active_process = None
            return "GAME_TESTED"
        except Exception as e:
            return f"Error: {e}"

    def list_files(self, dir: str = ".") -> str:
        """List files in a directory recursively (up to 200 entries).

        Args:
            dir: Directory path to list

        Returns:
            Newline-separated list of file paths.
        """
        try:
            files = [str(p) for p in Path(dir).rglob("*") if p.is_file()]
            return "\n".join(files[:200])
        except Exception as e:
            return f"Error: {e}"

    def search_code(self, pattern: str, dir: str = ".") -> str:
        """Search for a pattern in Python files using grep.

        Args:
            pattern: Regex pattern to search for
            dir: Directory to search in

        Returns:
            grep results as a string, up to 8000 chars.
        """
        try:
            result = subprocess.run(
                f"grep -r --include='*.py' '{pattern}' {dir}",
                shell=True, capture_output=True, text=True
            )
            return result.stdout[:8000] or "Nothing found."
        except Exception as e:
            return f"Error: {e}"

    def memory_log(self, message: str = None, action: str = None,
                   event: str = None, **kwargs) -> str:
        """Log a message or action to persistent memory file.

        Args:
            message: Log message text. Falls back to action or event or kwargs.

        Returns:
            '✅ Memory logged' on success, 'Error: ...' on failure.
        """
        try:
            log_path = Path("aether_memory.json")
            data: list = []
            if log_path.exists():
                try:
                    data = json.loads(log_path.read_text(encoding="utf-8"))
                except Exception:
                    data = []
            msg = message or action or event or json.dumps(kwargs, ensure_ascii=False)
            data.append({"t": time.time(), "msg": msg})
            log_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            return "✅ Memory logged"
        except Exception as e:
            return f"Error: {e}"

    def git_sync(self, message: str = "Aether auto commit") -> str:
        """Stage all changes, commit, and push to remote.

        Args:
            message: Commit message

        Returns:
            Git output or error message.
        """
        try:
            subprocess.run("git add .", shell=True, check=False)
            subprocess.run(f'git commit -m "{message}"', shell=True, check=False)
            result = subprocess.run("git push", shell=True, capture_output=True, text=True)
            return f"Git sync done.\n{result.stdout}{result.stderr}"
        except Exception as e:
            return f"Git error: {e}"

    def change_dir(self, path: str) -> str:
        """Change current working directory.

        Args:
            path: Directory path to change to

        Returns:
            Confirmation with new cwd or error.
        """
        try:
            os.chdir(path)
            return f"✅ cwd → {os.getcwd()}"
        except Exception as e:
            return f"Error: {e}"

    # ─── REPL ─────────────────────────────────────────────────────────────────

    def screenshot(self, url: str):
        try:
            

            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(url)
                img = page.screenshot(full_page=True)
                browser.close()
                return img
        except Exception:
            return None

    def analyze_ui(self, image):
        try:
            if image is None:
                return 0.0

            # lightweight placeholder signal (replaceable with VLM/CLIP later)
            return 0.5
        except Exception:
            return 0.0
        
    def vision_model(self, image):
        try:
            if image is None:
                return 0.0
            return 0.5
        except Exception:
            return 0.0

    def run(self):
        import queue
        console.print(Panel("🚀 Aether vNext | 0penAGI", style="bold green"))
        layout = Layout()

        layout.split_column(
            Layout(name="output"),
            Layout(size=3, name="input")
        )

        self.start_breathing()
        self.start_runtime()
        self.start_gateway()  # Start the persistent gateway loop

        # start task worker thread (async execution pipeline)
        threading.Thread(
            target=self._task_worker,
            daemon=True
        ).start()

        # input thread (non-blocking producer)
        def _input_thread(q):
            while True:
                try:
                    inp = console.input("\n[bold cyan]> [/]").strip()
                    q.put(inp)
                except Exception:
                    break

        threading.Thread(
            target=_input_thread,
            args=(self.input_queue,),
            daemon=True
        ).start()

        # Subscribe to events for UI feedback
        self.event_bus.subscribe(EventType.SKILL_DISCOVERED, lambda e: console.print(f"[dim cyan]🧬 skill: {e.payload.get('skill')}[/dim cyan]"))
        self.event_bus.subscribe(EventType.TOOL_FAILED, lambda e: console.print(f"[dim red]🔧 tool failed: {e.payload.get('tool')}[/dim red]"))
        self.event_bus.subscribe(EventType.MEMORY_SNAPSHOT, lambda e: console.print("[dim green]💾 memory snapshot saved[/dim green]"))

        try:
            while not getattr(self, "shutdown_requested", False):
                try:
                    task = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    task = None
                    continue

                if task:
                    if task.lower() in ("exit", "quit", "q"):
                        break
                    try:
                        self.task_queue.put(task)
                    except KeyboardInterrupt:
                        # first Ctrl+C = interrupt generation only
                        self.interrupt_requested = True
                        console.print("[yellow]⏸ generation interrupted[/yellow]")
                        continue

                # --- UI SAFE SPACE RENDER ---
                try:
                    output_text = "\n".join(self.output_buffer[-15:])
                    layout["output"].update(Markdown(output_text if output_text else " "))
                    layout["input"].update(Text("💬 safe input active — type anytime", style="dim cyan"))
                except Exception:
                    pass
        except KeyboardInterrupt:
            self.shutdown_requested = True
            self.interrupt_requested = True
            console.print("[yellow]⏸ graceful shutdown[/yellow]")
        finally:
            try:
                self.save_state()  # Final state persistence
                self.stop_gateway()
                self._set_state("IDLE")
            except Exception:
                pass


def main_cli():
    """Entry point for `python3 aether.py` and `aether` CLI command."""
    from aether_config import AetherConfig
    import sys

    if "--init" in sys.argv:
        AetherConfig().run_wizard()
        return

    cfg = AetherConfig()
    if not cfg.exists():
        print("First run detected — starting setup wizard")
        cfg.run_wizard()

    # Quick one-shot task
    task_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    task = " ".join(task_args) if task_args else None

    agent = AetherAgent(model=cfg.model)
    agent.skip_confirmation = cfg.config.get("skip_confirmation", False)

    if task:
        agent.think(task)
    else:
        agent.run()


if __name__ == "__main__":
    main_cli()
