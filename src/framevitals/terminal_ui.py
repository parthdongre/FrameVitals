"""Local, terminal-first FrameVitals interface.

Built with the Python standard library so the installed CLI can open an
interactive workspace without requiring an additional TUI framework.  All
analyses use the existing public protocol APIs; the interface itself never
reimplements diagnostics or mutates a source file.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ACTIONS = (
    ("Prism", "Diagnose a dataset, graph, tensor, or model"),
    ("Tide", "Compare a reference and current source"),
    ("Axiom", "Infer or check source expectations"),
    ("Pulse", "Capture a compact health snapshot"),
    ("Inspect", "Inspect source type and capabilities"),
    ("Forge preview", "Preview a cleaning plan (no changes)"),
)
DEPTHS = ("quick", "standard", "deep", "research")
from framevitals.file_formats import (
    format_for,
    is_available,
    prepare_file_source,
    render_formats,
    supported_extensions,
)

SUPPORTED_FILES = supported_extensions()


@dataclass
class TerminalReport:
    """Completed protocol result, with separately navigable text views."""

    action: str
    result: Any
    summary: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""
    status: str = "complete"


@dataclass
class TerminalState:
    action_index: int = 0
    source: str = ""
    reference: str = ""
    depth: str = "quick"
    page: int = 0
    scroll: int = 0
    busy: bool = False
    started: float = 0.0
    report: TerminalReport | None = None
    notice: str = "Choose a protocol, select a file (F), and press Enter."
    error: bool = False

    @property
    def action(self) -> str:
        return ACTIONS[self.action_index][0]


def _path_source(value: str | Path) -> Any:
    """Route local files through the same safe format registry as the CLI."""
    return prepare_file_source(value)


def _serialize(value: Any) -> str:
    if hasattr(value, "to_json"):
        result = value.to_json()
        if isinstance(result, str):
            return result
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def _findings(value: Any) -> list[dict[str, Any]]:
    for name in ("beacons", "findings"):
        items = getattr(value, name, None)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    if isinstance(value, dict):
        items = value.get("findings", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _summary(value: Any) -> str:
    render = getattr(value, "summary_text", None)
    if callable(render):
        return str(render())
    if isinstance(value, dict):
        return json.dumps(value, indent=2, default=str, ensure_ascii=False)
    return str(value)


def run_diagnostic(
    action: str,
    source: str | Path,
    *,
    reference: str | Path | None = None,
    depth: str = "quick",
) -> TerminalReport:
    """Execute one existing FrameVitals protocol in a side-effect-safe mode."""
    if action not in {name for name, _ in ACTIONS}:
        raise ValueError(f"Unknown terminal action: {action}")
    if depth not in DEPTHS:
        raise ValueError(f"Unknown Prism depth: {depth}")

    data = _path_source(source)
    baseline = _path_source(reference) if reference else None
    if action == "Forge preview" and format_for(source).category != "Tabular":
        raise ValueError("Forge preview currently supports tabular files only.")

    if action == "Prism":
        from framevitals.protocols import prism

        result = prism(data, reference=baseline, depth=depth, artifacts=False)
    elif action == "Tide":
        if baseline is None:
            raise ValueError("Tide needs a reference file (press B to select one).")
        from framevitals.protocols import tide

        result = tide(baseline, data)
    elif action == "Axiom":
        from framevitals.protocols import axiom

        result = axiom(
            baseline if baseline is not None else data,
            current=data if baseline is not None else None,
        )
    elif action == "Pulse":
        from framevitals.protocols import pulse

        result = pulse(data, depth=depth)
    elif action == "Inspect":
        import framevitals as fv

        result = fv.inspect_source(data)
    else:
        if baseline is not None:
            raise ValueError("Forge preview only operates on one tabular source.")
        from framevitals.protocols import forge

        result = forge(data, apply=False)

    status = getattr(result, "status", None)
    if status is None and isinstance(result, dict):
        status = result.get("status", "complete")
    return TerminalReport(
        action=action,
        result=result,
        summary=_summary(result),
        findings=_findings(result),
        detail=_serialize(result),
        status=str(status or "complete"),
    )


def save_report(report: TerminalReport, destination: str | Path) -> Path:
    """Explicit export; the interface never writes results without permission."""
    path = Path(destination).expanduser()
    if path.suffix.lower() not in {".json", ".html", ".htm"}:
        raise ValueError("Reports must end in .json or .html")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".html", ".htm"}:
        write_html = getattr(report.result, "to_html", None)
        if not callable(write_html):
            raise ValueError("HTML export is available for Prism/analysis reports only.")
        write_html(path)
    else:
        path.write_text(_serialize(report.result) + "\n", encoding="utf-8")
    return path


def _finding_lines(report: TerminalReport) -> list[str]:
    if not report.findings:
        return ["No Beacons reported for this operation.", "",
                "Absence of Beacons does not guarantee the source is healthy."]
    lines = []
    for index, item in enumerate(report.findings[:100], start=1):
        severity = str(item.get("severity", "info")).upper()
        lines.extend([
            f"{index:02d}  [{severity}] {item.get('title', item.get('code', 'Finding'))}",
            f"     {item.get('summary', item.get('description', ''))}",
        ])
        recommendation = item.get("recommendation")
        if recommendation:
            lines.append(f"     Next: {recommendation}")
        lines.append("")
    if len(report.findings) > 100:
        lines.append(f"+{len(report.findings) - 100} more Beacons (export JSON for all)")
    return lines


def view_lines(state: TerminalState) -> list[str]:
    if state.page == 3:
        return render_formats().splitlines()
    if state.error and state.notice and state.report is None:
        return [
            "ANALYSIS ERROR",
            "",
            state.notice,
            "",
            "Check the file format (L), select a valid file (P),",
            "or run 'framevitals doctor' for installation help.",
        ]
    if state.report is None:
        return [
            "WELCOME TO FRAMEVITALS",
            "",
            "Interactive diagnostics for structured data and ML artifacts.",
            "",
            f"Protocol   {state.action}",
            f"Source     {state.source or '(press F to select)'}",
            f"Reference  {state.reference or '(optional; press B)'}",
            f"Depth      {state.depth}",
            "",
            "ENTER to analyze    F to type path    P to browse files",
            "B to set baseline    D to change depth",
            "L lists every supported format and optional dependency",
            "",
            "Supports tables, JSON/YAML, graphs, tensors, model files,",
            "and PDF/DOCX/PPTX/TXT/Markdown/HTML/XML documents.",
            "",
            "This UI performs diagnostics without modifying source files.",
        ]
    if state.page == 0:
        return state.report.summary.splitlines() or ["No summary available."]
    if state.page == 1:
        return _finding_lines(state.report)
    lines = state.report.detail.splitlines()
    if len(lines) > 3500:
        return lines[:3500] + [
            "", f"... {len(lines) - 3500} additional lines; export full JSON."
        ]
    return lines or ["No detailed output available."]


def _text_lines(lines: list[str], width: int) -> list[str]:
    result: list[str] = []
    for line in lines:
        if not line:
            result.append("")
            continue
        result.extend(
            textwrap.wrap(
                line, width=max(15, width), replace_whitespace=False,
                drop_whitespace=False, break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
        )
    return result


def _run_plain() -> int:
    """Usable line-oriented fallback for terminals without curses support."""
    state = TerminalState()
    print("FRAMEVITALS  |  Terminal diagnostics")
    while True:
        print("\n" + "─" * 56)
        for i, (name, description) in enumerate(ACTIONS, 1):
            print(f"  {i}. {name:<14} {description}")
        print("  L. Supported file formats")
        print("  Q. Quit")
        try:
            choice = input("\nChoose a protocol: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 0
        if choice in {"q", "quit", "exit"}:
            return 0
        if choice in {"l", "list", "formats"}:
            print("\n" + render_formats())
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(ACTIONS):
            print("Choose 1–6 or Q.")
            continue
        state.action_index = int(choice) - 1
        try:
            source = input("Source file: ").strip()
            if not source:
                print("A source file is required.")
                continue
            reference = ""
            if state.action in {"Tide", "Axiom", "Prism"}:
                reference = input("Reference file (Enter to skip): ").strip()
            depth = "quick"
            if state.action in {"Prism", "Pulse"}:
                entered = input("Depth [quick/standard/deep/research] (quick): ").strip()
                depth = entered or "quick"
            report = run_diagnostic(state.action, source, reference=reference or None, depth=depth)
            print("\n" + report.summary)
            if report.findings:
                print("\nBEACONS")
                print("\n".join(_finding_lines(report)))
            destination = input("\nExport .json/.html path (Enter to skip): ").strip()
            if destination:
                print(f"Saved: {save_report(report, destination)}")
        except (EOFError, KeyboardInterrupt):
            print("\nOperation interrupted.")
            return 0
        except Exception as exc:
            print(f"Analysis error: {exc}", file=sys.stderr)


def _screen(stdscr: Any) -> int:
    import curses

    state = TerminalState()
    responses: queue.SimpleQueue[tuple[TerminalReport | None, str | None]] = queue.SimpleQueue()
    frames = "◐◓◑◒"
    stdscr.keypad(True)
    stdscr.timeout(120)
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
    except curses.error:
        pass

    def put(y: int, x: int, text: str, *, attr: int = 0) -> None:
        height, width = stdscr.getmaxyx()
        if 0 <= y < height and 0 <= x < width:
            try:
                stdscr.addnstr(y, x, str(text), width - x - 1, attr)
            except (curses.error, ValueError):
                pass

    def ask(label: str, initial: str = "") -> str | None:
        height, width = stdscr.getmaxyx()
        if height < 9 or width < 42:
            return None
        stdscr.timeout(-1)
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        prompt = f" {label} (Esc to cancel): "
        put(height - 3, 1, " " * (width - 3))
        put(height - 3, 1, prompt, attr=curses.A_BOLD)
        if initial:
            put(height - 2, 2, initial[: width - 5])
        curses.echo()
        try:
            # curses getstr permits paths with spaces; Escape remains a
            # cancelled entry if it arrives as the first byte.
            raw = stdscr.getstr(height - 2, 2, 2048)
            if raw.startswith(b"\x1b"):
                return None
            value = raw.decode("utf-8", errors="replace").strip()
            return value or initial
        except (curses.error, KeyboardInterrupt):
            return None
        finally:
            curses.noecho()
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            stdscr.timeout(120)

    def browse(initial: str = "") -> str | None:
        """Browse local folders with arrows/j/k, Enter and Backspace."""
        path = Path(initial).expanduser()
        location = path.parent if path.is_file() else path
        if not location.is_dir():
            location = Path.cwd()
        index = 0
        while True:
            try:
                entries = [
                    (entry.name, Path(entry.path), entry.is_dir())
                    for entry in os.scandir(location)
                    if not entry.name.startswith(".")
                ]
            except OSError:
                entries = []
            entries.sort(key=lambda item: (not item[2], item[0].lower()))
            entries.insert(0, ("..", location.parent, True))
            index = max(0, min(index, len(entries) - 1))
            height, width = stdscr.getmaxyx()
            stdscr.erase()
            put(0, 2, "FRAMEVITALS  /  FILE EXPLORER", attr=curses.A_BOLD)
            put(1, 2, str(location))
            put(2, 2, "↑↓ / j k move   Enter open/select   Backspace parent   Q cancel")
            put(3, 1, "─" * max(1, width - 3))
            visible = max(1, height - 6)
            offset = max(0, min(index - visible // 2, len(entries) - visible))
            for line, idx in enumerate(range(offset, min(len(entries), offset + visible))):
                name, _entry_path, is_dir = entries[idx]
                marker = "▸ " if idx == index else "  "
                suffix = "/" if is_dir else ""
                spec = None if is_dir else format_for(name)
                badge = "" if is_dir else (
                    " [unsupported]" if spec is None else
                    (f" [+{spec.extra}]" if spec.extra and not is_available(spec) else "")
                )
                style = curses.A_REVERSE if idx == index else 0
                put(
                    4 + line, 2, (marker + name + suffix + badge)[: max(1, width - 5)],
                    attr=style,
                )
            stdscr.refresh()
            key_pressed = stdscr.getch()
            if key_pressed == -1:
                continue
            if key_pressed in (ord("q"), ord("Q"), 27):
                return None
            if key_pressed in (curses.KEY_UP, ord("k")):
                index = (index - 1) % len(entries)
            elif key_pressed in (curses.KEY_DOWN, ord("j")):
                index = (index + 1) % len(entries)
            elif key_pressed in (curses.KEY_BACKSPACE, 127, 8):
                location = location.parent
                index = 0
            elif key_pressed in (10, 13, curses.KEY_ENTER):
                _name, target, is_dir = entries[index]
                if is_dir:
                    location = target
                    index = 0
                elif target.is_file():
                    # Reject unrecognized extensions here instead of letting
                    # the tabular loader fail with an unrelated parsing error.
                    if format_for(target) is None:
                        continue
                    return str(target)

    def worker(action: str, source: str, reference: str, depth: str) -> None:
        try:
            outcome = run_diagnostic(
                action, source, reference=reference or None, depth=depth,
            )
            responses.put((outcome, None))
        except Exception as exc:
            responses.put((None, f"{type(exc).__name__}: {exc}"))

    def draw() -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if width < 62 or height < 18:
            put(1, 2, "FrameVitals • enlarge terminal (min 62x18)")
            put(3, 2, "Q to quit")
            stdscr.refresh()
            return
        sidebar = 24
        cyan = curses.color_pair(1) if curses.has_colors() else curses.A_BOLD
        green = curses.color_pair(2) if curses.has_colors() else curses.A_BOLD
        yellow = curses.color_pair(3) if curses.has_colors() else curses.A_BOLD
        put(0, 2, "FRAMEVITALS", attr=cyan | curses.A_BOLD)
        put(0, 17, "Diagnostic terminal   /   0.3")
        put(1, 1, "─" * (width - 3))
        put(3, 2, "PROTOCOLS", attr=curses.A_BOLD)
        for i, (name, _description) in enumerate(ACTIONS):
            marker = "▸ " if state.action_index == i else "  "
            style = curses.A_REVERSE | curses.A_BOLD if state.action_index == i else 0
            put(5 + i * 2, 2, (marker + name).ljust(sidebar - 4), attr=style)
        for y in range(2, height - 2):
            put(y, sidebar, "│")
        x = sidebar + 2
        area_width = width - x - 3
        put(2, x, f"{state.action.upper()}  /  {ACTIONS[state.action_index][1]}",
            attr=cyan | curses.A_BOLD)
        put(3, x, f"Source: {state.source or '(F to select)'}")
        if state.reference:
            put(4, x, f"Reference: {state.reference}")
        else:
            put(4, x, f"Reference: {'required (B)' if state.action == 'Tide' else 'none (B)'}")
        put(5, x, f"Depth: {state.depth}    |    status: "
            + (f"running {frames[int(time.monotonic()*8)%len(frames)]}" if state.busy
               else (state.report.status if state.report else "ready")),
            attr=yellow if state.busy else green)
        put(7, x, "[ Overview ] [ Beacons ] [ Details ] [ Formats ]",
            attr=curses.A_BOLD)
        put(8, x, "─" * max(1, area_width))
        if state.report or state.page == 3:
            put(
                7, x + state.page * 11 + 1,
                ["OVERVIEW", "BEACONS", "DETAILS", "FORMATS"][state.page],
                attr=cyan | curses.A_BOLD,
            )
        body = _text_lines(view_lines(state), area_width)
        visible_height = max(0, height - 12)
        limit = max(0, len(body) - visible_height)
        state.scroll = max(0, min(limit, state.scroll))
        for i, line in enumerate(body[state.scroll:state.scroll + visible_height]):
            attr = curses.A_BOLD if line.startswith(("WELCOME", "BEACONS")) else 0
            put(9 + i, x, line, attr=attr)
        if limit:
            current_end = min(len(body), state.scroll + visible_height)
            put(height - 3, x, f"Lines {state.scroll + 1}-{current_end} / {len(body)}")
        put(height - 2, 1, "─" * (width - 3))
        put(
            height - 1, 1,
            " ↑↓ menu  Enter  F path  P browse  L formats  B base  Tab  E export  Q",
        )
        if state.notice:
            msg_color = curses.color_pair(4) if state.error and curses.has_colors() else yellow
            put(height - 3, x, " " * max(1, area_width))
            put(height - 3, x, state.notice[:area_width], attr=msg_color)
        stdscr.refresh()

    while True:
        try:
            response, error = responses.get_nowait()
            state.busy = False
            state.report = response
            state.page = 0
            state.scroll = 0
            state.error = error is not None
            state.notice = (
                error
                or (f"{response.action} complete. Tab to explore." if response else "")
            )
        except queue.Empty:
            pass
        draw()
        try:
            key = stdscr.getch()
        except curses.error:
            continue
        if key == -1:
            continue
        if key in (ord("q"), ord("Q")):
            # A running worker is daemonized and never holds terminal state.
            return 0
        if state.busy:
            continue
        if key in (curses.KEY_UP, ord("k")):
            state.action_index = (state.action_index - 1) % len(ACTIONS)
            state.report = None
            state.scroll = 0
        elif key in (curses.KEY_DOWN, ord("j")):
            state.action_index = (state.action_index + 1) % len(ACTIONS)
            state.report = None
            state.scroll = 0
        elif ord("1") <= key <= ord(str(len(ACTIONS))):
            state.action_index = key - ord("1")
            state.report = None
            state.scroll = 0
        elif key in (ord("p"), ord("P")):
            value = browse(state.source)
            if value is not None:
                state.source = value
                state.report = None
                state.notice = f"Selected {Path(value).name}"
                state.error = False
        elif key in (ord("f"), ord("F")):
            value = ask("Source file", state.source)
            if value is not None:
                state.source = value
                state.report = None
        elif key in (ord("b"), ord("B")):
            value = ask("Reference file", state.reference)
            if value is not None:
                state.reference = value
                state.report = None
        elif key in (ord("l"), ord("L")):
            state.page = 3
            state.scroll = 0
            state.notice = "Formats: Ready = installed, +extra = install needed"
        elif key in (ord("d"), ord("D")):
            state.depth = DEPTHS[(DEPTHS.index(state.depth) + 1) % len(DEPTHS)]
            state.notice = f"Depth: {state.depth}"
        elif key in (9, curses.KEY_RIGHT):
            state.page = (state.page + 1) % 4
            state.scroll = 0
        elif key == curses.KEY_LEFT:
            state.page = (state.page - 1) % 4
            state.scroll = 0
        elif key in (curses.KEY_NPAGE,):
            state.scroll += max(1, stdscr.getmaxyx()[0] - 14)
        elif key in (curses.KEY_PPAGE,):
            state.scroll = max(0, state.scroll - max(1, stdscr.getmaxyx()[0] - 14))
        elif key in (ord("e"), ord("E")):
            if state.report is None:
                state.notice = "Run a protocol before exporting."
                continue
            destination = ask("Export .json or .html", "framevitals-report.json")
            if destination:
                try:
                    output = save_report(state.report, destination)
                    state.notice = f"Saved {output.name}"
                    state.error = False
                except Exception as exc:
                    state.notice = f"Export error: {exc}"
                    state.error = True
        elif key in (10, 13, curses.KEY_ENTER, ord("r"), ord("R")):
            if not state.source.strip():
                state.notice = "Press F and set source file."
                state.error = True
                continue
            if state.action == "Tide" and not state.reference.strip():
                state.notice = "Press B to set the reference."
                state.error = True
                continue
            state.report = None
            state.busy = True
            state.started = time.monotonic()
            state.notice = "Analyzing source..."
            state.error = False
            thread = threading.Thread(
                target=worker,
                args=(state.action, state.source, state.reference, state.depth),
                daemon=True,
            )
            thread.start()
        elif key in (curses.KEY_HOME,):
            state.scroll = 0
        elif key in (curses.KEY_END,):
            state.scroll = 999_999
        elif key in (ord("x"), ord("X")):
            state.reference = ""
            state.notice = "Reference cleared."
        else:
            state.notice = "Enter run · F path · P browse · B reference · Tab views"
    return 0


def launch_terminal(*, plain: bool = False) -> int:
    """Start the interactive terminal UI, with a portable plain fallback."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            "Interactive FrameVitals requires a terminal (TTY). "
            "Use 'framevitals prism FILE' in scripts.",
            file=sys.stderr,
        )
        return 2
    if plain:
        return _run_plain()
    try:
        import curses
    except ImportError:
        return _run_plain()
    try:
        return curses.wrapper(_screen)
    except curses.error:
        return _run_plain()
