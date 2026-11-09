"""Console presentation of analysis results."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class ConsoleReporter:
    """Print results to the console while keeping a plain-text transcript for later saving."""

    def __init__(self) -> None:
        """Create a reporter with an empty transcript."""
        self._lines: list[str] = []
        self._warnings: list[str] = []

    @property
    def warnings(self) -> tuple[str, ...]:
        """Warnings displayed so far, in the order they were raised."""
        return tuple(self._warnings)

    @property
    def transcript(self) -> str:
        """Everything displayed so far as one text block, with colors stripped."""
        return "\n".join(self._lines)

    def display_section(self, title: str) -> None:
        """Display a titled section header framed by thin rules."""
        self._emit("")
        self._emit(self._rule())
        self._emit(f"{_BOLD}{_CYAN}{title}{_RESET}")
        self._emit(self._rule())

    def display_text(self, text: str = "") -> None:
        """Display a plain line of text."""
        self._emit(text)

    def display_info(self, message: str) -> None:
        """Display an important informational message, highlighted without capslock."""
        self._emit(f"{_CYAN}• {message}{_RESET}")

    def display_table(self, table: pd.DataFrame) -> None:
        """Display a data frame as a boxed grid table."""
        self._emit(self._render_grid_table(table))

    def display_warning(self, message: str) -> None:
        """Display a warning and remember it for the final summary."""
        self._warnings.append(message)
        self._emit(f"{_YELLOW}⚠ {message}{_RESET}")

    def save_transcript(self, path: Path) -> None:
        """Write the plain-text transcript to a UTF-8 file, creating folders if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.transcript, encoding="utf-8")

    def _emit(self, text: str) -> None:
        print(text)
        self._lines.append(self._strip_ansi(text))

    @staticmethod
    def _strip_ansi(text: str) -> str:
        return _ANSI_RE.sub("", text)

    @staticmethod
    def _rule(width: int | None = None) -> str:
        width = width or shutil.get_terminal_size(fallback=(80, 20)).columns
        return _DIM + ("-" * width) + _RESET

    @staticmethod
    def _render_grid_table(table: pd.DataFrame) -> str:
        """Render a DataFrame as a boxed grid table using thin line-drawing characters."""
        headers = [str(table.index.name or "")] + [str(c) for c in table.columns]
        rows = [[str(idx)] + [str(v) for v in row] for idx, row in zip(table.index, table.values, strict=False)]

        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def border(left: str, mid: str, right: str) -> str:
            return left + mid.join("─" * (w + 2) for w in widths) + right

        def format_row(cells: list[str]) -> str:
            padded = (f" {cell.ljust(w)} " for cell, w in zip(cells, widths, strict=False))
            return "│" + "│".join(padded) + "│"

        lines = [border("┌", "┬", "┐"), format_row(headers), border("├", "┼", "┤")]
        lines.extend(format_row(row) for row in rows)
        lines.append(border("└", "┴", "┘"))
        return "\n".join(lines)
