"""Base reporter with automatic file writing to organized directory structure."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from lstm_autoencoder.reporting.console_report import ConsoleReporter
from lstm_autoencoder.reporting.json_report import JsonReportWriter

if TYPE_CHECKING:
    from lstm_autoencoder.config import DatasetConfig


class ReportWritter:
    """Base class for automatic report writing with organized directory structure.

    This class extends ConsoleReporter to automatically save reports to:
    artifacts/report/(domain_data)/(pipeline)/(class_name)/timestamp/

    The path structure is automatically detected from the module structure.
    """

    def __init__(
        self,
        config: DatasetConfig,
        reporter: ConsoleReporter | None = None,
        domain_data: str = "general",
        pipeline: str = "general",
        class_name: str | None = None,
    ) -> None:
        """Initialize the report writer with config and optional reporter.

        Args:
            config: DatasetConfig containing artifacts_dir
            reporter: Optional ConsoleReporter, defaults to new instance
            domain_data: Optional domain data name (auto-detected if not provided)
            pipeline: Optional pipeline name (auto-detected if not provided)
            class_name: Optional class name (auto-detected if not provided)
        """
        self._config = config
        self._reporter = reporter or ConsoleReporter()
        self._summary: dict[str, object] = {}
        self._domain_data = domain_data
        self._pipeline = pipeline
        self._class_name = class_name
        self._command = self._get_command_line()

    def _get_command_line(self) -> str:
        """Get the command captured by the launcher or Python invocation.

        ``LSTM_AUTOENCODER_COMMAND`` is set by the shell launcher when the
        original shell command is available. Otherwise, fall back to Python's
        original invocation arguments without fabricating a package-manager
        prefix such as ``uv run``.
        """
        captured_command = os.getenv("LSTM_AUTOENCODER_COMMAND")
        if captured_command:
            return captured_command

        command = getattr(sys, "orig_argv", None) or sys.argv

        if sys.platform == "win32":
            return subprocess.list2cmdline(command)

        return shlex.join(command)

    @property
    def reporter(self) -> ConsoleReporter:
        """Get the underlying console reporter."""
        return self._reporter

    @property
    def summary(self) -> dict[str, object]:
        """Get the current summary dictionary."""
        return self._summary

    def set_path_info_from_module(self) -> None:
        """Auto-detect domain_data, pipeline, and class_name from calling context."""
        import inspect

        # Get the calling class's module path
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            return

        # Get the class that owns the calling method
        calling_class = frame.f_back.f_locals.get("self", None)
        if calling_class is None:
            return

        # Get the module path
        module = inspect.getmodule(calling_class)
        if module is None:
            return

        module_path = Path(module.__file__)

        # Extract domain_data and pipeline from module path
        # Expected structure: src/lstm_autoencoder/data/(domain_data)/(pipeline)/class.py
        # or: src/lstm_autoencoder/evaluation/(domain_data)/(pipeline)/class.py
        parts = module_path.parts

        try:
            lstm_idx = parts.index("lstm_autoencoder")
            # Check if data or evaluation is next
            if lstm_idx + 1 < len(parts):
                section = parts[lstm_idx + 1]
                if section in ("data", "evaluation") and lstm_idx + 2 < len(parts):
                    self._domain_data = parts[lstm_idx + 2]
                    if lstm_idx + 3 < len(parts):
                        self._pipeline = parts[lstm_idx + 3].replace(".py", "")
        except ValueError:
            pass

        # Get class name
        self._class_name = calling_class.__class__.__name__

    def _get_report_path(self) -> Path:
        """Generate the report path based on module structure.

        Path format: artifacts/report/(domain_data)/(pipeline)/(class_name)/timestamp/

        Returns:
            Path object for the report directory
        """
        # Create timestamp for unique report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build full path
        report_dir = (
            self._config.artifacts_dir
            / "report"
            / self._domain_data
            / self._pipeline
            / (self._class_name or "Unknown")
            / timestamp
        )

        return report_dir

    def display_section(self, title: str) -> None:
        """Display a titled section header."""
        self._reporter.display_section(title)

    def display_text(self, text: str = "") -> None:
        """Display a plain line of text."""
        self._reporter.display_text(text)

    def display_table(self, table: pd.DataFrame) -> None:
        """Display a data frame as an aligned text table."""
        self._reporter.display_table(table)

    def display_warning(self, message: str) -> None:
        """Display a warning and remember it for the final summary."""
        self._reporter.display_warning(message)

    def update_summary(self, data: dict[str, object]) -> None:
        """Update the summary dictionary with new data.

        Args:
            data: Dictionary of data to add to summary
        """
        self._summary.update(data)

    def save_reports(
        self,
        text_filename: str = "report.txt",
        json_filename: str = "report.json",
    ) -> Path:
        """Save both text transcript and JSON summary to organized directory.

        Args:
            text_filename: Name for the text report file
            json_filename: Name for the JSON report file

        Returns:
            Path to the report directory
        """
        report_dir = self._get_report_path()

        # Save text transcript
        text_path = report_dir / text_filename
        self._reporter.save_transcript(text_path)

        # Add command to summary before saving JSON
        summary_with_command = {"command": self._command, **self._summary}

        # Save JSON summary
        json_path = report_dir / json_filename
        JsonReportWriter().write(summary_with_command, json_path)

        self._reporter.display_text(f"\nResult is saved on: {report_dir}")

        return report_dir

    def get_warnings(self) -> tuple[str, ...]:
        """Get warnings displayed so far."""
        return self._reporter.warnings
