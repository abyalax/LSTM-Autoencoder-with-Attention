"""Persistence of machine-readable analysis summaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


class JsonReportWriter:
    """Write analysis summaries to JSON files."""

    def write(self, summary: Mapping[str, object], path: Path) -> None:
        """Serialize the summary to a UTF-8 JSON file, creating folders if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(dict(summary), indent=2, default=str)
        path.write_text(text, encoding="utf-8")
