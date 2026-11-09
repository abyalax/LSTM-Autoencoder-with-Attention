"""Utility functions for colored console output."""

from __future__ import annotations

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_WHITE = "\033[37m"


def print_header(text: str) -> None:
    """Print a colored header (bold cyan)."""
    print(f"{_BOLD}{_CYAN}{text}{_RESET}")


def print_info(text: str) -> None:
    """Print a colored info message (cyan)."""
    print(f"{_CYAN}{text}{_RESET}")


def print_success(text: str) -> None:
    """Print a colored success message (green)."""
    print(f"{_GREEN}{text}{_RESET}")


def print_warning(text: str) -> None:
    """Print a colored warning message (yellow)."""
    print(f"{_YELLOW}{text}{_RESET}")


def print_error(text: str) -> None:
    """Print a colored error message (red)."""
    print(f"{_RED}{text}{_RESET}")


def print_dim(text: str) -> None:
    """Print a dimmed text."""
    print(f"{_DIM}{text}{_RESET}")


def print_bold(text: str) -> None:
    """Print bold text."""
    print(f"{_BOLD}{text}{_RESET}")


def print_blue(text: str) -> None:
    """Print blue text."""
    print(f"{_BLUE}{text}{_RESET}")


def print_magenta(text: str) -> None:
    """Print magenta text."""
    print(f"{_MAGENTA}{text}{_RESET}")


def colorize(text: str, color: str = _CYAN, bold: bool = False) -> str:
    """Return a colored string without printing it.

    Args:
        text: The text to colorize
        color: ANSI color code (default: cyan)
        bold: Whether to make the text bold (default: False)

    Returns:
        Colorized string with reset code
    """
    prefix = _BOLD if bold else ""
    return f"{prefix}{color}{text}{_RESET}"
