"""Command activity on stderr, leaving JSON results on stdout."""

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.text import Text

MUSIC_FRAMES = ("♪", "♫", "♬", "♫")
ASCII_FRAMES = ("|", "/", "-", "\\")


@contextmanager
def command_status(label: str, *, frames: tuple[str, ...] = MUSIC_FRAMES) -> Iterator[None]:
    """Show an interchangeable animation only when both output streams are terminals."""
    console = Console(stderr=True)
    if not sys.stdout.isatty() or not console.is_terminal or console.is_dumb_terminal:
        print(label, file=sys.stderr)
        yield
        return

    try:
        "".join(frames).encode(console.encoding)
    except (UnicodeError, LookupError):
        frames = ASCII_FRAMES
    started = time.monotonic()

    def render() -> Text:
        elapsed = time.monotonic() - started
        frame = frames[int(elapsed * 5) % len(frames)]
        return Text(f"{frame} {label} ({int(elapsed)}s)")

    with Live(
        render(), console=console, get_renderable=render, refresh_per_second=5,
        transient=True, redirect_stdout=False, redirect_stderr=True,
    ):
        try:
            yield
        finally:
            sys.stderr.flush()
