"""Barge-in provider base class + event shape.

A provider watches some external signal source (SuperWhisper recordings,
browser VAD, a push-to-talk hotkey, a mic-level silero VAD, …) and emits
``BargeinEvent``s through an ``on_event`` callback supplied at construction.
The mod3 provider registry wires that callback to the shared consumer helper
(``bargein._handle_bargein_start``), which takes the same action the legacy
``/tmp/mod3-barge-in.json`` file watcher takes today.

Concurrency: threads. Providers run their own polling loop on a daemon
thread started by ``start()`` and stopped by ``stop()``. This matches the
existing ``_bargein_watcher`` in server.py. The SuperWhisper provider's
inner loop does blocking filesystem + sqlite3 reads, so a thread is the
natural fit; an async shape would force every provider to wrap blocking
calls in ``asyncio.to_thread``.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal

from schemas.bargein import BargeinSource

BargeinEventType = Literal["user_speaking_start", "user_speaking_end"]


@dataclass
class BargeinEvent:
    """A single emission from a ``BargeinProvider``.

    ``metadata`` carries provider-specific detail (folder names, confidence
    scores, etc.) that the consumer may log but must not depend on for
    correctness.
    """

    source: BargeinSource
    event_type: BargeinEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)


BargeinCallback = Callable[[BargeinEvent], None]


class BargeinProvider(ABC):
    """Abstract barge-in provider.

    Subclasses implement ``_run`` as a blocking poll loop. ``start()`` spawns
    it on a daemon thread; ``stop()`` sets the stop-event and (best-effort)
    joins the thread.
    """

    source: BargeinSource  # class-level — subclasses set this

    def __init__(self, on_event: BargeinCallback):
        self._on_event = on_event
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the provider's background thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_guarded,
            name=f"bargein-{self.source}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal shutdown and best-effort join the thread."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _run(self) -> None:
        """Provider-specific poll loop. Must return when ``self._stop`` is set."""

    def _emit(
        self,
        event_type: BargeinEventType,
        metadata: dict | None = None,
    ) -> None:
        """Emit an event to the registered callback. Swallows callback errors."""
        try:
            self._on_event(
                BargeinEvent(
                    source=self.source,
                    event_type=event_type,
                    metadata=metadata or {},
                )
            )
        except Exception:
            # Provider must not die because the consumer threw.
            import logging

            logging.getLogger(f"bargein.{self.source}").exception("barge-in callback raised; continuing")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_guarded(self) -> None:
        """Wrap ``_run`` so an unexpected raise logs instead of vanishing silently."""
        import logging

        log = logging.getLogger(f"bargein.{self.source}")
        try:
            self._run()
        except Exception:
            log.exception("provider loop crashed")
