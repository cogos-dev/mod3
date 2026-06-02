"""Per-channel output queue — serial execution, non-blocking submission.

Each channel gets its own queue. Multiple speak() calls to the same
channel execute sequentially. Different channels run concurrently.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class QueuedJob:
    id: str
    channel: str
    submitted_at: float
    started_at: float | None = None
    finished_at: float | None = None
    status: str = "queued"  # queued, running, done, error
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChannelQueue:
    """Serial execution queue for a single channel."""

    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self._queue: deque[tuple[QueuedJob, Callable]] = deque()
        self._lock = threading.Lock()
        self._running = False
        self._current: QueuedJob | None = None

    @property
    def depth(self) -> int:
        return len(self._queue) + (1 if self._running else 0)

    @property
    def current_job(self) -> QueuedJob | None:
        return self._current

    def submit(self, fn: Callable, **metadata) -> QueuedJob:
        """Submit a job. Returns immediately with the job handle."""
        job = QueuedJob(
            id=uuid.uuid4().hex[:8],
            channel=self.channel_id,
            submitted_at=time.time(),
            metadata=metadata,
        )
        with self._lock:
            self._queue.append((job, fn))
            if not self._running:
                self._running = True
                threading.Thread(target=self._drain, daemon=True).start()
        return job

    def _drain(self):
        """Process jobs sequentially until queue is empty.

        The outer try/finally ensures ``_running`` is always reset to False
        when this thread exits — regardless of whether the exit is normal
        (queue emptied) or abnormal (BaseException raised by a job function or
        a signal handler).  Without the finally, a BaseException leaves
        ``_running = True`` permanently; the ``depth`` property then reports
        ``len(self._queue) + 1`` forever (the ghost of the last killed job),
        and subsequent ``submit()`` calls never start a new drain thread.

        Signals (KeyboardInterrupt, SystemExit) are re-raised after cleanup.
        """
        try:
            while True:
                with self._lock:
                    if not self._queue:
                        self._running = False
                        self._current = None
                        return
                    job, fn = self._queue.popleft()

                job.status = "running"
                job.started_at = time.time()
                self._current = job

                try:
                    job.result = fn()
                    job.status = "done"
                except Exception as e:
                    job.error = str(e)
                    job.status = "error"
                finally:
                    job.finished_at = time.time()
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            # Unconditional cleanup: reset flags so future submit() calls
            # start a new drain thread instead of silently accumulating.
            with self._lock:
                self._running = False
                self._current = None


class OutputQueueManager:
    """Manages per-channel output queues."""

    def __init__(self):
        self._queues: dict[str, ChannelQueue] = {}
        self._lock = threading.Lock()

    def get_queue(self, channel_id: str) -> ChannelQueue:
        if channel_id not in self._queues:
            with self._lock:
                if channel_id not in self._queues:
                    self._queues[channel_id] = ChannelQueue(channel_id)
        return self._queues[channel_id]

    def submit(self, channel_id: str, fn: Callable, **metadata) -> QueuedJob:
        """Submit a job to a channel's queue. Non-blocking."""
        return self.get_queue(channel_id).submit(fn, **metadata)

    def cancel_channel(self, channel_id: str) -> int:
        """Cancel all pending jobs for a channel. Returns number of jobs cancelled."""
        queue = self._queues.get(channel_id)
        if not queue:
            return 0
        with queue._lock:
            cancelled = len(queue._queue)
            queue._queue.clear()
        return cancelled

    def drop_queue(self, channel_id: str) -> bool:
        """Remove the ChannelQueue for a channel after its jobs are cancelled.

        The drain thread (if any) terminates naturally on the next iteration
        once its deque is empty and the running flag flips. Callers should
        invoke ``cancel_channel`` first; this method then frees the
        ChannelQueue reference so the channel name can be re-used without
        accumulating stale queue state across reconnects. Returns True if a
        queue existed and was removed.
        """
        with self._lock:
            return self._queues.pop(channel_id, None) is not None

    def status(self) -> dict[str, Any]:
        """Snapshot of all channel queues."""
        return {
            cid: {
                "depth": q.depth,
                "current": q.current_job.id if q.current_job else None,
            }
            for cid, q in self._queues.items()
        }
