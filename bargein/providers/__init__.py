"""Barge-in providers.

Each provider watches a different signal source (SuperWhisper, browser VAD,
hotkey, mic-level VAD, …) and emits ``BargeinEvent`` through a callback.
"""

from .base import BargeinCallback, BargeinEvent, BargeinEventType, BargeinProvider

__all__ = [
    "BargeinCallback",
    "BargeinEvent",
    "BargeinEventType",
    "BargeinProvider",
]
