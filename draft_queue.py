"""Draft Queue — speculative response blocks with status tracking.

Holds draft response blocks generated speculatively while the human is
still speaking. Each block has a status lifecycle:

    valid → spoken (played aloud)
    valid → stale  (invalidated by new context)
    valid → snipped (removed from queue by self-barge)

Thread-safe. Used by the agent loop for speculative inference and
self-barge operations (snip, inject, revise).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlockStatus(Enum):
    """Lifecycle states for a draft block."""

    VALID = "valid"  # Generated, awaiting playback
    STALE = "stale"  # Invalidated by new context
    SPOKEN = "spoken"  # Successfully played aloud
    SNIPPED = "snipped"  # Removed by self-barge
    SPEAKING = "speaking"  # Currently being spoken


@dataclass
class DraftBlock:
    """A single draft response block with metadata."""

    id: str
    text: str
    status: BlockStatus = BlockStatus.VALID
    created_at: float = field(default_factory=time.time)
    context_hash: str = ""  # Hash of context at generation time
    generation_ms: float = 0.0  # How long inference took
    tts_audio: bytes | None = None  # Pre-synthesized audio (if available)
    tts_duration_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_playable(self) -> bool:
        """Whether this block can be played."""
        return self.status == BlockStatus.VALID

    @property
    def is_active(self) -> bool:
        """Whether this block is still relevant (not stale/snipped)."""
        return self.status in (BlockStatus.VALID, BlockStatus.SPEAKING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status.value,
            "created_at": self.created_at,
            "has_audio": self.tts_audio is not None,
            "tts_duration_sec": self.tts_duration_sec,
            "generation_ms": self.generation_ms,
        }


class DraftQueue:
    """Thread-safe queue of speculative draft response blocks.

    The agent generates blocks speculatively while the human speaks.
    Blocks are played in order when the human stops. Blocks can be
    invalidated (stale), removed (snip), or replaced (revise) before
    they're spoken.

    Operations:
        add_block     — append a new draft block
        invalidate    — mark a block as stale (context changed)
        snip          — remove a block from the queue
        inject        — insert a new block at a position
        revise        — replace a block's text (and optionally audio)
        get_pending   — get all valid blocks awaiting playback
        mark_speaking — mark a block as currently being spoken
        mark_spoken   — mark a block as successfully spoken
        clear         — reset the queue
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._blocks: list[DraftBlock] = []
        self._spoken_history: list[DraftBlock] = []  # Archive of spoken blocks

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_block(
        self,
        text: str,
        context_hash: str = "",
        generation_ms: float = 0.0,
        **metadata,
    ) -> DraftBlock:
        """Add a new draft block to the end of the queue."""
        block = DraftBlock(
            id=uuid.uuid4().hex[:8],
            text=text,
            context_hash=context_hash,
            generation_ms=generation_ms,
            metadata=metadata,
        )
        with self._lock:
            self._blocks.append(block)
        return block

    def invalidate(self, block_id: str) -> bool:
        """Mark a block as stale. Returns True if found and invalidated."""
        with self._lock:
            for block in self._blocks:
                if block.id == block_id and block.is_active:
                    block.status = BlockStatus.STALE
                    return True
        return False

    def invalidate_all(self) -> int:
        """Mark all valid blocks as stale. Returns count invalidated."""
        count = 0
        with self._lock:
            for block in self._blocks:
                if block.status == BlockStatus.VALID:
                    block.status = BlockStatus.STALE
                    count += 1
        return count

    def snip(self, block_id: str) -> bool:
        """Remove a block from the queue. Returns True if found."""
        with self._lock:
            for i, block in enumerate(self._blocks):
                if block.id == block_id:
                    block.status = BlockStatus.SNIPPED
                    self._blocks.pop(i)
                    return True
        return False

    def inject(
        self,
        position: int,
        text: str,
        context_hash: str = "",
        generation_ms: float = 0.0,
        **metadata,
    ) -> DraftBlock:
        """Insert a new block at the given position."""
        block = DraftBlock(
            id=uuid.uuid4().hex[:8],
            text=text,
            context_hash=context_hash,
            generation_ms=generation_ms,
            metadata=metadata,
        )
        with self._lock:
            self._blocks.insert(position, block)
        return block

    def revise(
        self,
        block_id: str,
        new_text: str,
        new_audio: bytes | None = None,
        new_duration: float = 0.0,
    ) -> bool:
        """Replace a block's content. Returns True if found and revised."""
        with self._lock:
            for block in self._blocks:
                if block.id == block_id and block.is_active:
                    block.text = new_text
                    if new_audio is not None:
                        block.tts_audio = new_audio
                        block.tts_duration_sec = new_duration
                    block.metadata["revised_at"] = time.time()
                    return True
        return False

    # ------------------------------------------------------------------
    # Playback lifecycle
    # ------------------------------------------------------------------

    def get_pending(self) -> list[DraftBlock]:
        """Get all valid blocks awaiting playback, in order."""
        with self._lock:
            return [b for b in self._blocks if b.status == BlockStatus.VALID]

    def get_next(self) -> DraftBlock | None:
        """Get the next valid block to play, or None."""
        with self._lock:
            for block in self._blocks:
                if block.status == BlockStatus.VALID:
                    return block
        return None

    def mark_speaking(self, block_id: str) -> bool:
        """Mark a block as currently being spoken."""
        with self._lock:
            for block in self._blocks:
                if block.id == block_id:
                    block.status = BlockStatus.SPEAKING
                    return True
        return False

    def mark_spoken(self, block_id: str) -> bool:
        """Mark a block as successfully spoken and archive it."""
        with self._lock:
            for i, block in enumerate(self._blocks):
                if block.id == block_id:
                    block.status = BlockStatus.SPOKEN
                    self._spoken_history.append(block)
                    self._blocks.pop(i)
                    return True
        return False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Number of blocks in the queue (all statuses)."""
        with self._lock:
            return len(self._blocks)

    @property
    def pending_count(self) -> int:
        """Number of valid (playable) blocks."""
        with self._lock:
            return sum(1 for b in self._blocks if b.status == BlockStatus.VALID)

    @property
    def all_blocks(self) -> list[DraftBlock]:
        """Snapshot of all blocks in current queue."""
        with self._lock:
            return list(self._blocks)

    @property
    def spoken_text(self) -> str:
        """All text that has been successfully spoken."""
        with self._lock:
            return " ".join(b.text for b in self._spoken_history)

    def clear(self) -> int:
        """Clear the queue. Returns number of blocks removed."""
        with self._lock:
            count = len(self._blocks)
            self._blocks.clear()
            return count

    def status(self) -> dict[str, Any]:
        """Queue status snapshot."""
        with self._lock:
            return {
                "total": len(self._blocks),
                "valid": sum(1 for b in self._blocks if b.status == BlockStatus.VALID),
                "stale": sum(1 for b in self._blocks if b.status == BlockStatus.STALE),
                "speaking": sum(1 for b in self._blocks if b.status == BlockStatus.SPEAKING),
                "spoken_total": len(self._spoken_history),
                "blocks": [b.to_dict() for b in self._blocks],
            }
