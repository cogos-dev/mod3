"""Backward-compatible alias for :mod:`ledger_sink`.

The ledger sink module was renamed from ``theseus_sink`` to ``ledger_sink``
(the "THESEUS" name was an internal codename that leaked into public source).
This shim re-exports the same public functions under the old module name so
that any existing ``import theseus_sink`` keeps working. New code should
import ``ledger_sink`` directly; this file may be removed in a future
release.
"""

from __future__ import annotations

from ledger_sink import (  # noqa: F401 -- re-exported for backward compatibility
    enabled,
    main,
    sink_turn,
)

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
