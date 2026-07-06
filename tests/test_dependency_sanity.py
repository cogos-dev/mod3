"""Dependency-sanity checks — pins that must hold for imports to succeed.

These are cheap import-only checks (no model load, no HTTP server) for the two
failure modes hit during the 2026-07-02 deploy-worktree cutover attempt, where
a fresh install from requirements.txt drifted from the known-working dev venv:

  1. `torchaudio` was used by vad.py's silero-vad path but never declared in
     requirements.txt — a fresh install silently produced modalities.vad=false
     instead of failing loudly.
  2. Unpinned `transformers` floor resolved to 5.13.0 on a fresh install vs.
     5.9.0 in the dev venv. `mlx_lm.tokenizer_utils` registers a custom
     tokenizer class at import time via
     `AutoTokenizer.register("NewlineTokenizer", ...)`, passing a bare string
     as the first arg. transformers>=5.10's `register()` does `key.__module__`
     on that argument and raises `AttributeError`, which breaks the
     chatterbox-turbo model load path (HTTP 500 on /v1/speak and
     /v1/synthesize) entirely.

Both failures reproduce on plain `import`, with no network access and no
model weights required, which makes this the cheapest possible regression
guard against a future unpinned-floor drift.

Run: python3 -m pytest tests/test_dependency_sanity.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import skip_without_mlx  # noqa: E402


class TestTorchaudioAvailable:
    """VAD (vad.py) lazily imports torchaudio.functional for silero-vad resampling."""

    def test_torchaudio_importable(self):
        import torchaudio  # noqa: F401

    def test_torchaudio_functional_importable(self):
        # Exact import used by vad.py and mod3/worker/stt.py.
        import torchaudio.functional as F  # noqa: F401

        assert hasattr(F, "resample")


@skip_without_mlx
class TestMlxLmTokenizerRegistration:
    """mlx_lm's tokenizer_utils registers a custom tokenizer at import time.

    This is the cheapest possible reproduction of the chatterbox-turbo HTTP 500:
    a bad transformers version breaks this on bare `import mlx_lm`, before any
    model is ever loaded.
    """

    def test_mlx_lm_imports_without_error(self):
        # Regression guard: transformers>=5.10 raises
        # AttributeError: 'str' object has no attribute '__module__'
        # from within AutoTokenizer.register() at mlx_lm import time.
        import mlx_lm  # noqa: F401

    def test_newline_tokenizer_class_defined(self):
        # Sanity: the class the register() call above depends on is still
        # present under the name mlx_lm expects, in case the import above
        # ever starts silently swallowing the AttributeError somewhere
        # upstream instead of raising it.
        import mlx_lm.tokenizer_utils as tokenizer_utils

        assert hasattr(tokenizer_utils, "NewlineTokenizer")
