#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# Vendored from pipecat-ai/pipecat v1.2.1 (f8caa0da2b4c)
# Source: src/pipecat/audio/vad/silero.py
# Vendor rung: 1 (pinned copy + MANIFEST; no local patches)
# See mod3/vendor/MANIFEST.toml for provenance record.
#
# Load path adjusted: reads model from
#   vendor/pipecat_vad/data/silero_vad.onnx
# rather than the upstream importlib_resources path.
#

"""Silero Voice Activity Detection (VAD) implementation.

Adapted from pipecat-ai/pipecat for standalone use in mod3.
Reads the ONNX model from the local vendor directory.
Supports 8kHz and 16kHz sample rates.
"""

import os
import time

import numpy as np
from loguru import logger

from vendor.pipecat_vad.vad_analyzer import VADAnalyzer, VADParams

# How often should we reset internal model state (seconds)
_MODEL_RESET_STATES_TIME = 5.0

# Vendor-relative model path — resolved at import time so the class
# doesn't need to know the caller's cwd.
_VENDOR_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_PATH = os.path.join(_VENDOR_DIR, "data", "silero_vad.onnx")

try:
    import onnxruntime
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Silero VAD, you need to `pip install onnxruntime`.")
    raise Exception(f"Missing module(s): {e}")


class SileroOnnxModel:
    """ONNX runtime wrapper for the Silero VAD model."""

    def __init__(self, path: str = _DEFAULT_MODEL_PATH, force_onnx_cpu: bool = True):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        # Provider order: CoreML EP first for Apple Silicon Neural Engine / GPU
        # acceleration; CPU EP fallback on Linux/Windows or for unsupported ops.
        # ONNX Runtime silently drops unavailable providers, so this list is
        # safe cross-platform. force_onnx_cpu is preserved for callers that need
        # a strict CPU-only guarantee (e.g. reproducibility testing).
        if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(
                path,
                providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
                sess_options=opts,
            )
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()
        self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sr: int):
        if np.ndim(x) == 1:
            x = np.expand_dims(x, 0)
        if np.ndim(x) > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {np.ndim(x)}")
        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates}")
        if sr / np.shape(x)[1] > 31.25:
            raise ValueError("Input audio chunk is too short")
        return x, sr

    def reset_states(self, batch_size: int = 1):
        self._state = np.zeros((2, batch_size, 128), dtype="float32")
        self._context = np.zeros((batch_size, 0), dtype="float32")
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if np.shape(x)[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {np.shape(x)[-1]} "
                f"(Supported values: 256 for 8000 SR, 512 for 16000)"
            )

        batch_size = np.shape(x)[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if self._last_sr and self._last_sr != sr:
            self.reset_states(batch_size)
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states(batch_size)

        if not np.shape(self._context)[1]:
            self._context = np.zeros((batch_size, context_size), dtype="float32")

        x = np.concatenate((self._context, x), axis=1)

        if sr in [8000, 16000]:
            ort_inputs = {"input": x, "state": self._state, "sr": np.array(sr, dtype="int64")}
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs
            self._state = state
        else:
            raise ValueError(f"Unsupported sample rate: {sr}")

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size
        return out


class SileroVADAnalyzer(VADAnalyzer):
    """Voice Activity Detection analyzer using the Silero VAD ONNX model.

    Vendored from pipecat-ai/pipecat. Reads model from local vendor path.
    """

    def __init__(
        self,
        *,
        sample_rate: int | None = None,
        params: VADParams | None = None,
        model_path: str = _DEFAULT_MODEL_PATH,
    ):
        super().__init__(sample_rate=sample_rate, params=params)
        logger.debug(f"Loading Silero VAD model from {model_path} ...")
        self._model = SileroOnnxModel(model_path, force_onnx_cpu=True)
        self._last_reset_time = 0
        logger.debug("Loaded Silero VAD")

    def set_sample_rate(self, sample_rate: int):
        if sample_rate not in (8000, 16000):
            raise ValueError(
                f"Silero VAD sample rate must be 8000 or 16000 (got {sample_rate})"
            )
        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        return 512 if self.sample_rate == 16000 else 256

    def voice_confidence(self, buffer: bytes) -> float:
        """Return voice activity confidence [0.0, 1.0] for the given PCM buffer."""
        try:
            audio_int16 = np.frombuffer(buffer, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            new_confidence = self._model(audio_float32, self.sample_rate)[0]

            curr_time = time.time()
            if curr_time - self._last_reset_time >= _MODEL_RESET_STATES_TIME:
                self._model.reset_states()
                self._last_reset_time = curr_time

            # out shape is (batch, frames) — flatten to scalar
            return float(np.squeeze(new_confidence).item())
        except Exception as e:
            logger.error(f"Error analyzing audio with Silero VAD: {e}")
            return 0.0
