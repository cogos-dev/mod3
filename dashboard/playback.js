/**
 * Streaming audio playback engine.
 * Receives Int16 PCM chunks and plays them seamlessly via Web Audio API.
 * Tracks playback progress (samplesPlayed/totalSamples) for word-level
 * solidification animation.
 */
class AudioPlayback {
  constructor(sampleRate = 24000) {
    this.sampleRate = sampleRate;
    this.queue = [];
    this.isPlaying = false;
    this.audioContext = null;
    this.currentSource = null;
    this.nextStartTime = 0;
    this.onPlaybackStart = null;
    this.onPlaybackEnd = null;
    this.onProgress = null;  // (samplesPlayed, totalSamples) => void
    this.sinkId = undefined; // output device ID

    // Progress tracking
    this.totalSamples = 0;       // Total samples across all queued buffers
    this.samplesPlayed = 0;      // Samples played so far
    this._chunkStartSample = 0;  // Sample offset of current chunk
    this._currentChunkSamples = 0;
    this._playbackStartTime = 0; // audioContext.currentTime when chunk started
    this._progressTimer = null;
  }

  /** Current playback progress as 0.0-1.0 */
  get progress() {
    if (this.totalSamples === 0) return 0;
    return Math.min(1.0, this.samplesPlayed / this.totalSamples);
  }

  /** Estimated current playback time in seconds */
  get currentTime() {
    return this.samplesPlayed / this.sampleRate;
  }

  /** Total duration in seconds of all queued audio */
  get totalDuration() {
    return this.totalSamples / this.sampleRate;
  }

  _ensureContext() {
    if (!this.audioContext || this.audioContext.state === "closed") {
      this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
      // Apply selected output device if set
      if (this.sinkId !== undefined && this.audioContext.setSinkId) {
        this.audioContext.setSinkId(this.sinkId).catch(() => {});
      }
    }
    if (this.audioContext.state === "suspended") {
      this.audioContext.resume();
    }
  }

  async setOutputDevice(deviceId) {
    this.sinkId = deviceId;
    if (this.audioContext && this.audioContext.setSinkId) {
      await this.audioContext.setSinkId(deviceId);
    }
  }

  /** Enqueue raw Int16 PCM for playback. */
  enqueue(pcmArrayBuffer) {
    this._ensureContext();

    const int16 = new Int16Array(pcmArrayBuffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    const buffer = this.audioContext.createBuffer(1, float32.length, this.sampleRate);
    buffer.getChannelData(0).set(float32);
    this.queue.push(buffer);
    this.totalSamples += float32.length;

    if (!this.isPlaying) this._playNext();
  }

  /** Enqueue a full WAV file (with header) for playback. */
  async enqueueWav(wavArrayBuffer) {
    this._ensureContext();
    try {
      const audioBuffer = await this.audioContext.decodeAudioData(wavArrayBuffer.slice(0));
      this.queue.push(audioBuffer);
      this.totalSamples += audioBuffer.length;
      if (!this.isPlaying) this._playNext();
    } catch (err) {
      console.error("[AudioPlayback] Failed to decode WAV:", err);
    }
  }

  _playNext() {
    if (this.queue.length === 0) {
      this.isPlaying = false;
      this._stopProgressTimer();
      this.samplesPlayed = this.totalSamples; // Mark fully played
      if (this.onProgress) this.onProgress(this.samplesPlayed, this.totalSamples);
      if (this.onPlaybackEnd) this.onPlaybackEnd();
      return;
    }

    if (!this.isPlaying) {
      this.isPlaying = true;
      this.nextStartTime = this.audioContext.currentTime;
      if (this.onPlaybackStart) this.onPlaybackStart();
    }

    const buffer = this.queue.shift();
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);

    // Track progress for this chunk
    this._chunkStartSample = this.samplesPlayed;
    this._currentChunkSamples = buffer.length;

    source.onended = () => {
      // Mark chunk as fully played
      this.samplesPlayed = this._chunkStartSample + this._currentChunkSamples;
      if (this.onProgress) this.onProgress(this.samplesPlayed, this.totalSamples);
      this._playNext();
    };

    // Schedule this chunk right after the previous one for gapless playback
    const startTime = Math.max(this.nextStartTime, this.audioContext.currentTime);
    source.start(startTime);
    this._playbackStartTime = startTime;
    this.nextStartTime = startTime + buffer.duration;
    this.currentSource = source;

    // Start progress timer for smooth updates during playback
    this._startProgressTimer();
  }

  _startProgressTimer() {
    this._stopProgressTimer();
    this._progressTimer = setInterval(() => {
      if (!this.isPlaying || !this.audioContext) return;
      const elapsed = this.audioContext.currentTime - this._playbackStartTime;
      const chunkProgress = Math.min(elapsed * this.sampleRate, this._currentChunkSamples);
      this.samplesPlayed = this._chunkStartSample + Math.floor(chunkProgress);
      if (this.onProgress) this.onProgress(this.samplesPlayed, this.totalSamples);
    }, 50); // 20 fps progress updates
  }

  _stopProgressTimer() {
    if (this._progressTimer) {
      clearInterval(this._progressTimer);
      this._progressTimer = null;
    }
  }

  flush() {
    this.queue = [];
    this._stopProgressTimer();
    if (this.currentSource) {
      try { this.currentSource.stop(); } catch {}
    }
    this.isPlaying = false;
    this.nextStartTime = 0;
    // Keep samplesPlayed/totalSamples for interrupt context
    // (tells us how much was delivered before flush)
  }

  /** Reset all progress counters (call when starting a new response) */
  resetProgress() {
    this.totalSamples = 0;
    this.samplesPlayed = 0;
    this._chunkStartSample = 0;
    this._currentChunkSamples = 0;
  }

  setSampleRate(rate) {
    if (rate !== this.sampleRate) {
      this.sampleRate = rate;
      // Close old context so next enqueue creates one with the right rate
      if (this.audioContext) {
        this.audioContext.close();
        this.audioContext = null;
      }
    }
  }
}
