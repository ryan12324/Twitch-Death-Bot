"""
Audio-based death detection via spectrogram template matching.

Loads reference death-sound WAV files, computes their mel spectrograms,
then continuously compares a rolling audio buffer against them using
normalised cross-correlation — the same idea as the image template
matcher but applied to time-frequency representations of sound.

Usage:
    detector = AudioDetector("lego_jurassic_park")
    detector.feed_audio(pcm_bytes)          # called from AudioCapture
    score = detector.get_score()            # called from DeathDetector
"""

import logging
import threading
import wave
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

AUDIO_SAMPLES_DIR = Path(__file__).parent.parent / "game_profiles" / "audio_samples"

# Audio format (must match AudioCapture output)
SAMPLE_RATE = 16000

# Spectrogram parameters
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# Keep this many seconds in the rolling buffer
BUFFER_SECONDS = 5.0
_BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_SECONDS)

# Minimum audio needed before attempting a match (seconds)
_MIN_AUDIO_SECONDS = 0.5


class AudioDetector:
    """Detects death sounds by matching audio against reference samples."""

    def __init__(self, audio_samples_dir: str):
        self.samples_path = AUDIO_SAMPLES_DIR / audio_samples_dir
        self.reference_specs: list[np.ndarray] = []
        self._mel_fb: np.ndarray | None = None  # lazily built

        self._buffer: deque[float] = deque(maxlen=_BUFFER_SAMPLES)
        self._lock = threading.Lock()
        self.last_score: float = 0.0

        self._load_samples()

    # ------------------------------------------------------------------
    # Reference sample loading
    # ------------------------------------------------------------------

    def _load_samples(self) -> None:
        if not self.samples_path.exists():
            logger.warning("No audio samples directory at %s", self.samples_path)
            return

        for wav_file in sorted(self.samples_path.glob("*.wav")):
            try:
                samples = self._load_wav(wav_file)
                spec = self._mel_spectrogram(samples)
                self.reference_specs.append(spec)
                logger.info(
                    "Loaded audio sample: %s (%d samples, spec shape %s)",
                    wav_file.name, len(samples), spec.shape,
                )
            except Exception as e:
                logger.error("Failed to load audio sample %s: %s", wav_file, e)

        logger.info("Loaded %d audio reference(s) from %s",
                     len(self.reference_specs), self.samples_path)

    @staticmethod
    def _load_wav(path: Path) -> np.ndarray:
        """Load a WAV file and return mono float32 samples at SAMPLE_RATE."""
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        # Stereo -> mono
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        # Resample if the WAV is at a different rate
        if framerate != SAMPLE_RATE:
            try:
                from scipy.signal import resample
                num_out = int(len(samples) * SAMPLE_RATE / framerate)
                samples = resample(samples, num_out).astype(np.float32)
            except ImportError:
                # Fallback: simple linear interpolation
                x_old = np.linspace(0, 1, len(samples))
                x_new = np.linspace(0, 1, int(len(samples) * SAMPLE_RATE / framerate))
                samples = np.interp(x_new, x_old, samples).astype(np.float32)

        return samples

    # ------------------------------------------------------------------
    # Mel spectrogram
    # ------------------------------------------------------------------

    def _mel_filterbank(self) -> np.ndarray:
        """Build (or return cached) mel-scale filterbank matrix."""
        if self._mel_fb is not None:
            return self._mel_fb

        n_freqs = N_FFT // 2 + 1

        def _hz2mel(hz: float) -> float:
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def _mel2hz(m: float) -> float:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        low_mel = _hz2mel(0)
        high_mel = _hz2mel(SAMPLE_RATE / 2)
        mel_pts = np.linspace(low_mel, high_mel, N_MELS + 2)
        hz_pts = np.array([_mel2hz(m) for m in mel_pts])
        bins = np.floor((N_FFT + 1) * hz_pts / SAMPLE_RATE).astype(int)

        fb = np.zeros((N_MELS, n_freqs))
        for i in range(N_MELS):
            lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
            for j in range(lo, mid):
                if j < n_freqs:
                    fb[i, j] = (j - lo) / max(mid - lo, 1)
            for j in range(mid, hi):
                if j < n_freqs:
                    fb[i, j] = (hi - j) / max(hi - mid, 1)

        self._mel_fb = fb
        return fb

    def _mel_spectrogram(self, samples: np.ndarray) -> np.ndarray:
        """Compute a log-mel spectrogram from raw float32 audio."""
        try:
            from scipy.signal import spectrogram as scipy_spectrogram
            _, _, sxx = scipy_spectrogram(
                samples,
                fs=SAMPLE_RATE,
                nperseg=N_FFT,
                noverlap=N_FFT - HOP_LENGTH,
                window="hann",
            )
        except ImportError:
            # Pure-numpy fallback (slower, no windowing)
            n_frames = 1 + (len(samples) - N_FFT) // HOP_LENGTH
            if n_frames < 1:
                return np.zeros((N_MELS, 1))
            sxx = np.zeros((N_FFT // 2 + 1, n_frames))
            window = np.hanning(N_FFT)
            for i in range(n_frames):
                start = i * HOP_LENGTH
                segment = samples[start : start + N_FFT] * window
                fft = np.fft.rfft(segment)
                sxx[:, i] = np.abs(fft) ** 2
            sxx /= N_FFT

        mel_spec = self._mel_filterbank() @ sxx
        mel_spec = np.log1p(mel_spec * 1000)  # log scale for perceptual weighting
        return mel_spec

    # ------------------------------------------------------------------
    # Online audio feed
    # ------------------------------------------------------------------

    def feed_audio(self, pcm_data: bytes) -> None:
        """Ingest raw PCM audio (s16le mono 16 kHz) into the rolling buffer."""
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            self._buffer.extend(samples.tolist())

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_score(self) -> float:
        """Return the best match score (0.0–1.0) against any reference sound.

        Safe to call from the detection thread while feed_audio() is called
        from the audio-reader thread.
        """
        if not self.reference_specs:
            return 0.0

        with self._lock:
            n = len(self._buffer)
            if n < int(SAMPLE_RATE * _MIN_AUDIO_SECONDS):
                return 0.0
            audio = np.array(self._buffer, dtype=np.float32)

        buf_spec = self._mel_spectrogram(audio)

        best = 0.0
        for ref in self.reference_specs:
            score = self._match_spectrogram(buf_spec, ref)
            if score > best:
                best = score

        self.last_score = best
        logger.debug("audio_detector: score=%.4f", best)
        return best

    @staticmethod
    def _match_spectrogram(buf_spec: np.ndarray, ref_spec: np.ndarray) -> float:
        """Slide the reference spectrogram across the buffer and return the
        best normalised cross-correlation score, mapped to 0.0–1.0.

        This is the audio equivalent of cv2.matchTemplate with TM_CCOEFF_NORMED.
        """
        buf_mels, buf_t = buf_spec.shape
        ref_mels, ref_t = ref_spec.shape

        if buf_t < ref_t or buf_mels != ref_mels:
            return 0.0

        # Normalise reference once
        ref_mean = ref_spec.mean()
        ref_std = ref_spec.std()
        if ref_std < 1e-6:
            return 0.0
        ref_norm = (ref_spec - ref_mean) / ref_std

        best_corr = 0.0
        # Slide in steps of ~25 % of the reference length for efficiency
        step = max(1, ref_t // 4)

        for t in range(0, buf_t - ref_t + 1, step):
            window = buf_spec[:, t : t + ref_t]
            w_mean = window.mean()
            w_std = window.std()
            if w_std < 1e-6:
                continue
            corr = float(np.mean((window - w_mean) / w_std * ref_norm))
            if corr > best_corr:
                best_corr = corr

        # Map raw correlation to a 0–1 score.
        # Empirically, strong matches produce correlation > 0.4;
        # weak/no-match hovers around 0.0–0.15.
        score = max(0.0, min(1.0, (best_corr - 0.1) / 0.6))
        return score
