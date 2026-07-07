"""Real-time microphone audio analysis with frequency band extraction.

MicrophoneAnalyzer captures audio from a sounddevice input stream, computes
FFT spectra at ~40 FPS, and maintains circular-buffer histories for 32
logarithmic frequency bands. Provides both raw and normalized (short-term /
long-term) band power for shader effects to consume via get_extended_analysis().

SpectrogramPlotter (matplotlib) is included for standalone diagnostic use.
"""

import sounddevice as sd
import numpy as np
import os
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from math import gcd
from queue import Queue
from scipy import signal

# Optional: the `soundcard` library provides true cross-platform output
# loopback (WASAPI loopback on Windows, PulseAudio/PipeWire monitor on Linux)
# of ANY output device - something standard sounddevice/PortAudio cannot do.
# When present it backs the "loopback" source; otherwise we fall back to a
# Stereo Mix / monitor INPUT device discovered via sounddevice.
try:
    import soundcard as _soundcard
except Exception:
    _soundcard = None
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class CircularBuffer:
    """
    Fast circular buffer for storing time-series data
    Replaces expensive np.roll operations with index manipulation
    """
    def __init__(self, shape, dtype=np.float64):
        """
        Initialize circular buffer
        
        Args:
            shape: Shape of buffer. If tuple, creates 2D buffer (history_len, feature_dim)
                   If int, creates 1D buffer (history_len,)
            dtype: Data type for the buffer
        """
        if isinstance(shape, int):
            shape = (shape,)
        
        self.buffer = np.zeros(shape, dtype=dtype)
        self.max_len = shape[0]
        self.write_idx = 0
        self.filled = 0
        self._lock = threading.Lock()
    
    def append(self, data):
        """
        Add new data to the buffer (most recent)
        Thread-safe operation
        
        Args:
            data: New data to add (should match buffer shape except first dimension)
        """
        with self._lock:
            self.buffer[self.write_idx] = data
            self.write_idx = (self.write_idx + 1) % self.max_len
            self.filled = min(self.filled + 1, self.max_len)
    
    def get_ordered(self, n=None):
        """
        Get data in chronological order (most recent first)
        Thread-safe operation
        
        Args:
            n: Number of most recent items to return (None = all available)
            
        Returns:
            Array with most recent data at index 0
        """
        with self._lock:
            if self.filled == 0:
                return np.array([])
            
            n = min(n or self.filled, self.filled)
            
            if self.filled < self.max_len:
                # Buffer not full yet, return in reverse order
                return self.buffer[:self.filled][::-1][:n]
            
            # Buffer is full, need to reorder
            # Most recent item is at write_idx - 1
            indices = [(self.write_idx - 1 - i) % self.max_len for i in range(n)]
            return self.buffer[indices]
    
    def get_latest(self):
        """
        Get the most recent item
        Thread-safe operation
        
        Returns:
            Most recently added data item
        """
        with self._lock:
            if self.filled == 0:
                return None
            idx = (self.write_idx - 1) % self.max_len
            return self.buffer[idx].copy()
    
    def __len__(self):
        """Return number of items currently in buffer"""
        return self.filled
    
    def is_full(self):
        """Check if buffer is full"""
        return self.filled >= self.max_len

def find_loopback_device():
    """Find a loopback device (system audio output) on Windows"""
    devices = sd.query_devices()
    
    # On Windows, look for devices with "Stereo Mix", "Wave Out", "Loopback", or similar
    loopback_keywords = ['stereo mix', 'wave out', 'loopback', 'what u hear', 'what you hear']
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            device_name_lower = device['name'].lower()
            for keyword in loopback_keywords:
                if keyword in device_name_lower:
                    return i, device
    
    # If no explicit loopback device found, return None
    return None, None

def list_audio_devices_detailed():
    """Print all available audio devices with detailed information"""
    devices = sd.query_devices()
    print("\nAvailable Audio Devices:")
    print("-" * 80)
    for i, device in enumerate(devices):
        input_channels = device['max_input_channels']
        output_channels = device['max_output_channels']
        if input_channels > 0:  # Only show input-capable devices
            device_type = "INPUT"
            if any(keyword in device['name'].lower() for keyword in ['stereo mix', 'loopback', 'wave out']):
                device_type = "LOOPBACK"
            
            print(f"[{device_type}] Device ID {i}: {device['name']}")
            print(f"    Input channels: {input_channels}, Output channels: {output_channels}")
            print(f"    Sample rate: {device['default_samplerate']}")
            print()
    print("-" * 80)

def list_audio_devices():
    """Print all available audio devices and their properties"""
    devices = sd.query_devices()
    # print("\nAvailable Audio Devices:")
    # print("-" * 80)
    # for i, device in enumerate(devices):
    #     input_channels = device['max_input_channels']
    #     if input_channels > 0:  # Only show input devices
    #         print(f"Device ID {i}: {device['name']}")
    #         print(f"    Input channels: {input_channels}")
    #         print(f"    Sample rates: {device['default_samplerate']}")
    #         try:
    #             sd.check_input_settings(device=i)
    #             print(f"    Status: Available")
    #         except Exception as e:
    #             print(f"    Status: Unavailable ({str(e)})")
    #         print()
    # print("-" * 80)
    return devices

def find_device_by_name(name_fragment):
    """Find first device containing the given name fragment (case insensitive)"""
    devices = sd.query_devices()
    name_fragment = name_fragment.lower()
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            if name_fragment in device['name'].lower():
                return i, device
    return None, None


class MicrophoneAnalyzer:
    def __init__(self, device=None, device_name=None, use_loopback=False,
                 source=None, linein_device=None, loopback_device=None,
                 avg_window_short=20, avg_window_long=100,
                 use_exponential=False, ema_alpha_short=0.05, ema_alpha_long=0.01):
        """
        Initialize the microphone analyzer
        
        Args:
            device: Explicit device ID override (None = resolve from source)
            device_name: Name fragment for the microphone source
            use_loopback: Legacy flag; equivalent to source="loopback"
            source: Active input source - "microphone" | "linein" | "loopback"
                | "internal" (default "microphone", or "loopback" if use_loopback)
            linein_device: Name fragment for the wired line/aux input
            loopback_device: Name fragment to force a specific loopback/monitor
            avg_window_short: Number of frames for short-term average (default 20 = 0.5s at 40fps)
            avg_window_long: Number of frames for long-term average (default 100 = 2.5s at 40fps)
            use_exponential: Use exponential moving average instead of simple mean
            ema_alpha_short: Alpha for short-term EMA (higher = faster response, 0-1)
            ema_alpha_long: Alpha for long-term EMA (lower = slower response, 0-1)
        """
        # Source selection. One active input source at a time, switchable at
        # runtime via set_source(). The FFT/band pipeline runs at a FIXED
        # target rate (self.RATE); every source downmixes to mono and resamples
        # to it, so switching sources never invalidates the band masks.
        #   microphone : input device matched by `device_name`
        #   linein     : wired line/aux input matched by `linein_device`
        #   loopback   : system output - WASAPI loopback (Windows) / monitor
        #                source (Linux) / Stereo Mix fallback
        #   internal   : the show's own AudioEngine mix, pushed in via feed()
        # Legacy: use_loopback=True maps to source="loopback".
        if source is None:
            source = "loopback" if use_loopback else "microphone"
        self.source = source
        self._device_names = {
            "microphone": device_name,
            "linein": linein_device,
            "loopback": loopback_device,
            # Set live by Stories_OGL from BluetoothAudioReceiver.source_hint()
            # when a phone connects over A2DP (the underscored MAC, e.g.
            # "AA_BB_CC_DD_EE_FF"); the BlueZ node name contains it. Empty
            # until then; the resolver falls back to the generic "bluez" token.
            "bluetooth": None,
        }
        self.RATE = 44100              # fixed analysis target rate
        self.device = device           # optional explicit device-id override
        self._src_rate = self.RATE     # native rate of the open device stream
        self._capture_channels = 1     # channels captured from the device
        self._active_source = None     # set by _open_source()
        self._sc_thread = None         # soundcard loopback capture thread
        self._sc_stop = threading.Event()
        self._pa_thread = None         # parec (PipeWire/Pulse node) capture thread
        self._pa_stop = threading.Event()
        self._pa_proc = None           # running parec subprocess, if any
        print(f"[Audio] analyzer source='{source}' (target {self.RATE} Hz)")

        # Averaging parameters
        self.avg_window_short = avg_window_short
        self.avg_window_long = avg_window_long
        self.use_exponential = use_exponential
        self.ema_alpha_short = ema_alpha_short
        self.ema_alpha_long = ema_alpha_long
        
        # Initialize EMA accumulators if using exponential averaging
        if self.use_exponential:
            self.ema_short = None  # Will initialize on first frame
            self.ema_long = None
            print(f"Using Exponential Moving Average:")
            print(f"  Short-term alpha: {ema_alpha_short} (~{1/ema_alpha_short:.0f} frame time constant)")
            print(f"  Long-term alpha: {ema_alpha_long} (~{1/ema_alpha_long:.0f} frame time constant)")
        else:
            print(f"Using Simple Mean Average:")
            print(f"  Short-term window: {avg_window_short} frames ({avg_window_short/40:.2f}s)")
            print(f"  Long-term window: {avg_window_long} frames ({avg_window_long/40:.2f}s)")
        
        print("-" * 80 + "\n")
        
        # Sensitivity multiplier (controlled via web UI)
        self._sensitivity = 1.0

        # Noise gate. The normalized band outputs are RATIOS (power / its own
        # running average) - an AGC that rides its gain down into the noise
        # floor, so a silent input's hiss normalizes to ~1.0, indistinguishable
        # from steady music. The gate tracks the input's noise floor and
        # closes the normalized outputs when the level is at/near it, so
        # silence actually reads as silence downstream.
        self._NF_CEIL = 0.01     # floor estimate ceiling: music can never become "the floor"
        self._NF_INIT = 3e-4     # START sensitive (near digital silence) and
                                 # let the slow up-creep raise the floor only
                                 # if the input genuinely carries sustained
                                 # floor noise - NOT the other way round.
        self._noise_floor = self._NF_INIT  # adaptive floor estimate (RMS)
        self._rms_smooth = 0.0   # ~0.25s-smoothed input RMS the gate acts on
        self._gate = 0.0         # smoothed gate factor, 0 (closed) .. 1 (open)

        # Audio parameters
        self.CHUNK = 4096  # FFT size for good frequency resolution
        self.CALLBACK_BLOCKSIZE = 512  # Even smaller blocks for smoother updates (11.6ms at 44.1kHz)
        self.OVERLAP = self.CHUNK - self.CALLBACK_BLOCKSIZE
        self.CHANNELS = 1
        self.device = device
        
        # Audio buffer for overlapping windows
        self.audio_buffer = np.zeros(self.CHUNK)
        self.new_data_available = False
        
        
        # Analysis rate: 40 FPS
        self.FPS = 40
        self.frame_time = 1.0 / self.FPS

        # Analysis storage and threading
        self.running = False
        self.analysis_thread = None
        self.stream = None

        # Window function for FFT
        self.window = signal.windows.hann(self.CHUNK)

        # Bass detection parameters
        self.bass_range = (60, 180)
        
        # Prepare frequency analysis arrays
        self.freq_bins = np.fft.rfftfreq(self.CHUNK, 1/self.RATE)
        self.bass_mask = (self.freq_bins >= self.bass_range[0]) & (self.freq_bins <= self.bass_range[1])
        
        # Store spectrum history using circular buffer (5 seconds at 40Hz = 200 frames)
        self.history_len = 200
        self._spectrum_lock = threading.Lock()
        self.spectrum_history = CircularBuffer((self.history_len, len(self.freq_bins)))
        
        # Maximum tracking for normalization
        self.max_magnitude = 1e-10
        self.max_decay = 0.999
        
        # Extended analysis features
        self.num_bands = 32
        self.band_history_len = 1000
        
        # Create logarithmic frequency bands from 40Hz to 16kHz (more practical range)
        # This avoids subsonic frequencies and ultrasonic noise
        self.band_edges = np.logspace(np.log10(40), np.log10(16000), self.num_bands + 1)
        self.band_centers = np.sqrt(self.band_edges[:-1] * self.band_edges[1:])
        
        # Create masks for each band, ensuring we skip DC component (bin 0)
        self.band_masks = []
        for i in range(self.num_bands):
            mask = (self.freq_bins >= self.band_edges[i]) & (self.freq_bins < self.band_edges[i+1])
            # Ensure we have at least one bin per band
            if not np.any(mask):
                # If band is too narrow, expand it slightly
                mask = (self.freq_bins >= self.band_edges[i] * 0.9) & (self.freq_bins < self.band_edges[i+1] * 1.1)
            self.band_masks.append(mask)
        
        # Debug: Print band information to verify coverage
        print("\nFrequency Band Information:")
        print(f"FFT bin resolution: {self.RATE/self.CHUNK:.2f} Hz per bin")
        print(f"Usable frequency range: {self.RATE/self.CHUNK:.1f} Hz to {self.RATE/2:.1f} Hz")
        for i in range(min(5, self.num_bands)):  # Show first 5 bands
            mask = self.band_masks[i]
            num_bins = np.sum(mask)
            if num_bins > 0:
                bin_range = f"{self.freq_bins[mask][0]:.1f}-{self.freq_bins[mask][-1]:.1f} Hz"
            else:
                bin_range = "NO BINS"
            print(f"Band {i}: {self.band_edges[i]:.1f}-{self.band_edges[i+1]:.1f} Hz "
                  f"({num_bins} bins, actual: {bin_range}, center: {self.band_centers[i]:.1f} Hz)")
        print("...")
        # Show last 3 bands to check high frequency
        for i in range(max(0, self.num_bands-3), self.num_bands):
            mask = self.band_masks[i]
            num_bins = np.sum(mask)
            if num_bins > 0:
                bin_range = f"{self.freq_bins[mask][0]:.1f}-{self.freq_bins[mask][-1]:.1f} Hz"
            else:
                bin_range = "NO BINS"
            print(f"Band {i}: {self.band_edges[i]:.1f}-{self.band_edges[i+1]:.1f} Hz "
                  f"({num_bins} bins, actual: {bin_range}, center: {self.band_centers[i]:.1f} Hz)")
        print()
        
                # Storage for band power history using circular buffers (1000 frames x 32 bands)
        self._band_lock = threading.Lock()
        self.band_power_history = CircularBuffer((self.band_history_len, self.num_bands))

    def audio_callback(self, indata, frames, time_info, status):
        # Device-capture callback (sounddevice). Downmix to mono + resample to
        # the fixed analysis rate, then ingest.
        self._ingest(self._to_target(indata, self._src_rate))

    def _to_target(self, indata, src_rate):
        """Downmix (mean across channels) to mono and resample to self.RATE."""
        arr = np.asarray(indata, dtype=np.float64)
        mono = arr.mean(axis=1) if arr.ndim == 2 else arr
        if src_rate and int(src_rate) != self.RATE and len(mono):
            g = gcd(self.RATE, int(src_rate))
            mono = signal.resample_poly(mono, self.RATE // g, int(src_rate) // g)
        return mono

    def _ingest(self, mono):
        """Lock-free write of the latest mono samples into audio_buffer.
        Shared by the device callback and the internal-tap feed()."""
        n = len(mono)
        if n == 0:
            return
        if n >= self.CHUNK:
            self.audio_buffer = mono[-self.CHUNK:].astype(np.float64, copy=True)
        else:
            self.audio_buffer = np.roll(self.audio_buffer, -n)
            self.audio_buffer[-n:] = mono
        self.new_data_available = True
        self._last_ingest_t = time.time()

    def get_input_health(self):
        """Diagnose the raw input so a dead cable / muted desk / clipping
        preamp reads as a status instead of 'the visuals feel off'.

        Returns {status, rms, peak, clip_pct, dc, stale_s, silent_s}.
        status: no-signal (device stopped delivering), silent (alive but
        flat), clipping, dc-offset, or ok.
        """
        now = time.time()
        buf = self.audio_buffer
        rms = float(np.sqrt(np.mean(buf * buf)))
        peak = float(np.max(np.abs(buf))) if len(buf) else 0.0
        clip_pct = float(np.mean(np.abs(buf) > 0.985)) * 100.0
        dc = float(np.mean(buf))
        stale_s = now - getattr(self, '_last_ingest_t', 0.0)

        if rms < 1e-4:
            if getattr(self, '_silent_since', None) is None:
                self._silent_since = now
        else:
            self._silent_since = None
        silent_s = (now - self._silent_since) if getattr(self, '_silent_since', None) else 0.0

        if stale_s > 2.0:
            status = "no-signal"
        elif silent_s > 5.0:
            status = "silent"
        elif clip_pct > 2.0:
            status = "clipping"
        elif abs(dc) > 0.10:
            status = "dc-offset"
        else:
            status = "ok"
        return {"status": status, "rms": round(rms, 5), "peak": round(peak, 3),
                "clip_pct": round(clip_pct, 2), "dc": round(dc, 3),
                "stale_s": round(stale_s, 1), "silent_s": round(silent_s, 1),
                "gate": round(self._gate, 3),
                "noise_floor": round(self._noise_floor, 5)}

    def feed(self, stereo_buf):
        """Internal-source tap target. Called from the AudioEngine audio thread
        with its mixed output (frames, CHANNELS) @ 44100. No-op unless the
        'internal' source is active, so the engine can keep the tap wired
        permanently and we simply ignore it when another source is selected."""
        if self._active_source != "internal":
            return
        self._ingest(self._to_target(stereo_buf, 44100))

    def analyze_audio(self):
        while self.running:
            frame_start = time.time()
            
            # Always process, even if data hasn't changed much
            # Make a quick copy without holding a lock
            data = self.audio_buffer.copy()

            # --- Noise gate -------------------------------------------------
            # Gate on a ~0.25s-smoothed RMS, NOT the instantaneous frame RMS:
            # noise spikes cross any threshold occasionally, and a fast-attack
            # gate parks itself open on them (observed live: gate stuck at 0.8
            # on a silent loopback with rms 1.7x its floor).
            rms = float(np.sqrt(np.mean(data * data)))
            self._rms_smooth += (rms - self._rms_smooth) * 0.1  # tau ~0.25s @40fps
            stale = time.time() - getattr(self, '_last_ingest_t', 0.0) > 0.5
            # Adaptive floor estimate: tracks DOWN toward the smoothed quiet
            # level (not its minima), creeps UP very slowly (so sustained
            # music can't inflate it), hard-capped at _NF_CEIL. FROZEN while
            # stale - a paused source retains its last music buffer, and
            # learning from it would drag the floor up toward music level.
            if not stale:
                if self._rms_smooth < self._noise_floor:
                    self._noise_floor += (self._rms_smooth - self._noise_floor) * 0.05
                else:
                    # Up-creep ~30s: slow enough that a track can't lift the
                    # floor mid-song, fast enough to re-learn a noisier input.
                    self._noise_floor += (self._rms_smooth - self._noise_floor) * 1e-3
                self._noise_floor = min(max(self._noise_floor, 1e-6), self._NF_CEIL)
            # Soft threshold with wide margins: the floor converges onto the
            # smoothed silence level itself, so "music" must clear well above.
            lo = self._noise_floor * 2.5 + 1.5e-3
            hi = self._noise_floor * 5.0 + 3e-3
            x = min(max((self._rms_smooth - lo) / (hi - lo), 0.0), 1.0)
            target = x * x * (3.0 - 2.0 * x)  # smoothstep
            # A stale buffer (capture stopped delivering - e.g. the bluetooth
            # node suspends when the phone pauses) would otherwise re-analyze
            # the last chunk of music forever. Treat stale as silence.
            if stale:
                target = 0.0
            # Attack ~3 frames, release ~1.2s: quick to open on real music,
            # slow enough that quiet passages don't flicker the visuals off.
            alpha = 0.3 if target > self._gate else 0.02
            self._gate += (target - self._gate) * alpha
            # ----------------------------------------------------------------

            # Apply window and compute FFT
            windowed = data * self.window
            fft = np.fft.rfft(windowed)
            magnitudes = np.abs(fft)

            # Skip DC component (bin 0) to avoid DC offset issues
            magnitudes[0] = 0
            try:
                self.magnitudes = self.magnitudes * self.max_decay + magnitudes * (1 - self.max_decay)
            except AttributeError:
                # First frame - initialize magnitudes
                self.magnitudes = magnitudes.copy()

            # Update spectrum history using circular buffer. Gated: during
            # silence the ratio would read ~1.0 (noise / its own average),
            # which is exactly what steady music reads as.
            normalized_magnitudes = (magnitudes / (self.magnitudes + 10E-10)) * self._gate
            self.spectrum_history.append(normalized_magnitudes)
            
            # Calculate band powers and update history with better noise handling
            band_powers = np.zeros(self.num_bands)
            for i, mask in enumerate(self.band_masks):
                if np.any(mask):
                    # Use RMS instead of sum for better consistency across band widths
                    band_powers[i] = np.sqrt(np.mean(magnitudes[mask] ** 2))
                else:
                    band_powers[i] = 0
            
            # Apply smoothing to reduce noise in individual bands
            # This helps especially with high-frequency bands
            if not hasattr(self, '_prev_band_powers'):
                self._prev_band_powers = band_powers.copy()
            else:
                # Light smoothing (95% new, 5% old)
                band_powers = 0.95 * band_powers + 0.05 * self._prev_band_powers
                self._prev_band_powers = band_powers.copy()
            
                        # Update exponential moving averages if enabled
            if self.use_exponential:
                with self._band_lock:
                    if self.ema_short is None:
                        self.ema_short = band_powers.copy()
                        self.ema_long = band_powers.copy()
                    else:
                        self.ema_short = self.ema_alpha_short * band_powers + (1 - self.ema_alpha_short) * self.ema_short
                        self.ema_long = self.ema_alpha_long * band_powers + (1 - self.ema_alpha_long) * self.ema_long
            
            # Update circular buffers
            self.band_power_history.append(band_powers)
            
            # Maintain precise 40 FPS timing
            elapsed = time.time() - frame_start
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)



    @property
    def sensitivity(self):
        """Audio sensitivity multiplier (0.1–3.0). Scales band power output."""
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        self._sensitivity = max(0.1, min(3.0, float(value)))

    def get_spectrum_history(self):
        """Get spectrum history (most recent first)"""
        history = self.spectrum_history.get_ordered()
        return self.freq_bins.copy(), history

    def get_sound(self):
        """Get the current spectrum analysis"""
        latest = self.spectrum_history.get_latest()
        if latest is not None:
            return latest[2:6].sum()
        return 0

    def get_all_sound(self):
        """Get the current spectrum analysis"""
        latest = self.spectrum_history.get_latest()
        if latest is not None:
            return latest[2:31].mean()
        return 0

    def get_extended_analysis(self):
        """
        Returns a dictionary with comprehensive audio analysis data
        
        Returns:
            dict: {
                'raw_bands': 2D array (1000 x 32) - Raw power in each frequency band
                'norm_short': 2D array (1000 x 32) - Normalized to short-term average
                'norm_long': 2D array (1000 x 32) - Normalized to long-term average
                'norm_long_relu': 2D array (1000 x 32) - ReLU(norm_long - 1), highlights above-average activity
                'band_centers': 1D array (32) - Center frequency of each band (Hz)
                'band_edges': 1D array (33) - Edge frequencies of bands (Hz)
                'timestamp': float - Current time
                'averaging_method': str - 'exponential' or 'mean'
            }
        """
        raw = self.band_power_history.get_ordered()
        
        if len(raw) == 0:
            # Return empty data if buffer is empty
            return {
                'raw_bands': np.zeros((1, self.num_bands)),
                'norm_short': np.zeros((1, self.num_bands)),
                'norm_long': np.zeros((1, self.num_bands)),
                'norm_long_relu': np.zeros((1, self.num_bands)),
                'band_centers': self.band_centers.copy(),
                'band_edges': self.band_edges.copy(),
                'timestamp': time.time(),
                'averaging_method': 'exponential' if self.use_exponential else 'mean'
            }
        
        if self.use_exponential:
            # Use exponential moving averages
            with self._band_lock:
                if self.ema_short is not None and self.ema_long is not None:
                    mean_short = self.ema_short.copy()
                    mean_long = self.ema_long.copy()
                else:
                    # Fallback if EMAs not initialized yet
                    window_short = min(self.avg_window_short, len(raw))
                    window_long = min(self.avg_window_long, len(raw))
                    mean_short = np.mean(raw[:window_short], axis=0)
                    mean_long = np.mean(raw[:window_long], axis=0)
        else:
            # Use simple mean over window
            window_short = min(self.avg_window_short, len(raw))
            window_long = min(self.avg_window_long, len(raw))
            mean_short = np.mean(raw[:window_short], axis=0, keepdims=True)
            mean_long = np.mean(raw[:window_long], axis=0, keepdims=True)
        
                # Ensure no division by zero
        if self.use_exponential:
            mean_short = np.where(mean_short < 1e-10, 1e-10, mean_short)
            mean_long = np.where(mean_long < 1e-10, 1e-10, mean_long)
            norm_short = raw / mean_short[np.newaxis, :]
            norm_long = raw / mean_long[np.newaxis, :]
        else:
            mean_short = np.where(mean_short < 1e-10, 1e-10, mean_short)
            mean_long = np.where(mean_long < 1e-10, 1e-10, mean_long)
            norm_short = raw / mean_short
            norm_long = raw / mean_long
        
        # Calculate ReLU(norm_long - 1) - highlights when bands are above long-term average
        norm_long_relu = np.maximum(0, norm_long - 1)

        # Noise gate on the normalized (ratio) outputs: with a silent input
        # they read ~1.0 (noise / its own average) - steady-music territory.
        # raw_bands stays ungated: it reports honest absolute level.
        g = self._gate
        if g < 0.999:
            norm_short = norm_short * g
            norm_long = norm_long * g
            norm_long_relu = norm_long_relu * g

        # Apply sensitivity multiplier to raw and normalized bands
        s = self._sensitivity
        if s != 1.0:
            raw = raw * s
            norm_short = norm_short * s
            norm_long = norm_long * s
            norm_long_relu = norm_long_relu * s

        return {
            'raw_bands': raw,
            'norm_short': norm_short,
            'norm_long': norm_long,
            'norm_long_relu': norm_long_relu,
            'band_centers': self.band_centers.copy(),
            'band_edges': self.band_edges.copy(),
            'timestamp': time.time(),
            'averaging_method': 'exponential' if self.use_exponential else 'mean',
            'sensitivity': self._sensitivity,
            'gate': self._gate
        }

    def get_chroma(self):
        """12-bin PITCH-CLASS energy fold (C, C#, ... B order relative to A).

        Computes an on-demand FFT of the current audio buffer and folds bins
        between 80 Hz and 5 kHz onto the 12 pitch classes. Returns a
        normalized distribution (sums to ~1) so a chord reads as 3-4 hot
        bins; silence (gated against a slow peak) returns zeros. Consumed by
        harmony-aware shaders via outstate['chroma'].
        """
        if not hasattr(self, '_chroma_map'):
            freqs = np.fft.rfftfreq(self.CHUNK, 1.0 / self.RATE)
            mask = (freqs >= 80.0) & (freqs <= 5000.0)
            pcs = (np.round(12.0 * np.log2(freqs[mask] / 440.0)) % 12).astype(np.int64)
            self._chroma_map = (mask, pcs)
            self._chroma_window = np.hanning(self.CHUNK)
            self._chroma_peak = 1e-9
        mask, pcs = self._chroma_map
        mags = np.abs(np.fft.rfft(self.audio_buffer * self._chroma_window))
        power = mags[mask] ** 2
        chroma = np.bincount(pcs, weights=power, minlength=12)
        total = float(chroma.sum())
        self._chroma_peak = max(self._chroma_peak * 0.999, total, 1e-9)
        if total < self._chroma_peak * 0.02:
            return np.zeros(12, dtype=np.float32)
        return (chroma / total).astype(np.float32)

    def get_waveform(self, n=128, window=2048):
        """Latest TIME-DOMAIN waveform, AGC-normalized to roughly [-1, 1].

        Returns ``n`` samples downsampled from the most recent ``window``
        mono samples (~46ms at 44.1k). Normalized against a slow-decaying
        peak so quiet music still draws a full wave and silence draws a
        flat line. Consumed by MilkDrop-style oscilloscope shaders
        (published as outstate['waveform']).
        """
        buf = self.audio_buffer[-window:]
        step = max(1, window // n)
        usable = step * n
        if buf.shape[0] < usable:
            return np.zeros(n, dtype=np.float32)
        w = buf[-usable:].reshape(n, step).mean(axis=1)
        peak = float(np.max(np.abs(w))) if w.size else 0.0
        self._wave_peak = max(getattr(self, '_wave_peak', 1e-6) * 0.995,
                              peak, 1e-6)
        # Silence gate: don't amplify noise up to full scale. The absolute
        # 1e-4 check alone misses analog noise floors (~1e-2), which this
        # AGC would happily draw as a full-scale wave - scale by the noise
        # gate so silence actually draws a flat line.
        if self._wave_peak < 1e-4:
            return np.zeros(n, dtype=np.float32)
        return (np.clip(w / self._wave_peak, -1.0, 1.0) * self._gate).astype(np.float32)

    def get_current_bands(self, normalize='none'):
        """
        Get just the current frame's band values
        
        Args:
            normalize: 'none', 'short', or 'long' for different normalizations
            
        Returns:
            1D array of 32 band values
        """
        data = self.get_extended_analysis()
        
        if normalize == 'short':
            return data['norm_short'][0]
        elif normalize == 'long':
            return data['norm_long'][0]
        else:
            return data['raw_bands'][0]

    def start(self):
        self.running = True
        self.analysis_thread = threading.Thread(target=self.analyze_audio, daemon=True)
        self.analysis_thread.start()
        self._open_source(self.source)

    def set_source(self, source):
        """Switch the active input source at runtime (microphone / linein /
        loopback / internal / bluetooth). Safe to call live: the band masks are
        rate-fixed, so only the input plumbing changes."""
        if source == self._active_source:
            return
        self._open_source(source)

    def set_bluetooth_hint(self, fragment):
        """Update the device-name fragment used to resolve the 'bluetooth'
        capture source (the underscored MAC of the connected phone, or None
        when nothing is connected). If 'bluetooth' is the active source, reopen
        it so the new node is picked up immediately."""
        if self._device_names.get("bluetooth") == fragment:
            return
        self._device_names["bluetooth"] = fragment
        if self._active_source == "bluetooth":
            self._open_source("bluetooth")

    def _open_source(self, source):
        """(Re)open the input plumbing for the requested source."""
        self._close_stream()
        self.source = source
        self._active_source = source
        # Start the new source from silence so a stale window doesn't linger.
        self.audio_buffer = np.zeros(self.CHUNK)
        # Each source has its own noise floor (floating line-in >> digital BT
        # silence). Restart the estimate LOW (sensitive) and let it creep up
        # only if the new source proves noisy - starting from the ceiling
        # left real audio unable to open the gate until a silent gap dragged
        # the floor back down (deadlock; user-reported).
        self._noise_floor = self._NF_INIT
        self._rms_smooth = 0.0
        if source == "internal":
            # No device stream; AudioEngine.feed() drives _ingest.
            print("[Audio] source=internal (AudioEngine mix tap)")
            return

        # Loopback + bluetooth need nodes that only exist inside the
        # PipeWire/Pulse graph (a sink's monitor, a bluez A2DP source).
        # PortAudio (sounddevice) never enumerates those - it only sees
        # ALSA-level devices like "pulse" and "default" - so the generic path
        # below silently falls back to the default input (the mic/line-in)
        # and the visuals react to room bleed instead of the actual audio.
        # Capture the node directly: pw-record (native PipeWire - REQUIRED on
        # PipeWire hosts, whose pulse shim can serve hw-sink monitors broken/
        # silent, breaking parec AND the soundcard library), else parec
        # (genuine PulseAudio hosts).
        if source in ("loopback", "bluetooth"):
            if self._start_pulse_capture(source):
                return
            if sys.platform.startswith("linux"):
                # Expected on Linux - say so. On Windows this is the normal
                # route to the WASAPI (soundcard) path below; stay quiet.
                print(f"[Audio] no pw-record/parec; trying legacy paths for '{source}'")

        # Legacy loopback: the `soundcard` library (WASAPI loopback - the
        # Windows path). Falls back further to a Stereo Mix / monitor INPUT
        # device via sounddevice if unavailable.
        if source == "loopback" and _soundcard is not None:
            if self._start_soundcard_loopback(self._device_names.get("loopback")):
                return
            print("[Audio] soundcard loopback unavailable; trying a Stereo Mix / "
                  "monitor input device")

        try:
            dev_id, native_rate, channels = self._resolve_device(source)
        except Exception as e:
            print(f"[Audio] could not resolve '{source}' source: {e}; using default input")
            dev_id, native_rate, channels = (None, self.RATE, 1)
        self._src_rate = int(native_rate or self.RATE)
        self._capture_channels = max(1, min(2, int(channels or 1)))
        try:
            self.stream = sd.InputStream(
                samplerate=self._src_rate,
                blocksize=self.CALLBACK_BLOCKSIZE,
                channels=self._capture_channels,
                device=dev_id,
                callback=self.audio_callback,
            )
            self.stream.start()
            label = sd.query_devices(dev_id)["name"] if dev_id is not None else "default"
            print(f"[Audio] source={source} via '{label}' @ {self._src_rate} Hz")
        except Exception as e:
            print(f"[Audio] failed to open '{source}' source: {e}; analyzer will be silent")
            self.stream = None

    def _start_soundcard_loopback(self, name_fragment=None):
        """Start a background thread capturing system output loopback via the
        `soundcard` library, feeding _ingest at the target rate. Returns True
        on success."""
        try:
            mic = None
            if name_fragment:
                frag = name_fragment.lower()
                for m in _soundcard.all_microphones(include_loopback=True):
                    if getattr(m, "isloopback", False) and frag in m.name.lower():
                        mic = m
                        break
            if mic is None:
                mic = _soundcard.get_microphone(
                    str(_soundcard.default_speaker().name), include_loopback=True)
        except Exception as e:
            print(f"[Audio] soundcard loopback resolve failed: {e}")
            return False

        self._src_rate = self.RATE                      # soundcard records at our rate
        self._sc_stop.clear()

        def _loop():
            # soundcard warns on every missed capture packet ("data
            # discontinuity in recording"). Under a busy render loop the
            # Python capture thread occasionally misses one; a dropped ~10ms
            # block is invisible to the 40fps FFT analysis, so the warning
            # is pure log spam - silence it.
            import warnings
            warnings.filterwarnings(
                "ignore", message="data discontinuity in recording")
            try:
                # Read 4 callback blocks (~46ms) per wakeup: fewer thread
                # wakeups = far fewer missed packets in the first place.
                chunk = self.CALLBACK_BLOCKSIZE * 4
                with mic.recorder(samplerate=self.RATE, channels=2,
                                  blocksize=self.CALLBACK_BLOCKSIZE) as rec:
                    print(f"[Audio] source=loopback via soundcard '{mic.name}' @ {self.RATE} Hz")
                    while not self._sc_stop.is_set():
                        data = rec.record(numframes=chunk)
                        # Already at target rate; _to_target just downmixes.
                        self._ingest(self._to_target(data, self.RATE))
            except Exception as e:
                print(f"[Audio] soundcard loopback capture stopped: {e}")

        self._sc_thread = threading.Thread(target=_loop, daemon=True)
        self._sc_thread.start()
        return True

    @staticmethod
    def _resolve_pulse_node(fragment=None):
        """Name of the PipeWire/Pulse *source* node for the connected Bluetooth
        device, or None. Prefers a bluez node containing `fragment` (the
        underscored MAC), else any bluez node.

        Scans BOTH layers: `pactl list sources` (genuine PulseAudio hosts) and
        the native graph via `pw-cli ls Node` - on PipeWire the pulse shim can
        omit bluez nodes entirely (observed live: bluez_input.<MAC>.2 existed
        natively and captured fine while pactl listed no bluez source at all)."""
        names = []
        try:
            out = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True, text=True, timeout=5).stdout
            names += [line.split("\t")[1] for line in out.splitlines()
                      if "\t" in line]
        except Exception:
            pass
        if shutil.which("pw-cli"):
            try:
                out = subprocess.run(
                    ["pw-cli", "ls", "Node"],
                    capture_output=True, text=True, timeout=5).stdout
                names += [m.group(1) for m in
                          re.finditer(r'node\.name = "([^"]+)"', out)]
            except Exception:
                pass
        # Capture-side bluez nodes only: bluez_input/bluez_source (a phone
        # streaming in), NOT bluez_output (us playing to headphones) or
        # bluez_midi.
        bluez = [n for n in names
                 if n.lower().startswith(("bluez_input", "bluez_source"))]
        if fragment:
            frag = fragment.lower()
            for n in bluez:
                if frag in n.lower():
                    return n
        return bluez[0] if bluez else None

    def _pw_capture_cmd(self, source):
        """Resolve (argv, label) for a native graph capture of `source`, or
        None if the node isn't there (yet). Prefers pw-record: on PipeWire
        hosts the pulse compatibility shim can serve hw-sink monitor sources
        broken/silent (observed live: parec returned ZERO bytes from a real
        monitor while pw-record captured it fine), which also breaks the
        soundcard library. parec remains the genuine-PulseAudio fallback."""
        pw = shutil.which("pw-record") is not None
        if source == "bluetooth":
            node = self._resolve_pulse_node(self._device_names.get("bluetooth"))
            if node is None:
                return None
            if pw:
                return (["pw-record", "--target", node,
                         "--format", "f32", "--rate", str(self.RATE),
                         "--channels", "2", "--raw", "-"], node)
            return (["parec", f"--device={node}", "--raw",
                     "--format=float32le", f"--rate={self.RATE}",
                     "--channels=2", "--latency-msec=50"], node)
        if source == "loopback":
            sink = None
            frag = (self._device_names.get("loopback") or "").lower()
            try:
                if frag:
                    out = subprocess.run(
                        ["pactl", "list", "sinks", "short"],
                        capture_output=True, text=True, timeout=5).stdout
                    sinks = [line.split("\t")[1] for line in out.splitlines()
                             if "\t" in line]
                    sink = next((s for s in sinks if frag in s.lower()), None)
                if sink is None:
                    sink = subprocess.run(
                        ["pactl", "get-default-sink"],
                        capture_output=True, text=True, timeout=5).stdout.strip()
            except Exception:
                return None
            if not sink:
                return None
            if pw:
                # Target the SINK node in capture-sink mode. The pulse-layer
                # "<sink>.monitor" alias does not exist in the native graph -
                # pw-record with that name silently falls back to the default
                # source (the mic), which is exactly the bug this replaces.
                return (["pw-record", "--target", sink,
                         "-P", "{ stream.capture.sink = true }",
                         "--format", "f32", "--rate", str(self.RATE),
                         "--channels", "2", "--raw", "-"], f"{sink} (output mix)")
            return (["parec", f"--device={sink}.monitor", "--raw",
                     "--format=float32le", f"--rate={self.RATE}",
                     "--channels=2", "--latency-msec=50"], f"{sink}.monitor")
        return None

    def _start_pulse_capture(self, source):
        """Start a background thread capturing `source` ("loopback" or
        "bluetooth") from the PipeWire/Pulse graph via pw-record/parec,
        feeding _ingest at the target rate. Returns True on success (thread
        started; node resolution retries briefly inside the thread since a
        bluez node can appear a moment after BlueZ reports the device
        connected)."""
        if shutil.which("pactl") is None:
            return False
        if shutil.which("pw-record") is None and shutil.which("parec") is None:
            return False

        self._src_rate = self.RATE  # capture command resamples to our rate
        stop = threading.Event()    # per-thread: a fast source-switch must not
        self._pa_stop = stop        # cross-talk with a lingering old thread

        def _loop():
            # PERSISTENT capture loop. The target node is TRANSIENT: a bluez
            # A2DP source exists only while the phone is actually streaming -
            # it appears seconds after play and vanishes seconds after pause.
            # So: poll for the node for as long as this source stays active,
            # capture while it lives, and go back to waiting when it drops.
            # (The analyzer's stale-buffer gate reads the in-between as
            # silence, so nothing downstream free-runs on the last chunk.)
            waiting_said = False
            while not stop.is_set():
                resolved = self._pw_capture_cmd(source)
                if resolved is None:
                    if not waiting_said:
                        print(f"[Audio] waiting for a '{source}' capture node "
                              "(bluetooth: appears when the phone starts playing)")
                        waiting_said = True
                    stop.wait(1.0)
                    continue
                argv, label = resolved
                try:
                    proc = subprocess.Popen(
                        argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"[Audio] {argv[0]} failed to start: {e}")
                    stop.wait(2.0)
                    continue
                self._pa_proc = proc
                if stop.is_set():  # closed while we were spawning
                    proc.terminate()
                    return
                print(f"[Audio] source={source} via {argv[0]} '{label}' @ {self.RATE} Hz")
                waiting_said = False
                # Read ~4 callback blocks (~46ms) per wakeup, matching the
                # soundcard loopback path. Non-blocking reads + a 1s node
                # watchdog: pw-record does NOT exit when its target node
                # vanishes - it silently falls back to the DEFAULT SOURCE
                # (the mic!), which is exactly the bug this path exists to
                # fix. Re-verify the resolved target every second and
                # restart the capture the moment it changes or disappears.
                nbytes = self.CALLBACK_BLOCKSIZE * 4 * 2 * 4  # frames * ch * f32
                fd = proc.stdout.fileno()
                pending = b""
                last_verify = time.time()
                try:
                    while not stop.is_set():
                        r, _, _ = select.select([fd], [], [], 0.25)
                        if r:
                            chunk = os.read(fd, nbytes)
                            if not chunk:  # EOF: capture process died
                                break
                            pending += chunk
                            usable = len(pending) // 8 * 8  # whole stereo f32 frames
                            if usable:
                                data = np.frombuffer(pending[:usable],
                                                     dtype="<f4").reshape(-1, 2)
                                pending = pending[usable:]
                                # Already at target rate; just downmixes.
                                self._ingest(self._to_target(data, self.RATE))
                        if time.time() - last_verify >= 1.0:
                            last_verify = time.time()
                            if self._pw_capture_cmd(source) != resolved:
                                print(f"[Audio] '{source}' capture node changed/"
                                      "vanished; reopening")
                                break
                except Exception as e:
                    print(f"[Audio] {source} capture stopped: {e}")
                finally:
                    proc.terminate()
                    self._pa_proc = None
                if not stop.is_set():
                    stop.wait(0.5)  # brief breather, then re-resolve

        self._pa_thread = threading.Thread(target=_loop, daemon=True)
        self._pa_thread.start()
        return True

    def _close_stream(self):
        # Stop a soundcard loopback thread, if running.
        if self._sc_thread is not None:
            self._sc_stop.set()
            self._sc_thread.join(timeout=1.0)
            self._sc_thread = None
        # Stop a parec bluetooth capture thread, if running.
        if self._pa_thread is not None:
            self._pa_stop.set()
            if self._pa_proc is not None:
                try:
                    self._pa_proc.terminate()
                except Exception:
                    pass
                self._pa_proc = None
            self._pa_thread.join(timeout=1.0)
            self._pa_thread = None
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _resolve_device(self, source):
        """Return (device_id, native_rate, channels) for the requested source.
        Cross-platform (Windows / Linux). Used for device INPUT capture; the
        loopback soundcard path is handled separately in _open_source."""
        if self.device is not None:
            d = sd.query_devices(self.device)
            return self.device, d["default_samplerate"], d["max_input_channels"]

        if source == "linein":
            frag = self._device_names.get("linein")
            dev_id = (self._find_input_by_name(frag) if frag else
                      self._find_input_by_keywords(["line in", "line-in", "aux", "line"]))
        elif source == "loopback":
            # Fallback path (no soundcard): a Stereo Mix / monitor INPUT device.
            frag = self._device_names.get("loopback")
            dev_id = (self._find_input_by_name(frag) if frag else
                      self._find_input_by_keywords(
                          ["monitor", "loopback", "stereo mix", "what u hear"]))
        elif source == "bluetooth":
            # A connected A2DP phone appears as a PipeWire/Pulse capture node
            # named bluez_input.<MAC> / bluez_source.<MAC>. Prefer the specific
            # connected MAC (set live via set_bluetooth_hint), else any "bluez".
            frag = self._device_names.get("bluetooth")
            dev_id = (self._find_input_by_name(frag) if frag else None)
            if dev_id is None:
                dev_id = self._find_input_by_keywords(["bluez", "bluetooth"])
        else:  # microphone (and any unknown source)
            frag = self._device_names.get("microphone")
            dev_id = self._find_input_by_name(frag) if frag else sd.default.device[0]

        if dev_id is None:
            dev_id = sd.default.device[0]
        d = sd.query_devices(dev_id)
        return dev_id, d["default_samplerate"], d["max_input_channels"]

    @staticmethod
    def _find_input_by_name(fragment):
        if not fragment:
            return None
        frag = fragment.lower()
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and frag in d["name"].lower():
                return i
        return None

    @staticmethod
    def _find_input_by_keywords(keywords):
        devs = sd.query_devices()
        for kw in keywords:
            for i, d in enumerate(devs):
                if d["max_input_channels"] > 0 and kw in d["name"].lower():
                    return i
        return None

    def stop(self):
        self.running = False
        self._close_stream()
        if self.analysis_thread is not None:
            self.analysis_thread.join(timeout=1.0)

class SpectrogramPlotter:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
        # Setup the figure with 4 subplots
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 12))
        
        # Create grid: 3 rows for spectrograms, 1 for info
        gs = self.fig.add_gridspec(4, 1, height_ratios=[3, 3, 3, 1], hspace=0.3)
        
        self.ax_raw = self.fig.add_subplot(gs[0])
        self.ax_norm_short = self.fig.add_subplot(gs[1])
        self.ax_norm_long = self.fig.add_subplot(gs[2])
        self.ax_info = self.fig.add_subplot(gs[3])
        
        # Get initial data
        analysis = analyzer.get_extended_analysis()
        
        # Time window for display (5 seconds = 200 frames at 40fps)
        display_frames = 200
        time_extent = display_frames / analyzer.FPS
        
        # Initialize raw bands plot
        self.img_raw = self.ax_raw.imshow(
            analysis['raw_bands'][:display_frames].T,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            extent=[0, time_extent, 0, self.analyzer.num_bands],
            cmap='magma',
            vmin=0, vmax=np.percentile(analysis['raw_bands'], 95)
        )
        self.ax_raw.set_ylabel('Band Index')
        self.ax_raw.set_title('Raw Band Power')
        plt.colorbar(self.img_raw, ax=self.ax_raw, label='Power')
        
        # Initialize short-term normalized plot
        avg_method = "EMA" if analyzer.use_exponential else "Mean"
        if analyzer.use_exponential:
            short_time = 1 / (analyzer.ema_alpha_short * analyzer.FPS)
            short_label = f"τ≈{short_time:.2f}s"
        else:
            short_time = analyzer.avg_window_short / analyzer.FPS
            short_label = f"{short_time:.1f}s"
        
        self.img_norm_short = self.ax_norm_short.imshow(
            analysis['norm_short'][:display_frames].T,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            extent=[0, time_extent, 0, self.analyzer.num_bands],
            cmap='viridis',
            vmin=0, vmax=2
        )
        self.ax_norm_short.set_ylabel('Band Index')
        self.ax_norm_short.set_title(f'Short-term Normalized ({avg_method}, {short_label})')
        plt.colorbar(self.img_norm_short, ax=self.ax_norm_short, label='Relative Power')
        
        # Initialize long-term normalized plot
        if analyzer.use_exponential:
            long_time = 1 / (analyzer.ema_alpha_long * analyzer.FPS)
            long_label = f"τ≈{long_time:.2f}s"
        else:
            long_time = analyzer.avg_window_long / analyzer.FPS
            long_label = f"{long_time:.1f}s"
        
        self.img_norm_long = self.ax_norm_long.imshow(
            analysis['norm_long'][:display_frames].T,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            extent=[0, time_extent, 0, self.analyzer.num_bands],
            cmap='plasma',
            vmin=0, vmax=2
        )
        self.ax_norm_long.set_xlabel('Time (seconds ago)')
        self.ax_norm_long.set_ylabel('Band Index')
        self.ax_norm_long.set_title(f'Long-term Normalized ({avg_method}, {long_label})')
        plt.colorbar(self.img_norm_long, ax=self.ax_norm_long, label='Relative Power')
        
        # Setup info display
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(
            0.5, 0.5, '', 
            transform=self.ax_info.transAxes,
            fontsize=14,
            ha='center',
            va='center',
            family='monospace'
        )
        
        # Add frequency labels on right side
        freq_ticks = [0, 7, 15, 23, 31]
        freq_labels = [f"{self.analyzer.band_centers[i]:.0f} Hz" for i in freq_ticks]
        
        for ax in [self.ax_raw, self.ax_norm_short, self.ax_norm_long]:
            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(freq_labels)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()

    def update(self, frame):
        analysis = self.analyzer.get_extended_analysis()
        
        # Display last 200 frames (5 seconds at 40fps)
        display_frames = 200
        
        # Update images
        self.img_raw.set_array(analysis['raw_bands'][:display_frames].T)
        self.img_norm_short.set_array(analysis['norm_short'][:display_frames].T)
        self.img_norm_long.set_array(analysis['norm_long'][:display_frames].T)
        
        # Update color limits for raw data adaptively
        raw_95 = np.percentile(analysis['raw_bands'][:display_frames], 95)
        if raw_95 > 0:
            self.img_raw.set_clim(0, raw_95)
        
                # Update info text
        current_bands = analysis['raw_bands'][0]
        total_power = np.sum(current_bands)
        max_band = np.argmax(current_bands)
        max_freq = self.analyzer.band_centers[max_band]
        
        info_str = (
            f"Total Power: {total_power:.2e}  |  "
            f"Peak Band: {max_band} ({max_freq:.0f} Hz)"
        )
        self.info_text.set_text(info_str)
        
        return self.img_raw, self.img_norm_short, self.img_norm_long, self.info_text

    def start(self):
        self.ani = FuncAnimation(
            self.fig, 
            self.update,
            interval=25,  # 40 FPS
            blit=True,
            cache_frame_data=False
        )
        plt.show()

if __name__ == "__main__":
    print("Audio Analysis System Starting...")
    print("Running at 40 FPS")
    print("Use Ctrl+C to stop\n")
    
    # Choose your audio source:
    
    # Option 1: Use exponential averaging with fast/slow response
    analyzer = MicrophoneAnalyzer(
        device_name="HD Pro Webcam C920",
        use_exponential=True,
        ema_alpha_short=0.1,  # Fast response (10 frames ~= 0.25s)
        ema_alpha_long=0.02   # Slow response (50 frames ~= 1.25s)
    )
    
    # Option 2: Use mean averaging with custom windows
    # analyzer = MicrophoneAnalyzer(
    #     device_name="HD Pro Webcam C920",
    #     use_exponential=False,
    #     avg_window_short=40,   # 1 second
    #     avg_window_long=200    # 5 seconds
    # )
    
    # Option 3: Use loopback (system audio output)
    # analyzer = MicrophoneAnalyzer(use_loopback=True)
    
    # Option 4: Use default input device with default settings
    # analyzer = MicrophoneAnalyzer()
    
    analyzer.start()
    
    try:
        # Wait a moment to collect some data
        time.sleep(1)
        
        # List all devices to help with debugging
        list_audio_devices_detailed()
        
        # Start visualization
        plotter = SpectrogramPlotter(analyzer)
        plotter.start()
    except KeyboardInterrupt:
        print("\nStopping analysis...")
        analyzer.stop()
        plt.close('all')