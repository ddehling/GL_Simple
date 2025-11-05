import sounddevice as sd
import numpy as np
import threading
import time
from queue import Queue
from scipy import signal
from scipy.signal import find_peaks
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
                 avg_window_short=20, avg_window_long=100,
                 use_exponential=False, ema_alpha_short=0.05, ema_alpha_long=0.01):
        """
        Initialize the microphone analyzer
        
        Args:
            device: Device ID to use (None for default)
            device_name: Name fragment to search for device
            use_loopback: Try to find loopback/stereo mix device
            avg_window_short: Number of frames for short-term average (default 20 = 0.5s at 40fps)
            avg_window_long: Number of frames for long-term average (default 100 = 2.5s at 40fps)
            use_exponential: Use exponential moving average instead of simple mean
            ema_alpha_short: Alpha for short-term EMA (higher = faster response, 0-1)
            ema_alpha_long: Alpha for long-term EMA (lower = slower response, 0-1)
        """
        # Print device information
        devices = list_audio_devices()
        
        # Handle device selection
        if use_loopback:
            print("\nSearching for loopback device...")
            device_id, device_info = find_loopback_device()
            if device_id is not None:
                device = device_id
                print(f"Found loopback device: {device_info['name']}")
            else:
                print("No loopback device found.")
                print("\nOn Windows, you may need to enable 'Stereo Mix':")
                print("1. Right-click the sound icon in taskbar")
                print("2. Select 'Sounds' > 'Recording' tab")
                print("3. Right-click in empty space and check 'Show Disabled Devices'")
                print("4. Right-click 'Stereo Mix' and select 'Enable'")
                print("\nAlternatively, install VB-Audio Virtual Cable or similar software.")
                print("\nFalling back to default input device...")
                device = None
        elif device_name is not None:
            device_id, device_info = find_device_by_name(device_name)
            if device_id is not None:
                device = device_id
                print(f"\nFound device matching '{device_name}'")
            else:
                print(f"\nNo device found matching '{device_name}'. Using default.")
                device = None
        
        if device is None:
            device = sd.default.device[0]
            
        print(f"\nUsing Device ID {device}: {devices[device]['name']}")

        
        # Try to use the default sample rate, but fall back to a standard rate if needed
        try:
            self.RATE = int(devices[device]['default_samplerate'])
            # Test if the rate is supported
            sd.check_input_settings(device=device, samplerate=self.RATE)
            print(f"Sample Rate: {self.RATE}")
        except sd.PortAudioError:
            # Try common sample rates
            for test_rate in [44100, 48000, 22050, 16000, 8000]:
                try:
                    sd.check_input_settings(device=device, samplerate=test_rate)
                    self.RATE = test_rate
                    print(f"Default sample rate not supported. Using alternate rate: {self.RATE}")
                    break
                except sd.PortAudioError:
                    continue
            else:
                raise Exception("Could not find a supported sample rate for this device")
        
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
        
        # BPM detection parameters (enhanced)
        self.bpm_range = (60, 300)
        self.onset_history = CircularBuffer(self.band_history_len)
        self.spectral_flux_history = CircularBuffer(self.band_history_len)
        self.bass_onset_history = CircularBuffer(self.band_history_len)
        self.mid_energy_history = CircularBuffer(self.band_history_len)
        self.last_bpm = 120.0
        self.bpm_confidence = 0.0
        self.bpm_update_counter = 0
        self.prev_magnitudes = None
        
        # Tempo tracking (NEW)
        self.tempo_belief = 120.0
        self.tempo_variance = 100.0
        self.tempo_history = CircularBuffer(100)
        self.beat_phase = 0.0
        self.last_beat_time = time.time()
        
        # Tempo locking (NEW)
        self.tempo_locked = False
        self.tempo_lock_strength = 0.0  # 0-1, how confident we are in lock
        self.frames_at_current_tempo = 0  # How long we've been stable

    def audio_callback(self, indata, frames, time_info, status):
        # Lock-free write - just shift and append
        # This is fast enough that we don't need a lock
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata[:, 0]
        self.new_data_available = True

    def analyze_audio(self):
        while self.running:
            frame_start = time.time()
            
            # Always process, even if data hasn't changed much
            # Make a quick copy without holding a lock
            data = self.audio_buffer.copy()
            
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

            # Update spectrum history using circular buffer
            normalized_magnitudes = magnitudes / (self.magnitudes + 10E-10)
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
            
            # Calculate spectral flux
            if self.prev_magnitudes is not None:
                diff = magnitudes - self.prev_magnitudes
                spectral_flux = np.sum(np.maximum(diff, 0))
            else:
                spectral_flux = 0
            self.prev_magnitudes = magnitudes.copy()
            
            # Update circular buffers
            self.band_power_history.append(band_powers)
            self.spectral_flux_history.append(spectral_flux)
            
            # NEW: Calculate different onset types
            # Bass onset (sub-bass to low-mid, bands 0-8, roughly 40-250 Hz)
            bass_onset = np.sum(band_powers[0:8])
            self.bass_onset_history.append(bass_onset)
            
            # Mid-range energy (bands 8-16, roughly 250-1000 Hz)
            mid_energy = np.sum(band_powers[8:16])
            self.mid_energy_history.append(mid_energy)
            
            # General onset strength (broader range)
            onset_strength = np.sum(band_powers[2:12])
            self.onset_history.append(onset_strength)
            
            # Update BPM estimation periodically
            self.bpm_update_counter += 1
            if self.bpm_update_counter >= 40:
                self._update_bpm_estimate()
                self.bpm_update_counter = 0
            
            # Update beat phase tracking
            self._update_beat_phase(frame_start)
            
            # Maintain precise 40 FPS timing
            elapsed = time.time() - frame_start
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)



    def _update_beat_phase(self, current_time):
        """Track the current phase within a beat cycle"""
        if self.tempo_belief > 0:
            beat_duration = 60.0 / self.tempo_belief
            time_since_beat = current_time - self.last_beat_time
            self.beat_phase = (time_since_beat % beat_duration) / beat_duration
            
            # Reset if we've passed a beat
            if time_since_beat >= beat_duration:
                self.last_beat_time = current_time

    def _compute_onset_strength_function(self, onset_data):
        """Compute normalized onset strength with envelope following"""
        if len(onset_data) < 10:
            return None
        
        # Remove DC component
        onset_data = onset_data - np.mean(onset_data)
        
        if np.std(onset_data) < 1e-6:
            return None
        
        # Normalize
        onset_data = onset_data / np.std(onset_data)
        
        # Apply envelope following to smooth onset function
        onset_envelope = np.copy(onset_data)
        for i in range(1, len(onset_envelope)):
            if onset_envelope[i] < onset_envelope[i-1]:
                onset_envelope[i] = onset_envelope[i-1] * 0.95 + onset_envelope[i] * 0.05
        
        return onset_envelope

    def _autocorrelation_analysis(self, signal_data):
        """
        Perform autocorrelation analysis and return candidate tempos with scores
        
        Returns:
            List of (tempo, score, lag) tuples sorted by score
        """
        if signal_data is None or len(signal_data) < 100:
            return []
        
        # Compute autocorrelation
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            return []
        
        # Convert frame indices to BPM range
        min_lag = max(2, int(self.FPS * 60 / self.bpm_range[1]))
        max_lag = min(len(autocorr) - 1, int(self.FPS * 60 / self.bpm_range[0]))
        
        if min_lag >= max_lag:
            return []
        
        valid_autocorr = autocorr[min_lag:max_lag]
        
        # Find peaks with minimum prominence
        peaks, properties = find_peaks(valid_autocorr, prominence=0.08, distance=3)
        
        if len(peaks) == 0:
            # Fallback to maximum
            peak_idx = np.argmax(valid_autocorr)
            lag = peak_idx + min_lag
            bpm = 60 * self.FPS / lag
            score = valid_autocorr[peak_idx]
            return [(bpm, score, lag)]
        
        # Convert all peaks to BPM and store with lags
        peak_bpms = []
        for i, peak in enumerate(peaks):
            lag = peak + min_lag
            bpm = 60 * self.FPS / lag
            peak_bpms.append((bpm, valid_autocorr[peak], lag, properties['prominences'][i]))
        
        # OCTAVE ERROR DETECTION AND CORRECTION
        # Check for octave relationships among peaks
        corrected_candidates = []
        used_peaks = set()
        
        for i, (bpm, score, lag, prominence) in enumerate(peak_bpms):
            if i in used_peaks:
                continue
            
            # Look for octave-related peaks (2x, 0.5x)
            octave_group = [(bpm, score, lag, prominence, i)]
            
            for j, (other_bpm, other_score, other_lag, other_prom) in enumerate(peak_bpms):
                if i == j or j in used_peaks:
                    continue
                
                ratio = bpm / other_bpm
                
                # Check for 2:1 or 1:2 relationship
                if 1.9 < ratio < 2.1:  # This peak is ~2x the other
                    octave_group.append((other_bpm, other_score, other_lag, other_prom, j))
                    used_peaks.add(j)
                elif 0.48 < ratio < 0.52:  # This peak is ~0.5x the other
                    octave_group.append((other_bpm, other_score, other_lag, other_prom, j))
                    used_peaks.add(j)
            
            # If we found octave-related peaks, choose the best one
            if len(octave_group) > 1:
                # Score each candidate in the octave group
                best_candidate = None
                best_total_score = -1
                
                for candidate_bpm, candidate_score, candidate_lag, candidate_prom, idx in octave_group:
                    # Preference for "sweet spot" BPM range (80-180 BPM for most music)
                    if 80 <= candidate_bpm <= 180:
                        range_bonus = 0.3
                    elif 60 <= candidate_bpm < 80 or 180 < candidate_bpm <= 200:
                        range_bonus = 0.1
                    else:
                        range_bonus = -0.1
                    
                    # Prefer the slower tempo if both are strong (usually more accurate)
                    # But not if it's too slow
                    if candidate_bpm < 70:
                        tempo_preference = -0.2
                    elif candidate_bpm < 100:
                        tempo_preference = 0.1
                    else:
                        tempo_preference = 0.0
                    
                    total = candidate_score + range_bonus + tempo_preference
                    
                    if total > best_total_score:
                        best_total_score = total
                        best_candidate = (candidate_bpm, candidate_score, candidate_lag, candidate_prom)
                
                # Use the best from the octave group
                bpm, score, lag, prominence = best_candidate
                used_peaks.add(i)
            
            # Score this peak
            base_score = score
            
            # Bonus for peak sharpness
            sharpness_bonus = prominence * 0.5
            
            # STRONGER bonus for sweet spot BPM range
            if 90 <= bpm <= 180:
                range_bonus = 0.4
            elif 80 <= bpm < 90 or 180 < bpm <= 200:
                range_bonus = 0.2
            elif 70 <= bpm < 80 or 200 < bpm <= 220:
                range_bonus = 0.0
            else:
                range_bonus = -0.3  # Penalize unusual tempos
            
            # Bonus for being near our current tempo belief (if not too far off)
            tempo_diff = abs(bpm - self.tempo_belief)
            if self.tempo_locked:
                # When locked, check if this might be an octave error of our belief
                belief_ratio = bpm / self.tempo_belief
                if 1.9 < belief_ratio < 2.1 or 0.48 < belief_ratio < 0.52:
                    # This is an octave of our locked tempo - penalize
                    tracking_bonus = -0.5 * self.tempo_lock_strength
                elif tempo_diff < 2:
                    tracking_bonus = 0.8 * self.tempo_lock_strength
                elif tempo_diff < 5:
                    tracking_bonus = 0.4 * self.tempo_lock_strength
                elif tempo_diff < 10:
                    tracking_bonus = 0.1 * self.tempo_lock_strength
                else:
                    tracking_bonus = -0.2 * self.tempo_lock_strength
            else:
                if tempo_diff < 5:
                    tracking_bonus = 0.3
                elif tempo_diff < 10:
                    tracking_bonus = 0.15
                else:
                    tracking_bonus = 0
            
            # Bonus for common tempos
            common_bpms = [120, 128, 140, 150, 160, 174]
            common_bonus = 0
            for common_bpm in common_bpms:
                if abs(bpm - common_bpm) < 2:
                    common_bonus = 0.15
                    break
            
            total_score = base_score + sharpness_bonus + range_bonus + tracking_bonus + common_bonus
            corrected_candidates.append((bpm, total_score, lag))
        
        # Sort by score
        corrected_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # FINAL OCTAVE CHECK on top candidate
        if len(corrected_candidates) > 1:
            best_bpm = corrected_candidates[0][0]
            
            # Check if second-best is an octave relationship
            for other_bpm, other_score, other_lag in corrected_candidates[1:4]:
                ratio = best_bpm / other_bpm
                
                # If we chose the faster tempo but slower is close in score, reconsider
                if 1.9 < ratio < 2.1:  # Best is 2x another strong candidate
                    score_diff = corrected_candidates[0][1] - other_score
                    if score_diff < 0.3:  # Scores are close
                        # Prefer the slower tempo
                        print(f"Octave correction: Choosing {other_bpm:.1f} over {best_bpm:.1f}")
                        corrected_candidates[0], corrected_candidates[1] = corrected_candidates[1], corrected_candidates[0]
                        break
        
        return corrected_candidates


    def _update_bpm_estimate(self):
        """Enhanced BPM estimation using multiple methods"""
        # Use longer analysis window for more stability
        analysis_length = min(800, len(self.spectral_flux_history))
        
        if analysis_length < 100:
            return
        
        # Get data from different onset detection methods
        bass_data = self.bass_onset_history.get_ordered(analysis_length)
        mid_data = self.mid_energy_history.get_ordered(analysis_length)
        flux_data = self.spectral_flux_history.get_ordered(analysis_length)
        
        # Compute onset strength functions
        bass_osf = self._compute_onset_strength_function(bass_data)
        mid_osf = self._compute_onset_strength_function(mid_data)
        flux_osf = self._compute_onset_strength_function(flux_data)
        
        # Analyze each method
        all_candidates = []
        method_best_scores = {}
        
        if bass_osf is not None:
            bass_candidates = self._autocorrelation_analysis(bass_osf)
            all_candidates.extend([(bpm, score * 1.2, 'bass') for bpm, score, _ in bass_candidates[:3]])
            if len(bass_candidates) > 0:
                method_best_scores['bass'] = bass_candidates[0][1]
        
        if mid_osf is not None:
            mid_candidates = self._autocorrelation_analysis(mid_osf)
            all_candidates.extend([(bpm, score * 1.0, 'mid') for bpm, score, _ in mid_candidates[:3]])
            if len(mid_candidates) > 0:
                method_best_scores['mid'] = mid_candidates[0][1]
        
        if flux_osf is not None:
            flux_candidates = self._autocorrelation_analysis(flux_osf)
            all_candidates.extend([(bpm, score * 0.9, 'flux') for bpm, score, _ in flux_candidates[:3]])
            if len(flux_candidates) > 0:
                method_best_scores['flux'] = flux_candidates[0][1]
        
        if len(all_candidates) == 0:
            self.bpm_confidence = 0.0
            return
        
        # Sort all candidates by score
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Find consensus among top candidates
        tempo_clusters = {}
        for bpm, score, method in all_candidates[:10]:
            found_cluster = False
            for cluster_bpm in list(tempo_clusters.keys()):
                if abs(bpm - cluster_bpm) < 3:
                    tempo_clusters[cluster_bpm].append((bpm, score, method))
                    found_cluster = True
                    break
            
            if not found_cluster:
                tempo_clusters[bpm] = [(bpm, score, method)]
        
        # Score clusters
        cluster_scores = {}
        cluster_methods = {}
        for cluster_bpm, members in tempo_clusters.items():
            avg_bpm = np.mean([bpm for bpm, _, _ in members])
            total_score = sum([score for _, score, _ in members])
            unique_methods = set([method for _, _, method in members])
            cluster_methods[avg_bpm] = unique_methods
            consensus_bonus = len(unique_methods) * 0.3
            
            cluster_scores[avg_bpm] = total_score + consensus_bonus
        
        best_bpm = max(cluster_scores.keys(), key=lambda k: cluster_scores[k])
        best_cluster_score = cluster_scores[best_bpm]
        contributing_methods = cluster_methods[best_bpm]
        
        # IMPROVED CONFIDENCE CALCULATION
        raw_scores = [method_best_scores[m] for m in contributing_methods if m in method_best_scores]
        if len(raw_scores) > 0:
            avg_raw_score = np.mean(raw_scores)
            base_confidence = np.clip((avg_raw_score - 0.2) / 0.6, 0, 1)
        else:
            base_confidence = 0.3
        
        method_agreement = len(contributing_methods) / 3.0
        
        # Check if detected tempo is consistently different from current belief
        tempo_diff = abs(best_bpm - self.tempo_belief)
        
        # If detection strongly disagrees with lock, consider breaking lock
        if self.tempo_locked and tempo_diff > 5:
            # Check if this disagreement is consistent
            if not hasattr(self, '_disagreement_counter'):
                self._disagreement_counter = 0
            
            self._disagreement_counter += 1
            
            # After 5 consecutive disagreements, break the lock
            if self._disagreement_counter > 5:
                print(f"Breaking tempo lock: detected {best_bpm:.1f} vs locked {self.tempo_belief:.1f}")
                self.tempo_locked = False
                self.tempo_lock_strength = 0.0
                self._disagreement_counter = 0
                self.frames_at_current_tempo = 0
        else:
            if hasattr(self, '_disagreement_counter'):
                self._disagreement_counter = max(0, self._disagreement_counter - 1)
        
        # Check tempo stability
        tempo_stability = 1.0
        if len(self.tempo_history) >= 20:
            recent_tempos = self.tempo_history.get_ordered(20)
            tempo_std = np.std(recent_tempos)
            if tempo_std < 1.0:
                tempo_stability = 1.0
                self.frames_at_current_tempo += 1
            elif tempo_std < 2.0:
                tempo_stability = 0.9
            elif tempo_std < 3.0:
                tempo_stability = 0.7
                self.frames_at_current_tempo = 0
            else:
                tempo_stability = 0.4
                self.frames_at_current_tempo = 0
        
        # How close to current belief?
        if tempo_diff < 2:
            tempo_consistency = 1.0
        elif tempo_diff < 5:
            tempo_consistency = 0.85
        elif tempo_diff < 10:
            tempo_consistency = 0.6
        else:
            tempo_consistency = 0.3
        
        uncertainty_factor = np.clip(1.0 - (self.tempo_variance / 100.0), 0.3, 1.0)
        
        # Combine factors
        confidence = (
            base_confidence ** 0.4 *
            method_agreement ** 0.2 *
            tempo_stability ** 0.25 *
            tempo_consistency ** 0.1 *
            uncertainty_factor ** 0.05
        )
        
        confidence = min(0.95, confidence)
        
        # Update tempo lock status (less aggressive)
        if self.frames_at_current_tempo > 20 and tempo_stability > 0.85 and base_confidence > 0.6:
            self.tempo_locked = True
            self.tempo_lock_strength = min(0.7, self.tempo_lock_strength + 0.05)  # Max 0.7 instead of 1.0
        elif tempo_stability < 0.6:
            self.tempo_lock_strength = max(0.0, self.tempo_lock_strength - 0.1)
            if self.tempo_lock_strength < 0.3:
                self.tempo_locked = False
        
        # Update with less sticky behavior
        if confidence > 0.2:
            innovation = best_bpm - self.tempo_belief
            
            # Reduced lock resistance
            if self.tempo_locked:
                kalman_gain = 0.15 * (1 - self.tempo_lock_strength * 0.5)  # More responsive
            else:
                kalman_gain = (self.tempo_variance * confidence) / (self.tempo_variance * confidence + 10)
            
            self.tempo_belief = self.tempo_belief + kalman_gain * innovation
            self.tempo_variance = (1 - kalman_gain * confidence) * self.tempo_variance + 0.5
            self.tempo_variance = np.clip(self.tempo_variance, 1.0, 100.0)
            
            # Less aggressive smoothing
            if self.tempo_locked:
                smoothing = 0.85  # Reduced from 0.95
            elif confidence > 0.6:
                smoothing = 0.75  # Reduced from 0.85
            elif confidence > 0.4:
                smoothing = 0.80  # Reduced from 0.90
            else:
                smoothing = 0.85  # Reduced from 0.93
            
            self.last_bpm = smoothing * self.last_bpm + (1 - smoothing) * self.tempo_belief
            
            # Less aggressive snapping - only when VERY stable and close
            if self.tempo_locked and tempo_stability > 0.9 and self.frames_at_current_tempo > 30:
                common_bpms = [120, 128, 140, 150, 160, 174]
                for common_bpm in common_bpms:
                    if abs(self.last_bpm - common_bpm) < 1.0:  # Tighter tolerance
                        self.last_bpm = common_bpm
                        self.tempo_belief = common_bpm
                        break
            
            self.tempo_history.append(self.last_bpm)
        else:
            self.tempo_variance = min(100.0, self.tempo_variance * 1.05)
        
        self.bpm_confidence = confidence


    def get_beat_info(self):
        """
        Get current beat tracking information
        
        Returns:
            dict: {
                'bpm': Current BPM estimate,
                'confidence': Confidence in BPM (0-1),
                'phase': Current beat phase (0-1),
                'tempo_variance': Uncertainty in tempo,
                'time_to_next_beat': Seconds until next beat
            }
        """
        beat_duration = 60.0 / self.tempo_belief if self.tempo_belief > 0 else 0
        time_to_next = beat_duration * (1.0 - self.beat_phase) if beat_duration > 0 else 0
        
        return {
            'bpm': self.last_bpm,
            'confidence': self.bpm_confidence,
            'phase': self.beat_phase,
            'tempo_variance': self.tempo_variance,
            'time_to_next_beat': time_to_next
        }

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
                'band_centers': 1D array (32) - Center frequency of each band (Hz)
                'band_edges': 1D array (33) - Edge frequencies of bands (Hz)
                'bpm': float - Estimated beats per minute
                'bpm_confidence': float - Confidence in BPM estimate (0-1)
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
                'band_centers': self.band_centers.copy(),
                'band_edges': self.band_edges.copy(),
                'bpm': self.last_bpm,
                'bpm_confidence': self.bpm_confidence,
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
        
        return {
            'raw_bands': raw,
            'norm_short': norm_short,
            'norm_long': norm_long,
            'band_centers': self.band_centers.copy(),
            'band_edges': self.band_edges.copy(),
            'bpm': self.last_bpm,
            'bpm_confidence': self.bpm_confidence,
            'timestamp': time.time(),
            'averaging_method': 'exponential' if self.use_exponential else 'mean'
        }

    
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
        self.analysis_thread = threading.Thread(target=self.analyze_audio)
        self.analysis_thread.start()
        self.stream = sd.InputStream(
            channels=self.CHANNELS,
            samplerate=self.RATE,
            blocksize=self.CALLBACK_BLOCKSIZE,
            callback=self.audio_callback,
            device=self.device
        )
        self.stream.start()

    def stop(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        if self.analysis_thread is not None:
            self.analysis_thread.join()

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
        
        # Update info text with confidence
        current_bands = analysis['raw_bands'][0]
        total_power = np.sum(current_bands)
        max_band = np.argmax(current_bands)
        max_freq = self.analyzer.band_centers[max_band]
        
        # Color code BPM by confidence
        confidence = analysis['bpm_confidence']
        if confidence > 0.5:
            conf_str = "HIGH"
        elif confidence > 0.3:
            conf_str = "MED"
        else:
            conf_str = "LOW"
        
        info_str = (
            f"BPM: {analysis['bpm']:.1f} (conf: {confidence})  |  "
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