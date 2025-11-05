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
        
        # BPM detection parameters
        self.bpm_range = (60, 300)  # Typical electronic music range
        self.onset_history = CircularBuffer(self.band_history_len)
        self.spectral_flux_history = CircularBuffer(self.band_history_len)
        self.last_bpm = 120.0
        self.bpm_confidence = 0.0
        self.bpm_update_counter = 0
        self.prev_magnitudes = None

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
            
            # Bass energy onset
            onset_strength = np.sum(band_powers[2:12])
            self.onset_history.append(onset_strength)
            
            # Update BPM estimation periodically
            self.bpm_update_counter += 1
            if self.bpm_update_counter >= 40:
                self._update_bpm_estimate()
                self.bpm_update_counter = 0
            
            # Maintain precise 40 FPS timing
            elapsed = time.time() - frame_start
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)


    def _update_bpm_estimate(self):
        """Estimate BPM using improved onset detection and autocorrelation with peak finding"""
        # Use last 600 frames for BPM detection (15 seconds at 40fps)
        analysis_length = min(600, len(self.spectral_flux_history))
        onset_data = self.spectral_flux_history.get_ordered(analysis_length)
        
        if len(onset_data) < 100:  # Not enough data yet
            return
        
        # Normalize and remove DC component
        onset_data = onset_data - np.mean(onset_data)
        
        if np.std(onset_data) < 1e-6:  # No significant audio
            self.bpm_confidence = 0.0
            return
        
        onset_data = onset_data / np.std(onset_data)
        
        # Apply envelope following to smooth onset function
        onset_envelope = np.copy(onset_data)
        for i in range(1, len(onset_envelope)):
            if onset_envelope[i] < onset_envelope[i-1]:
                onset_envelope[i] = onset_envelope[i-1] * 0.95 + onset_envelope[i] * 0.05
        
        # Compute autocorrelation
        autocorr = np.correlate(onset_envelope, onset_envelope, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
        
        # Normalize autocorrelation
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        
        # Convert frame indices to BPM range
        min_lag = int(self.FPS * 60 / self.bpm_range[1])  # Frames for max BPM (300)
        max_lag = int(self.FPS * 60 / self.bpm_range[0])  # Frames for min BPM (60)
        
        if max_lag >= len(autocorr):
            max_lag = len(autocorr) - 1
        
        if min_lag >= max_lag:
            return
        
        # Find peaks in the autocorrelation
        valid_autocorr = autocorr[min_lag:max_lag]
        
        # Find peaks with minimum prominence
        peaks, properties = find_peaks(valid_autocorr, prominence=0.1, distance=5)
        
        if len(peaks) == 0:
            # No clear peaks, fall back to maximum
            peak_idx = np.argmax(valid_autocorr) + min_lag
            confidence = valid_autocorr[np.argmax(valid_autocorr)]
        else:
            # Get the most prominent peak
            peak_prominences = properties['prominences']
            best_peak = peaks[np.argmax(peak_prominences)]
            peak_idx = best_peak + min_lag
            confidence = valid_autocorr[best_peak]
        
        estimated_bpm = 60 * self.FPS / peak_idx
        
        # Check for half/double tempo ambiguity
        # If we have a strong peak at half or double tempo, use that instead
        half_tempo_lag = peak_idx * 2
        double_tempo_lag = peak_idx // 2
        
        if double_tempo_lag >= min_lag and double_tempo_lag < max_lag:
            if autocorr[double_tempo_lag] > confidence * 0.9:
                peak_idx = double_tempo_lag
                estimated_bpm = 60 * self.FPS / peak_idx
                confidence = autocorr[double_tempo_lag]
        
        if half_tempo_lag < max_lag:
            if autocorr[half_tempo_lag] > confidence * 0.9:
                peak_idx = half_tempo_lag
                estimated_bpm = 60 * self.FPS / peak_idx
                confidence = autocorr[half_tempo_lag]
        
        # Update confidence
        self.bpm_confidence = confidence
        
        # Only update if confidence is reasonable
        if confidence > 0.2:
            # Smooth BPM estimate more aggressively when confidence is low
            smoothing = 0.85 if confidence < 0.4 else 0.7
            self.last_bpm = smoothing * self.last_bpm + (1 - smoothing) * estimated_bpm
            
            # Snap to common BPM values if close
            common_bpms = [120, 128, 140, 150, 160, 174]
            for common_bpm in common_bpms:
                if abs(self.last_bpm - common_bpm) < 2:
                    self.last_bpm = common_bpm
                    break

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
            f"BPM: {analysis['bpm']:.1f} (conf: {conf_str})  |  "
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