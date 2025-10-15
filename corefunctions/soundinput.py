import sounddevice as sd
import numpy as np
import threading
import time
from queue import Queue
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    def __init__(self, device=None, device_name=None, use_loopback=False):
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
        
        print("-" * 80 + "\n")
        
        # Audio parameters
        self.CHUNK = 2048
        self.CHANNELS = 1
        self.device = device
        
        # Analysis rate: 40 FPS
        self.FPS = 40
        self.frame_time = 1.0 / self.FPS

        # Analysis storage and threading
        self.data_queue = Queue()
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
        
        # Store spectrum history (5 seconds at 40Hz = 200 frames)
        self.history_len = 200
        self._spectrum_lock = threading.Lock()
        self.spectrum_history = np.zeros((self.history_len, len(self.freq_bins)))
        
        # Maximum tracking for normalization
        self.max_magnitude = 1e-10
        self.max_decay = 0.999
        
        # Extended analysis features
        self.num_bands = 32
        self.band_history_len = 1000
        
        # Create logarithmic frequency bands from 10Hz to 22kHz
        self.band_edges = np.logspace(np.log10(10), np.log10(22000), self.num_bands + 1)
        self.band_centers = np.sqrt(self.band_edges[:-1] * self.band_edges[1:])
        
        # Create masks for each band
        self.band_masks = []
        for i in range(self.num_bands):
            mask = (self.freq_bins >= self.band_edges[i]) & (self.freq_bins < self.band_edges[i+1])
            self.band_masks.append(mask)
        
        # Storage for band power history (1000 frames x 32 bands)
        self._band_lock = threading.Lock()
        self.band_power_history = np.zeros((self.band_history_len, self.num_bands))
        
        # BPM detection parameters
        self.bpm_range = (60, 300)  # Typical electronic music range
        self.onset_history = np.zeros(self.band_history_len)
        self.spectral_flux_history = np.zeros(self.band_history_len)
        self.last_bpm = 120.0
        self.bpm_confidence = 0.0
        self.bpm_update_counter = 0
        self.prev_magnitudes = None

    def audio_callback(self, indata, frames, time_info, status):
        self.data_queue.put(indata.copy())

    def analyze_audio(self):
        while self.running:
            frame_start = time.time()
            
            if not self.data_queue.empty():
                # Get audio data
                data = self.data_queue.get()[:,0]
                
                # Apply window and compute FFT
                windowed = data * self.window
                fft = np.fft.rfft(windowed)
                magnitudes = np.abs(fft)

                try:
                    self.magnitudes = self.magnitudes * self.max_decay + magnitudes * (1 - self.max_decay)
                except:  # noqa: E722
                    self.magnitudes = magnitudes

                # Update spectrum history
                with self._spectrum_lock:
                    self.spectrum_history = np.roll(self.spectrum_history, 1, axis=0)
                    self.spectrum_history[0] = magnitudes/(self.magnitudes+10E-10)
                
                # Calculate band powers and update history
                band_powers = np.zeros(self.num_bands)
                for i, mask in enumerate(self.band_masks):
                    if np.any(mask):
                        band_powers[i] = np.mean(magnitudes[mask] ** 2)  # Power = magnitude squared
                
                # Calculate spectral flux (better onset detector)
                if self.prev_magnitudes is not None:
                    diff = magnitudes - self.prev_magnitudes
                    spectral_flux = np.sum(np.maximum(diff, 0))  # Only positive changes
                else:
                    spectral_flux = 0
                self.prev_magnitudes = magnitudes.copy()
                
                with self._band_lock:
                    # Roll and update band power history
                    self.band_power_history = np.roll(self.band_power_history, 1, axis=0)
                    self.band_power_history[0] = band_powers
                    
                    # Use spectral flux for onset detection (better than simple sum)
                    self.spectral_flux_history = np.roll(self.spectral_flux_history, 1)
                    self.spectral_flux_history[0] = spectral_flux
                    
                    # Also keep the bass energy onset
                    onset_strength = np.sum(band_powers[2:12])  # Focus on ~40Hz to ~600Hz
                    self.onset_history = np.roll(self.onset_history, 1)
                    self.onset_history[0] = onset_strength
                
                # Update BPM estimation periodically (every 40 frames = 1 second at 40fps)
                self.bpm_update_counter += 1
                if self.bpm_update_counter >= 40:
                    self._update_bpm_estimate()
                    self.bpm_update_counter = 0
            
            # Maintain 40 FPS
            elapsed = time.time() - frame_start
            sleep_time = max(0, self.frame_time - elapsed)
            time.sleep(sleep_time)

    def _update_bpm_estimate(self):
        """Estimate BPM using improved onset detection and autocorrelation with peak finding"""
        with self._band_lock:
            # Use last 600 frames for BPM detection (15 seconds at 40fps)
            analysis_length = min(600, len(self.spectral_flux_history))
            onset_data = self.spectral_flux_history[:analysis_length].copy()
            
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
            min_lag = int(self.FPS * 60 / self.bpm_range[1])  # Frames for max BPM (180)
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
        with self._spectrum_lock:
            return self.freq_bins.copy(), self.spectrum_history.copy()

    def get_sound(self):
        """Get the current spectrum analysis"""
        with self._spectrum_lock:
            return self.spectrum_history[0][2:6].sum()

    def get_all_sound(self):
        """Get the current spectrum analysis"""
        with self._spectrum_lock:
            return self.spectrum_history[0][2:31].mean()

    def get_extended_analysis(self):
        """
        Returns a dictionary with comprehensive audio analysis data
        
        Returns:
            dict: {
                'raw_bands': 2D array (1000 x 32) - Raw power in each frequency band
                'norm_20': 2D array (1000 x 32) - Normalized to mean of last 20 frames
                'norm_100': 2D array (1000 x 32) - Normalized to mean of last 100 frames
                'band_centers': 1D array (32) - Center frequency of each band (Hz)
                'band_edges': 1D array (33) - Edge frequencies of bands (Hz)
                'bpm': float - Estimated beats per minute
                'bpm_confidence': float - Confidence in BPM estimate (0-1)
                'timestamp': float - Current time
            }
        """
        with self._band_lock:
            raw = self.band_power_history.copy()
            
            # Calculate normalized versions
            # Norm 20: normalize to mean of last 20 points
            mean_20 = np.mean(raw[:20], axis=0, keepdims=True)
            mean_20 = np.where(mean_20 < 1e-10, 1e-10, mean_20)  # Avoid division by zero
            norm_20 = raw / mean_20
            
            # Norm 100: normalize to mean of last 100 points
            mean_100 = np.mean(raw[:100], axis=0, keepdims=True)
            mean_100 = np.where(mean_100 < 1e-10, 1e-10, mean_100)
            norm_100 = raw / mean_100
            
            return {
                'raw_bands': raw,
                'norm_20': norm_20,
                'norm_100': norm_100,
                'band_centers': self.band_centers.copy(),
                'band_edges': self.band_edges.copy(),
                'bpm': self.last_bpm,
                'bpm_confidence': self.bpm_confidence,
                'timestamp': time.time()
            }

    
    def get_current_bands(self, normalize='none'):
        """
        Get just the current frame's band values
        
        Args:
            normalize: 'none', '20', or '100' for different normalizations
            
        Returns:
            1D array of 32 band values
        """
        data = self.get_extended_analysis()
        
        if normalize == '20':
            return data['norm_20'][0]
        elif normalize == '100':
            return data['norm_100'][0]
        else:
            return data['raw_bands'][0]

    def start(self):
        self.running = True
        self.analysis_thread = threading.Thread(target=self.analyze_audio)
        self.analysis_thread.start()
        self.stream = sd.InputStream(
            channels=self.CHANNELS,
            samplerate=self.RATE,
            blocksize=self.CHUNK,
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
        self.ax_norm20 = self.fig.add_subplot(gs[1])
        self.ax_norm100 = self.fig.add_subplot(gs[2])
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
        
        # Initialize norm 20 plot
        self.img_norm20 = self.ax_norm20.imshow(
            analysis['norm_20'][:display_frames].T,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            extent=[0, time_extent, 0, self.analyzer.num_bands],
            cmap='viridis',
            vmin=0, vmax=2
        )
        self.ax_norm20.set_ylabel('Band Index')
        self.ax_norm20.set_title('Normalized to Last 20 Frames (0.5s)')
        plt.colorbar(self.img_norm20, ax=self.ax_norm20, label='Relative Power')
        
        # Initialize norm 100 plot
        self.img_norm100 = self.ax_norm100.imshow(
            analysis['norm_100'][:display_frames].T,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            extent=[0, time_extent, 0, self.analyzer.num_bands],
            cmap='plasma',
            vmin=0, vmax=2
        )
        self.ax_norm100.set_xlabel('Time (seconds ago)')
        self.ax_norm100.set_ylabel('Band Index')
        self.ax_norm100.set_title('Normalized to Last 100 Frames (2.5s)')
        plt.colorbar(self.img_norm100, ax=self.ax_norm100, label='Relative Power')
        
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
        
        for ax in [self.ax_raw, self.ax_norm20, self.ax_norm100]:
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
        self.img_norm20.set_array(analysis['norm_20'][:display_frames].T)
        self.img_norm100.set_array(analysis['norm_100'][:display_frames].T)
        
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
        
        return self.img_raw, self.img_norm20, self.img_norm100, self.info_text



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
    # Option 1: Use loopback (system audio output)
    #analyzer = MicrophoneAnalyzer(use_loopback=False)
    
    # Option 2: Use a specific microphone
    analyzer = MicrophoneAnalyzer(device_name="HD Pro Webcam C920")
    
    # Option 3: Use default input device
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