import sounddevice as sd
import numpy as np
import threading
import time
from queue import Queue
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    def __init__(self, device=None, device_name=None):
        # Print device information
        devices = list_audio_devices()
        
        # Handle device selection
        if device_name is not None:
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
        self.RATE = int(devices[device]['default_samplerate'])
        self.device = device

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
        
        # Store spectrum history (5 seconds at 20Hz = 100 frames)
        self.history_len = 100
        self._spectrum_lock = threading.Lock()
        self.spectrum_history = np.zeros((self.history_len, len(self.freq_bins)))
        
        # Maximum tracking for normalization
        self.max_magnitude = 1e-10
        self.max_decay = 0.999

    def audio_callback(self, indata, frames, time_info, status):
        self.data_queue.put(indata.copy())

    def analyze_audio(self):
        while self.running:
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

            time.sleep(0.01)

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
        
        # Setup the figure
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Initialize empty image
        freqs, history = analyzer.get_spectrum_history()
        self.img = self.ax.imshow(
            history,
            aspect='auto',
            origin='upper',
            interpolation='nearest',
            vmin=1.3, vmax=2,
            extent=[freqs[0], freqs[-1], -5, 0],  # 5 seconds time window
            cmap='magma'  # Use a perceptually uniform colormap
        )
        
        # Configure the plot
        self.ax.set_yscale('linear')
        self.ax.set_xscale('log')  # Logarithmic frequency scale
        self.ax.set_xlim(20, analyzer.RATE/2)  # From 20Hz to Nyquist frequency
        self.ax.set_ylim(-5, 0)
        
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Time (seconds)')
        self.ax.set_title('Real-time Spectrum Analysis')
        
        # Add colorbar
        plt.colorbar(self.img, ax=self.ax, label='Magnitude (normalized dB)')
        
        # Add frequency markers
        self.add_frequency_markers()
        
        # Add bass range indicator
        self.ax.axvspan(analyzer.bass_range[0], analyzer.bass_range[1], 
                       color='red', alpha=0.2, label='Bass Range')
        self.ax.legend()

        # Adjust layout
        plt.tight_layout()

    def add_frequency_markers(self):
        # Add frequency markers at notable points
        markers = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        self.ax.set_xticks(markers)
        self.ax.set_xticklabels([str(x) for x in markers])
        
        # Add minor grid lines
        self.ax.grid(True, which='both', alpha=0.2)

    def update(self, frame):
        freqs, spectrum = self.analyzer.get_spectrum_history()
        self.img.set_array(spectrum)
        return self.img,

    def start(self):
        self.ani = FuncAnimation(
            self.fig, 
            self.update,
            interval=50,  # 20 FPS
            blit=True,
            cache_frame_data=False
        )
        plt.show()

if __name__ == "__main__":
    print("Bass Detection System Starting...")
    print("Use Ctrl+C to stop")
    
    analyzer = MicrophoneAnalyzer(device_name="TONOR")  # Change this to match your device
    analyzer.start()
    
    try:
        plotter = SpectrogramPlotter(analyzer)
        plotter.start()
    except KeyboardInterrupt:
        print("\nStopping analysis...")
        analyzer.stop()
        plt.close('all')