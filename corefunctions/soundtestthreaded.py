import numpy as np
import sounddevice as sd
import time
import threading
import heapq
import soundfile as sf
from pathlib import Path
import librosa

class AudioCache:
    def __init__(self):
        self._cache = {}

    def get(self, filename, duration, sample_rate, channels, skip_time=0,volume=1):
        cache_key = f"{filename}_{skip_time}"
        if cache_key not in self._cache:
            print(f"Loading {cache_key} into cache")
            # Load the audio file
            if filename.suffix == '.mp3':
                audio_data, _ = librosa.load(filename, sr=sample_rate, mono=False)
                audio_data = np.transpose(audio_data)
                if audio_data.ndim == 1:
                    audio_data = np.column_stack((audio_data, audio_data))
            elif filename.suffix == '.flac':
                audio_data, _ = sf.read(filename, dtype='float32')
                if audio_data.ndim == 1:
                    audio_data = np.column_stack((audio_data, audio_data))

            elif Path(filename).exists():
                audio_data, _ = sf.read(filename, dtype='float32')
                if audio_data.ndim == 1:
                    audio_data = np.column_stack((audio_data, audio_data))
            else:
                print('File not found:', filename)
                audio_data = np.zeros((int(sample_rate * duration), channels), dtype=np.float32)

            # Apply skip time and trim to the required duration
            skip_samples = int(sample_rate * skip_time)
            total_samples = int(sample_rate * (duration + skip_time))
            if skip_samples > 0 and skip_samples < len(audio_data):
                audio_data = audio_data[skip_samples:total_samples]*volume
            audio_data = audio_data[:int(sample_rate * duration)]*volume
            self._cache[cache_key] = audio_data

        return np.copy(self._cache[cache_key])
    
class AudioEvent:
    def __init__(self, sound_file, execution_time, duration, repeat_interval=None, name=None, skip_time=0):
        self.sound_file = sound_file
        self.execution_time = execution_time
        self.duration = duration
        self.skip_time = skip_time
        self.audio_data = None
        self.position = 0
        self.repeat_interval = repeat_interval
        self.id = id(self)
        self.next_id = 0
        self.is_active = False
        self.name = name
        self.fade = None
        self.fade_remaining = 0
        self.fade_in = None
        self.fade_in_remaining = 0
        self.is_looping = repeat_interval is not None

    def __lt__(self, other):
        return self.execution_time < other.execution_time

class AudioEngine:
    def __init__(self):
        self.running = True
        self.fps = 30
        self.channels = 2
        self.sample_rate = 44100
        self.buffer_size = 1024
        self.audio_buffer = np.zeros((self.buffer_size, self.channels), dtype=np.float32)
        self.event_heap = []
        self.event_dict = {}
        self.active_events = set()
        self.lock = threading.RLock()
        self.audio_cache = AudioCache()

    def safe_lock(self, timeout=1.0):
        acquired = self.lock.acquire(timeout=timeout)
        if not acquired:
            print("Warning: Lock acquisition timed out")
            return False
        return True

    def load_audio(self, filename, duration, skip_time=0,volume=1):
        return self.audio_cache.get(filename, duration, self.sample_rate, self.channels, skip_time,volume)



    def start_audio_stream(self):
        self.stream = sd.OutputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            callback=self.audio_callback,
            blocksize=self.buffer_size
        )
        self.stream.start()

    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        with self.lock:
            self.update_audio_buffer()
            outdata[:] = self.audio_buffer


    def update_audio_buffer(self):
        if not self.safe_lock():
            return
            
        try:
            self.audio_buffer.fill(0)
            current_time = time.time()
            
            # Start new events
            while self.event_heap and self.event_heap[0].execution_time <= current_time:
                event = heapq.heappop(self.event_heap)
                if event.id in self.event_dict:
                    event.is_active = True
                    self.active_events.add(event.id)
            
            # Process active events
            events_to_remove = set()
            
            for event_id in list(self.active_events):
                if event_id not in self.event_dict:
                    events_to_remove.add(event_id)
                    continue
                    
                event = self.event_dict[event_id]
                if event.audio_data is None:
                    events_to_remove.add(event_id)
                    continue

                # Check if event should still be playing
                if not event.is_looping and event.position >= len(event.audio_data):
                    events_to_remove.add(event_id)
                    continue

                # Handle looping audio
                if event.is_looping and event.position >= len(event.audio_data):
                    event.position = 0

                # Get the audio chunk
                remaining = len(event.audio_data) - event.position
                chunk_size = min(self.buffer_size, remaining)
                
                if chunk_size <= 0:
                    events_to_remove.add(event_id)
                    continue
                    
                chunk = np.copy(event.audio_data[event.position:event.position + chunk_size])
                
                # Apply fades if needed
                if event.fade_remaining > 0:
                    fade_factor = event.fade_remaining / event.fade
                    chunk *= fade_factor
                    event.fade_remaining -= chunk_size/self.sample_rate
                    if event.fade_remaining <= 0:
                        events_to_remove.add(event_id)
                        continue
                
                if event.fade_in_remaining > 0:
                    progress = (event.fade_in - event.fade_in_remaining) / event.fade_in
                    chunk *= progress
                    event.fade_in_remaining -= chunk_size/self.sample_rate
                
                self.audio_buffer[:len(chunk)] += chunk
                event.position += chunk_size
            
            # Clean up completed events
            for event_id in events_to_remove:
                self.active_events.discard(event_id)
                if event_id in self.event_dict:
                    del self.event_dict[event_id]
                    #print(f"Event {event_id} cancelled")
                    #print(f"After cancellation - Active events: {len(self.active_events)}, Event dict: {len(self.event_dict)}",len(self.event_heap))
            # Normalize if needed
            max_val = np.max(np.abs(self.audio_buffer))
            if max_val > 1:
                self.audio_buffer /= max_val
                
        finally:
            self.lock.release()

    def schedule_event(self, sound_file, execution_time, duration, repeat_interval=None, inname=None, fade_in_duration=None, skip_time=0):
        event = AudioEvent(sound_file, execution_time, duration, repeat_interval, name=inname, skip_time=skip_time)
        # Pre-load the audio data from cache
        event.audio_data = self.load_audio(event.sound_file, duration, skip_time)
        if fade_in_duration is not None:
            event.fade_in = fade_in_duration
            event.fade_in_remaining = fade_in_duration
        with self.lock:
            heapq.heappush(self.event_heap, event)
            self.event_dict[event.id] = event
        return event.id

    def fade_out_audio(self, event_name, duration=5):
        if not self.safe_lock():
            return
            
        try:
            #print(f"Fading out audio for {event_name}")
            for event in list(self.event_dict.values()):
                if event.name == event_name:
                    #print(f"Applying fade to event {event.id}")
                    # Calculate remaining audio duration
                    remaining_time = max((len(event.audio_data) - event.position) / self.sample_rate,0)
                    fade_duration = min(duration, remaining_time)
                    
                    # Stop looping and apply fade
                    event.is_looping = False
                    event.fade = fade_duration
                    event.fade_remaining = fade_duration
                    
        finally:
            self.lock.release()





    def run(self):
        self.start_audio_stream()
        try:
            while self.running:
                time.sleep(1 / self.fps)
        except KeyboardInterrupt:
            print("Stopping the audio engine...")
        finally:
            self.running = False
            self.stream.stop()
            self.stream.close()

class ThreadedAudioEngine(AudioEngine):
    def __init__(self):
        super().__init__()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            self.stream.stop()
            self.stream.close()

if __name__ == "__main__":
    engine = ThreadedAudioEngine()
    engine.start()
    
    # Test cases
    engine.schedule_event("C:\\Users\\diete\\Desktop\\devel-local\\LED-Sign\\media\\sounds\\Rain Into Puddle.wav", time.time() + 1, 5,repeat_interval=4,inname='toot')  # Single play
    
    try:
        while True:
            time.sleep(15)
            #engine.stop_repeating_by_name('toot')
            print("event stopped")
            engine.fade_out_audio('toot')
            time.sleep(40)
            engine.stop()
            break
    except KeyboardInterrupt:
        engine.stop()