import numpy as np
from sacn import sACNsender
import math
import threading
import queue
try:
    from corefunctions.fast_pixel_extract import extract_and_pack_pixels_unchecked, process_all_universes
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available, using slower numpy operations")
class SACNPixelSender:
    def __init__(self, receivers,start_universe=1):
        """
        Initialize the SACNPixelSender with receiver configurations.
        :param receivers: List of dicts, each with 'ip', 'pixel_count', and 'addressing_array' keys.
        """
        self.receivers = receivers
        self.sender = sACNsender()
        
        # Enable manual flush for synchronized sending
        self.sender.manual_flush = True
        self.sender.start()

        # Set up universes for each receiver
        self.receiver_universes = []
        universe_counter = start_universe
        
        # Pre-compute and cache coordinate arrays (they never change)
        self._cached_coords = []
        self._universe_slices = []
        
        for receiver in receivers:
            universe_count = math.ceil(receiver['pixel_count'] / 170)
            receiver_universes = list(range(universe_counter, universe_counter + universe_count))
            self.receiver_universes.append(receiver_universes)
            universe_counter += universe_count

            # Activate universes for this receiver
            for universe in receiver_universes:
                self.sender.activate_output(universe)
                self.sender[universe].destination = receiver['ip']
            
            # Pre-compute clipped coordinates (they never change frame-to-frame)
            # Clip to actual frame dimensions for safety
            x_coords = np.clip(receiver['addressing_array'][:, 0], 0, 127).astype(np.int32)
            y_coords = np.clip(receiver['addressing_array'][:, 1], 0, 299).astype(np.int32)
            self._cached_coords.append((x_coords, y_coords))
            
            # Pre-compute universe slice ranges for this receiver
            slices = []
            for i in range(universe_count):
                start = i * 170
                end = min(start + 170, receiver['pixel_count'])
                needs_padding = (end - start) * 3 < 510
                slices.append((start, end, needs_padding))
            self._universe_slices.append(slices)
        
        # Pre-allocate buffers for data extraction (avoids allocations during send)
        self._receiver_buffers = []
        self._universe_buffers = []
        self._universe_starts = []
        self._universe_ends = []
        
        for receiver in receivers:
            # Flat buffer to hold extracted pixel data (N*3 bytes)
            pixel_count = receiver['pixel_count']
            self._receiver_buffers.append(np.empty(pixel_count * 3, dtype=np.uint8))
            
            # Pre-compute universe start/end indices
            universe_count = math.ceil(pixel_count / 170)
            starts = np.array([i * 170 for i in range(universe_count)], dtype=np.int32)
            ends = np.array([min((i + 1) * 170, pixel_count) for i in range(universe_count)], dtype=np.int32)
            self._universe_starts.append(starts)
            self._universe_ends.append(ends)
            
            # Create 2D buffer array for batch universe processing (for Numba)
            universe_buffer_2d = np.zeros((universe_count, 510), dtype=np.uint8)
            self._universe_buffers.append(universe_buffer_2d)
        
        # Async sending support
        self._send_queue = queue.Queue(maxsize=2)  # Small queue to avoid lag
        self._send_thread = None
        self._stop_thread = False
        self._async_enabled = False

    def enable_async_send(self):
        """Enable asynchronous sending in a background thread"""
        if self._async_enabled:
            return
        
        self._async_enabled = True
        self._stop_thread = False
        self._send_thread = threading.Thread(target=self._send_worker, daemon=True)
        self._send_thread.start()
        print("sACN async sending enabled")
    
    def disable_async_send(self):
        """Disable asynchronous sending and wait for thread to finish"""
        if not self._async_enabled:
            return
        
        self._async_enabled = False
        self._stop_thread = True
        if self._send_thread:
            self._send_thread.join(timeout=1.0)
        print("sACN async sending disabled")
    
    def _send_worker(self):
        """Background worker that sends frames from queue"""
        while not self._stop_thread:
            try:
                # Get frame data from queue (timeout to check stop flag)
                frame_data = self._send_queue.get(timeout=0.1)
                if frame_data is None:  # Poison pill to stop
                    break
                
                # Actually send the data
                self._send_immediate(frame_data)
                
            except queue.Empty:
                continue
    
    def create_mask(self, height, width):
        """
        Creates a binary mask showing which pixels are mapped by receivers.
        
        :param height: Height of the source image
        :param width: Width of the source image
        :return: numpy array of shape (height, width) with 1s where pixels are mapped
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for receiver in self.receivers:
            # Clip coordinates to valid image boundaries
            x_coords = np.clip(receiver['addressing_array'][:, 0], 0, height - 1)
            y_coords = np.clip(receiver['addressing_array'][:, 1], 0, width - 1)
            
            # Set mapped pixels to 1
            mask[x_coords, y_coords] = 1
            
        return mask

    def send(self, source_array, verify=False):
        """
        Send pixel data to all configured receivers.
        If async is enabled, queues data for background sending.
        Otherwise sends immediately.
        """
        if self._async_enabled:
            # Drop frame if queue is full (prevents lag buildup)
            # Note: We don't copy the array - caller must not modify after calling send()
            try:
                self._send_queue.put_nowait((source_array, verify))
            except queue.Full:
                pass  # Skip this frame if queue is full
        else:
            self._send_immediate((source_array, verify))
    
    def _send_immediate(self, frame_data):
        """
        Send pixel data using optimized Numba functions (releases GIL).
        :param frame_data: Tuple of (source_array, verify)
        """
        source_array, verify = frame_data
        
        if verify:
            print(f"[ImageToDMX] Sending frame shape={source_array.shape}")
        
        # Process each receiver using optimized Numba (releases GIL)
        for rx_idx, (receiver, universes) in enumerate(zip(self.receivers, self.receiver_universes)):
            x_coords, y_coords = self._cached_coords[rx_idx]
            pixel_buffer = self._receiver_buffers[rx_idx]
            
            if NUMBA_AVAILABLE:
                # Ultra-fast: extract pixels with no bounds checking (coords pre-validated)
                extract_and_pack_pixels_unchecked(
                    source_array,
                    x_coords,
                    y_coords,
                    pixel_buffer
                )
                
                # Process all universes in parallel
                universe_buffer_2d = self._universe_buffers[rx_idx]
                process_all_universes(
                    pixel_buffer,
                    self._universe_starts[rx_idx],
                    self._universe_ends[rx_idx],
                    universe_buffer_2d
                )
                
                # Send all universes
                for u, universe in enumerate(universes):
                    self.sender[universe].dmx_data = universe_buffer_2d[u].tobytes()
                    
            else:
                # Fallback to numpy
                height, width = source_array.shape[:2]
                x_valid = np.minimum(x_coords, height - 1)
                y_valid = np.minimum(y_coords, width - 1)
                pixels = source_array[x_valid, y_valid]
                pixel_buffer[:] = pixels.flatten()
                
                # Pack into universe buffers
                slices = self._universe_slices[rx_idx]
                universe_buffer_2d = self._universe_buffers[rx_idx]
                
                for u, (start, end, needs_padding) in enumerate(slices):
                    byte_start = start * 3
                    byte_end = end * 3
                    byte_count = byte_end - byte_start
                    
                    universe_buffer_2d[u, :byte_count] = pixel_buffer[byte_start:byte_end]
                    if needs_padding:
                        universe_buffer_2d[u, byte_count:] = 0
                    
                    self.sender[universes[u]].dmx_data = universe_buffer_2d[u].tobytes()
        
        # Flush all universes
        self.sender.flush()

    def close(self):
        """
        Properly close the sACN sender
        """
        # Stop async thread if running
        self.disable_async_send()
        self.sender.stop()

    def analyze_row_groups(self, max_pixels_per_group=170):
        """
        Analyze and group pixels in rows that belong to the same receiver.
        
        :param max_pixels_per_group: Maximum number of pixels per group (default 170 for sACN universe limit)
        :return: Dictionary mapping receiver indices to lists of pixel groups
        """
        receiver_groups = {}
        
        for idx, receiver in enumerate(self.receivers):
            # Get coordinates from addressing array
            coordinates = receiver['addressing_array']
            #find the unique rows
            rows = np.unique(coordinates[:, 0])
            #find the number of pixels in each row
            row_counts = {row: np.sum(coordinates[:, 0] == row) for row in rows}         
            #initialize the group list
            groups = []
            #initialize the current group
            current_group = []
            #initialize the current row
            current_row = None
            #iterate through the rows
            group_pixel_count = 0
            pixels_in_group = []
            for row in rows:
                if current_row is None:
                    current_row = row
                #if the row is different from the current row
                row_count = row_counts[row]
                if (group_pixel_count + row_count) <= max_pixels_per_group:
                    #store the current group
                    current_group.append(row)
                    #reset the group count
                    group_pixel_count += row_count
                    #reset the current group
                else:
                    groups.append(current_group)
                    current_group = [row]
                    pixels_in_group.append(group_pixel_count)
                    group_pixel_count = row_count
            #handle the last group
            groups.append(current_group)
            pixels_in_group.append(group_pixel_count)
            print(groups, pixels_in_group,receiver['ip'])

# The rest of the code (generate_frame_data and main function) remains the same

def generate_frame_data():
    """
    Generate random RGB pixel data for each frame.
    In a real application, this would pull data from your actual source.
    """
    width, height = 100, 100  # Example dimensions
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

def make_indicesHS(filename):
    in_list=np.loadtxt(filename, delimiter=',').tolist()
    indices = []
    for sublist in in_list:       
        if sublist[2]>0:
            for m in range(int(sublist[2])):
                indices.append([sublist[0], m+sublist[1]])
        else:
            for m in range(int(-sublist[2])):
                indices.append([sublist[0], sublist[1]-sublist[2]-1-m])   
    return np.array(indices).astype(int)

def make_indicesVS(filename):
    in_list=np.loadtxt(filename, delimiter=',').tolist()
    indices = []
    for sublist in in_list:       
        if sublist[2]>0:
            for m in range(int(sublist[2])):
                indices.append([m+sublist[1], sublist[0]])
        else:
            for m in range(int(-sublist[2])):
                indices.append([sublist[1]-sublist[2]-1-m, sublist[0]])   
    return np.array(indices).astype(int)

def make_indices_V_rect_alternate(width, height,start):
    indices = []
    for x in range(width):
        if x % 2 == 0:
            for y in range(height):
                indices.append([y, x+start])
        else:
            for y in range(height-1, -1, -1):
                indices.append([y, x+start])
    return np.array(indices).astype(int)

def main():
    receivers = [
        {
            'ip': '192.168.68.121',
            'pixel_count': 3500,
            #'addressing_array':make_indicesH()
            'addressing_array':make_indicesHS(r"data.txt")
        }
    ]

    sender = SACNPixelSender(receivers)
    sender.analyze_row_groups(255)
    sender.close()

if __name__ == "__main__":
    main()