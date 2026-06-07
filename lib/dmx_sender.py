"""sACN/E1.31 DMX pixel sender.

SACNPixelSender maps rendered frame pixels to DMX universes and transmits
them to physical LED receivers over the network. Supports both the sacn
library (standard) and raw UDP sockets (lower latency), and optional
Numba-accelerated pixel extraction for high pixel counts.

Pixel coordinates are specified per receiver in one of two equivalent forms:
  1. Legacy: ``addressing_array`` (Nx2 array of [row, col] indices) plus
     ``pixel_count`` (= len(addressing_array)).
  2. Strip-based: ``strips`` (list of ``core.strip.StripBinding``); the
     sender concatenates each strip's pixel_indices to form the same Nx2
     addressing array internally. Strips are the path forward; the legacy
     form is kept for backwards compatibility.
"""

import numpy as np
from sacn import sACNsender
import math
import threading
import queue
import socket
import struct
import time
import traceback

from core.strip import StripBinding, addressing_array_from_strips
try:
    from lib.pixel_extract import extract_and_pack_pixels_unchecked, process_all_universes
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
def _normalize_receiver(rx: dict) -> dict:
    """Return a dict that has both ``addressing_array`` + ``pixel_count``
    populated, regardless of whether the input used the legacy or strip-based
    form. The strip list (when present) is preserved on the output for any
    downstream consumer that wants to reason per-strip later (Phase 6+).
    """
    if 'strips' in rx and rx['strips']:
        strips = rx['strips']
        # Accept either StripBinding instances or dicts.
        if not isinstance(strips[0], StripBinding):
            from core.strip import strips_from_yaml_list  # local to avoid cycle at import
            strips = strips_from_yaml_list(strips)
        addr = addressing_array_from_strips(strips)
        out = dict(rx)
        out['strips'] = strips
        out['addressing_array'] = addr
        out['pixel_count'] = int(addr.shape[0])
        return out
    if 'addressing_array' not in rx:
        raise ValueError(
            "Receiver must specify either 'addressing_array' or 'strips'."
        )
    out = dict(rx)
    if 'pixel_count' not in out:
        out['pixel_count'] = int(np.asarray(out['addressing_array']).shape[0])
    return out


class SACNPixelSender:
    def __init__(self, receivers,start_universe=1, skip_network=True, use_raw_udp=False, per_receiver_universe=False, bind_ip=""):
        """
        Initialize the SACNPixelSender with receiver configurations.
        :param receivers: List of dicts. Each dict has 'ip' + 'protocol' plus
                          either ('pixel_count', 'addressing_array') for the
                          legacy form or ('strips') for the strip-based form.
        :param skip_network: If True, skip actual network transmission (for testing)
        :param use_raw_udp: If True, use raw UDP sockets instead of sACN library (much faster)
        :param per_receiver_universe: If True, each receiver restarts at start_universe. If False, use global sequential universes.
        :param bind_ip: If set, bind UDP socket to this IP (forces traffic out a specific interface)
        """
        receivers = [_normalize_receiver(rx) for rx in receivers]
        self.receivers = receivers
        self.skip_network = skip_network
        self.use_raw_udp = use_raw_udp
        self.per_receiver_universe = per_receiver_universe

        # Any DDP receiver in the list forces a UDP socket regardless of
        # use_raw_udp — DDP doesn't go through the sacn library at all.
        any_ddp = any(rx.get('protocol', 'sacn').lower() == 'ddp' for rx in receivers)
        any_sacn = any(rx.get('protocol', 'sacn').lower() == 'sacn' for rx in receivers)

        if use_raw_udp or any_ddp:
            # Create UDP socket for raw packet sending
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Set socket to non-blocking for async operation
            self.udp_socket.setblocking(False)
            # Increase send buffer size
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
            # Bind to specific interface if configured
            if bind_ip:
                self.udp_socket.bind((bind_ip, 0))

            # Pre-build sACN packet headers for each universe
            self._sacn_headers = []
            self._sequence_numbers = []
            # Per-receiver DDP sequence (1..15 cycle). 0 means "no sequence"
            # in DDP, which the receiver won't use for drop detection.
            self._ddp_sequence = []
        elif any_sacn:
            self.sender = sACNsender()
            # Enable manual flush for synchronized sending
            self.sender.manual_flush = True
            self.sender.start()

        # Per-receiver protocol cache, lower-cased and defaulted.
        self.receiver_protocols = [
            rx.get('protocol', 'sacn').lower() for rx in receivers
        ]

        # Set up universes for each receiver (sACN only — DDP is byte-offset
        # based and doesn't need universe assignment).
        self.receiver_universes = []
        universe_counter = start_universe

        # Pre-compute and cache coordinate arrays (they never change)
        self._cached_coords = []
        self._universe_slices = []

        for rx_idx, receiver in enumerate(receivers):
            protocol = self.receiver_protocols[rx_idx]
            universe_count = math.ceil(receiver['pixel_count'] / 170)

            # Universe assignment is sACN-only; DDP uses byte offsets and so
            # this list is empty for DDP receivers (kept index-aligned anyway).
            if protocol == 'sacn':
                if per_receiver_universe:
                    receiver_universes = list(range(start_universe, start_universe + universe_count))
                else:
                    receiver_universes = list(range(universe_counter, universe_counter + universe_count))
                    universe_counter += universe_count
            else:  # 'ddp'
                receiver_universes = []
            self.receiver_universes.append(receiver_universes)

            if protocol == 'ddp':
                # DDP needs only a single per-receiver sequence counter
                # (1..15 cycle; 0 means "no sequence"). Packet headers are
                # rebuilt per-chunk because the byte offset changes.
                self._ddp_sequence.append(1)
                # Index-align the sACN state arrays even though unused.
                if hasattr(self, '_sacn_headers'):
                    self._sacn_headers.append([])
                    self._sequence_numbers.append([])
            elif self.use_raw_udp:
                # Pre-build sACN packet headers for each universe of this receiver
                receiver_headers = []
                receiver_seqs = []
                for universe in receiver_universes:
                    # Build sACN E1.31 packet header (126 bytes + 512 DMX data)
                    header = self._build_sacn_header(universe)
                    receiver_headers.append(header)
                    receiver_seqs.append(0)
                self._sacn_headers.append(receiver_headers)
                self._sequence_numbers.append(receiver_seqs)
                self._ddp_sequence.append(1)   # unused
            else:
                # Activate universes for this receiver via the sacn library
                for universe in receiver_universes:
                    self.sender.activate_output(universe)
                    self.sender[universe].destination = receiver['ip']
                    self.sender[universe].multicast = False  # Use unicast for better performance
                    self.sender[universe].ttl = 20  # Reduce TTL for local network
            
            # Pre-compute clipped coordinates (they never change frame-to-frame)
            # Determine max dimensions from addressing array itself
            max_x = receiver['addressing_array'][:, 0].max()
            max_y = receiver['addressing_array'][:, 1].max()
            x_coords = np.clip(receiver['addressing_array'][:, 0], 0, max_x).astype(np.int32)
            y_coords = np.clip(receiver['addressing_array'][:, 1], 0, max_y).astype(np.int32)
            self._cached_coords.append((x_coords, y_coords))

            # Pre-compute universe slice ranges for this receiver
            slices = []
            for i in range(universe_count):
                start = i * 170
                end = min(start + 170, receiver['pixel_count'])
                needs_padding = (end - start) * 3 < 510
                slices.append((start, end, needs_padding))
            self._universe_slices.append(slices)

        # Per-receiver strip plans: list of {group_id, x_coords, y_coords,
        # byte_offset, byte_count}. The hot path iterates these to extract
        # each strip from its own group canvas (Phase 6+); for legacy
        # single-source callers, a 'main' group is synthesized below.
        self._strip_plans: list[list[dict]] = []
        for receiver in receivers:
            plans = []
            byte_offset = 0
            for strip in (receiver.get('strips') or []):
                idx = np.asarray(strip.pixel_indices, dtype=np.int32)
                length = int(idx.shape[0])
                plans.append({
                    'group_id': strip.group_id,
                    'x_coords': idx[:, 0].copy(),
                    'y_coords': idx[:, 1].copy(),
                    'byte_offset': byte_offset,
                    'byte_count': length * 3,
                })
                byte_offset += length * 3
            if not plans:
                # Receiver has no strips (legacy addressing_array-only form);
                # treat the full array as one anonymous strip on 'main'.
                addr = np.asarray(receiver['addressing_array'], dtype=np.int32)
                plans.append({
                    'group_id': 'main',
                    'x_coords': addr[:, 0].copy(),
                    'y_coords': addr[:, 1].copy(),
                    'byte_offset': 0,
                    'byte_count': int(addr.shape[0]) * 3,
                })
            self._strip_plans.append(plans)
        
        # Pre-allocate buffers for data extraction (avoids allocations during send)
        self._receiver_buffers = []
        self._universe_buffers = []
        self._universe_memviews = []
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
            
            # Pre-allocate memoryview objects to avoid tobytes() overhead
            universe_memviews = [memoryview(universe_buffer_2d[u]) for u in range(universe_count)]
            self._universe_memviews.append(universe_memviews)
        
        # Async sending support. Queue size 4 absorbs short scheduling
        # hiccups (Windows timer jitter, browser keystroke contention)
        # without snowballing into stale lag — at 40 FPS that's 100ms
        # of buffering before frames get dropped. Smaller numbers (the
        # original 2) caused visible LED flicker any time the worker
        # paused for a single frame.
        self._send_queue = queue.Queue(maxsize=4)
        self._send_thread = None
        self._stop_thread = False
        self._async_enabled = False
        # Counters for diagnostics — printed periodically by the worker.
        self._frames_sent = 0
        self._frames_errored = 0

    def enable_async_send(self):
        """Enable asynchronous sending in a background thread"""
        if self._async_enabled:
            return
        
        self._async_enabled = True
        self._stop_thread = False
        self._send_thread = threading.Thread(target=self._send_worker, daemon=True)
        self._send_thread.start()
    
    def disable_async_send(self):
        """Disable asynchronous sending and wait for thread to finish"""
        if not self._async_enabled:
            return
        
        self._async_enabled = False
        self._stop_thread = True
        if self._send_thread:
            self._send_thread.join(timeout=1.0)
    
    def _send_worker(self):
        """Background worker that drains the send queue.

        Originally only caught ``queue.Empty``; any other exception in
        ``_send_immediate`` (transient network error, malformed frame,
        Numba compile bailout, etc.) killed the daemon thread silently
        and the LED output went dark / flickery while ``send()`` kept
        queueing frames into a queue with no consumer.

        Now: catch every exception, log it once per minute (so a
        sustained network outage doesn't spam), and keep going. The
        thread only stops on the explicit ``_stop_thread`` flag or a
        ``None`` poison pill.
        """
        last_error_log = 0.0
        while not self._stop_thread:
            now = time.monotonic()
            try:
                frame_data = self._send_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame_data is None:
                break
            try:
                self._send_immediate(frame_data)
                self._frames_sent += 1
            except Exception as e:
                self._frames_errored += 1
                # Fast initial errors so a regression shows up immediately;
                # then throttle to once per 5s if errors persist.
                if now - last_error_log > 5.0:
                    last_error_log = now
                    print(f"[DMXSender] worker exception "
                          f"(errored={self._frames_errored}, sent={self._frames_sent}): {e}")
                    traceback.print_exc()
    
    def _build_sacn_header(self, universe):
        """Build a pre-formatted sACN E1.31 packet header (126 bytes)"""
        header = bytearray(126)
        
        # Root Layer
        header[0:2] = b'\x00\x10'  # Preamble Size
        header[2:4] = b'\x00\x00'  # Post-amble Size
        header[4:16] = b'ASC-E1.17\x00\x00\x00'  # ACN Packet Identifier
        header[16:18] = struct.pack('>H', 0x7000 | (638 & 0x0FFF))  # Flags and Length
        header[18:22] = struct.pack('>I', 0x00000004)  # Vector
        header[22:38] = b'\x00' * 16  # CID (sender ID)
        
        # Framing Layer
        header[38:40] = struct.pack('>H', 0x7000 | (638 - 38))  # Flags and Length
        header[40:44] = struct.pack('>I', 0x00000002)  # Vector
        header[44:108] = b'Raw UDP sACN\x00' + b'\x00' * 51  # Source Name (64 bytes)
        header[108] = 100  # Priority
        header[109:111] = b'\x00\x00'  # Sync Address
        header[111] = 0  # Sequence Number (updated per packet)
        header[112] = 0  # Options
        header[113:115] = struct.pack('>H', universe)  # Universe
        
        # DMP Layer
        header[115:117] = struct.pack('>H', 0x7000 | (638 - 115))  # Flags and Length
        header[117] = 0x02  # Vector
        header[118] = 0xa1  # Address Type & Data Type
        header[119:121] = b'\x00\x00'  # First Property Address
        header[121:123] = b'\x00\x01'  # Address Increment
        header[123:125] = struct.pack('>H', 513)  # Property count (1 + 512)
        header[125] = 0  # DMX START code
        
        return bytes(header)
    
    # ---- DDP --------------------------------------------------------------
    # DDP_MAX_PAYLOAD: 1440 keeps the resulting UDP datagram under a typical
    # 1500-byte MTU (10-byte DDP header + 8-byte UDP + 20-byte IP + Ethernet
    # framing). Bumping this just causes IP fragmentation on the wire which
    # defeats the whole point of the protocol switch.
    DDP_MAX_PAYLOAD = 1440
    DDP_PORT = 4048

    def _send_ddp_receiver(self, rx_idx, ip):
        """Send the receiver's full pixel buffer as a sequence of DDP packets.
        Each packet carries up to DDP_MAX_PAYLOAD bytes at a known byte offset
        into the destination's pixel data; the last packet has the PUSH flag
        set so the receiver knows it's safe to display."""
        pixel_buffer = self._receiver_buffers[rx_idx]
        total_bytes = len(pixel_buffer)
        seq = self._ddp_sequence[rx_idx]

        offset = 0
        while offset < total_bytes:
            chunk_size = min(self.DDP_MAX_PAYLOAD, total_bytes - offset)
            is_last = (offset + chunk_size) >= total_bytes

            # 10-byte DDP header (big-endian)
            #   byte 0 (flags1): version 01 in bits 7-6, push in bit 0
            #   byte 1 (flags2/seq): low nibble = sequence
            #   byte 2 (data type): 0x01 = RGB
            #   byte 3 (destination id): 1 = default output
            #   bytes 4-7: 32-bit byte offset
            #   bytes 8-9: 16-bit payload length
            flags1 = 0x40 | (0x01 if is_last else 0x00)
            header = struct.pack('>BBBBIH', flags1, seq & 0x0f, 0x01, 1,
                                 offset, chunk_size)
            packet = header + bytes(pixel_buffer[offset:offset + chunk_size])

            try:
                self.udp_socket.sendto(packet, (ip, self.DDP_PORT))
            except OSError:
                # BlockingIOError = send buffer full;
                # gaierror / ENETUNREACH = bad address or unreachable host.
                # Either way one frame to one receiver is dropped and the
                # render loop keeps going — alternative is a crash on the
                # first bogus IP in config.yaml.
                pass

            offset += chunk_size

        # Cycle 1..15; spec reserves 0 as "no sequence / don't validate".
        self._ddp_sequence[rx_idx] = (seq % 15) + 1

    def _send_udp_universes(self, rx_idx, ip, universes):
        """Send universe data via raw UDP sockets"""
        universe_buffer_2d = self._universe_buffers[rx_idx]
        headers = self._sacn_headers[rx_idx]
        sequences = self._sequence_numbers[rx_idx]
        
        for u, universe in enumerate(universes):
            # Update sequence number in header (byte 111)
            seq = sequences[u]
            header = bytearray(headers[u])
            header[111] = seq
            
            # Construct packet: header + DMX data (510 bytes)
            packet = header + universe_buffer_2d[u, :510].tobytes()
            
            # Send via UDP to port 5568 (sACN)
            try:
                self.udp_socket.sendto(packet, (ip, 5568))
            except OSError:
                # See _send_ddp_receiver for the rationale — drop the
                # packet rather than letting one bad receiver crash the
                # render loop.
                pass
            
            # Increment sequence number (0-255)
            sequences[u] = (seq + 1) % 256
    
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

    def send(self, frames, verify=False):
        """Send pixel data to all configured receivers.

        ``frames`` is a dict ``{group_id: source_array}`` (Phase 6+
        multi-group form). Single-source callers may pass a bare ndarray;
        it's treated as the ``'main'`` group canvas for back-compat.

        If async is enabled, queues data for background sending.
        Otherwise sends immediately.
        """
        if self._async_enabled:
            # Drop frame if queue is full (prevents lag buildup)
            # Note: We don't copy arrays — caller must not modify after calling send()
            try:
                self._send_queue.put_nowait((frames, verify))
            except queue.Full:
                pass  # Skip this frame if queue is full
        else:
            self._send_immediate((frames, verify))

    def _send_immediate(self, frame_data):
        """Send pixel data using optimized Numba functions (releases GIL).

        :param frame_data: Tuple of (frames, verify). ``frames`` is either
            a dict ``{group_id: ndarray}`` or a bare ndarray (legacy).
        """
        frames, verify = frame_data

        if not isinstance(frames, dict):
            # Legacy single-source callers — wrap as a one-group dict.
            frames = {'main': frames}

        # Process each receiver using optimized Numba (releases GIL)
        for rx_idx, (receiver, universes) in enumerate(zip(self.receivers, self.receiver_universes)):
            pixel_buffer = self._receiver_buffers[rx_idx]
            protocol = self.receiver_protocols[rx_idx]

            # ---- per-strip pixel extraction (each strip pulls from its
            # own group canvas; receiver buffer is the concatenation) ----
            for plan in self._strip_plans[rx_idx]:
                source = frames.get(plan['group_id'])
                if source is None:
                    # Group not in this frame; leave the prior strip data in
                    # the buffer rather than blanking it.
                    continue
                buf_slice = pixel_buffer[plan['byte_offset']:
                                         plan['byte_offset'] + plan['byte_count']]
                if NUMBA_AVAILABLE:
                    extract_and_pack_pixels_unchecked(
                        source, plan['x_coords'], plan['y_coords'], buf_slice,
                    )
                else:
                    h, w = source.shape[:2]
                    x_valid = np.minimum(plan['x_coords'], h - 1)
                    y_valid = np.minimum(plan['y_coords'], w - 1)
                    pixels = source[x_valid, y_valid]
                    buf_slice[:] = pixels.flatten()

            if self.skip_network:
                continue

            # ---- protocol-specific send ----
            if protocol == 'ddp':
                # DDP: ship pixel_buffer directly as ~1440-byte chunks. No
                # universe packing needed — far fewer packets per frame
                # avoids the ETH RX cliff that bites sACN at ~6 packets.
                self._send_ddp_receiver(rx_idx, receiver['ip'])
                continue

            # ---- sACN packing + send ----
            if NUMBA_AVAILABLE:
                # Process all universes in parallel
                universe_buffer_2d = self._universe_buffers[rx_idx]
                process_all_universes(
                    pixel_buffer,
                    self._universe_starts[rx_idx],
                    self._universe_ends[rx_idx],
                    universe_buffer_2d
                )

                if self.use_raw_udp:
                    self._send_udp_universes(rx_idx, receiver['ip'], universes)
                else:
                    memviews = self._universe_memviews[rx_idx]
                    for u, universe in enumerate(universes):
                        self.sender[universe].dmx_data = memviews[u]

            else:
                # Pack into universe buffers (numpy fallback)
                slices = self._universe_slices[rx_idx]
                universe_buffer_2d = self._universe_buffers[rx_idx]

                for u, (start, end, needs_padding) in enumerate(slices):
                    byte_start = start * 3
                    byte_end = end * 3
                    byte_count = byte_end - byte_start

                    universe_buffer_2d[u, :byte_count] = pixel_buffer[byte_start:byte_end]
                    if needs_padding:
                        universe_buffer_2d[u, byte_count:] = 0

                if self.use_raw_udp:
                    self._send_udp_universes(rx_idx, receiver['ip'], universes)
                else:
                    memviews = self._universe_memviews[rx_idx]
                    for u, (start, end, needs_padding) in enumerate(slices):
                        self.sender[universes[u]].dmx_data = memviews[u]
        
        # Flush all universes
        if not self.skip_network and not self.use_raw_udp:
            self.sender.flush()

    def close(self):
        """
        Properly close the sACN sender
        """
        # Stop async thread if running
        self.disable_async_send()
        # The UDP socket is created either when use_raw_udp is set OR when any
        # receiver speaks DDP; the sacn library is only used when neither
        # condition holds.
        if hasattr(self, 'udp_socket') and self.udp_socket is not None:
            self.udp_socket.close()
        if hasattr(self, 'sender') and self.sender is not None:
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
            for y in range(height-1, -1, -1):
                indices.append([y, x+start])
        else:
            for y in range(height):
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