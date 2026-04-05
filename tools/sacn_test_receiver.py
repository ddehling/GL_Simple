#!/usr/bin/env python3
"""
sACN/E1.31 Test Receiver
========================
Drop this on a Raspberry Pi (or any machine) to verify sACN packets
are arriving from GL_Simple over the network.

Setup on the Pi:
    # Set a static IP matching one of the GL_Simple receivers
    sudo ip addr add 192.168.68.140/24 dev eth0

    # Run this script (no dependencies beyond Python 3)
    python3 sacn_test_receiver.py

    # Or listen on a specific IP / all universes
    python3 sacn_test_receiver.py --bind 0.0.0.0 --universes 1-4

No external packages required — just Python 3 standard library.
"""

import argparse
import socket
import struct
import sys
import time
from collections import defaultdict

SACN_PORT = 5568
HEADER_SIZE = 126


def parse_sacn_packet(data):
    """Parse an E1.31 sACN packet, return dict or None if invalid."""
    if len(data) < HEADER_SIZE:
        return None

    # Verify ACN packet identifier
    acn_id = data[4:16]
    if acn_id != b'ASC-E1.17\x00\x00\x00':
        return None

    source_name = data[44:108].split(b'\x00')[0].decode('utf-8', errors='replace')
    priority = data[108]
    sequence = data[111]
    universe = struct.unpack('>H', data[113:115])[0]
    prop_count = struct.unpack('>H', data[123:125])[0]
    dmx_channel_count = prop_count - 1  # subtract START code

    dmx_data = data[HEADER_SIZE:] if len(data) > HEADER_SIZE else b''

    return {
        'source': source_name,
        'priority': priority,
        'sequence': sequence,
        'universe': universe,
        'channel_count': dmx_channel_count,
        'dmx_data': dmx_data,
    }


def format_dmx_preview(dmx_data, num_pixels=4):
    """Show first N pixels as RGB triplets."""
    pixels = []
    for i in range(min(num_pixels, len(dmx_data) // 3)):
        r, g, b = dmx_data[i*3], dmx_data[i*3+1], dmx_data[i*3+2]
        pixels.append(f"({r:3d},{g:3d},{b:3d})")
    return " ".join(pixels)


def color_block(dmx_data, num_pixels=8):
    """ANSI color blocks for a quick visual of pixel data."""
    blocks = []
    for i in range(min(num_pixels, len(dmx_data) // 3)):
        r, g, b = dmx_data[i*3], dmx_data[i*3+1], dmx_data[i*3+2]
        blocks.append(f"\033[48;2;{r};{g};{b}m  \033[0m")
    return "".join(blocks)


def main():
    parser = argparse.ArgumentParser(description="sACN/E1.31 Test Receiver")
    parser.add_argument("--bind", default="0.0.0.0",
                        help="IP address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=SACN_PORT,
                        help=f"UDP port (default: {SACN_PORT})")
    parser.add_argument("--universes", default=None,
                        help="Filter universes, e.g. '1-4' or '1,2,3' (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show every packet (default: summary mode)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI color output")
    args = parser.parse_args()

    # Parse universe filter
    universe_filter = None
    if args.universes:
        universe_filter = set()
        for part in args.universes.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                universe_filter.update(range(int(lo), int(hi) + 1))
            else:
                universe_filter.add(int(part))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind, args.port))

    print(f"sACN Test Receiver listening on {args.bind}:{args.port}")
    if universe_filter:
        print(f"Filtering universes: {sorted(universe_filter)}")
    print(f"Mode: {'verbose' if args.verbose else 'summary (updates every second)'}")
    print("-" * 60)

    # Stats for summary mode
    stats = defaultdict(lambda: {
        'packets': 0,
        'last_seq': -1,
        'drops': 0,
        'last_dmx': b'',
        'source': '',
        'first_seen': time.time(),
    })
    last_print = time.time()

    try:
        while True:
            data, addr = sock.recvfrom(2048)
            pkt = parse_sacn_packet(data)
            if pkt is None:
                continue

            uni = pkt['universe']
            if universe_filter and uni not in universe_filter:
                continue

            if args.verbose:
                preview = format_dmx_preview(pkt['dmx_data'])
                color = "" if args.no_color else f" {color_block(pkt['dmx_data'])}"
                print(f"[{addr[0]}] U:{uni:3d} seq:{pkt['sequence']:3d} "
                      f"ch:{pkt['channel_count']:3d} | {preview}{color}")
            else:
                # Accumulate stats
                s = stats[uni]
                s['packets'] += 1
                s['source'] = f"{addr[0]} ({pkt['source']})"
                if s['last_seq'] >= 0:
                    expected = (s['last_seq'] + 1) % 256
                    if pkt['sequence'] != expected:
                        s['drops'] += 1
                s['last_seq'] = pkt['sequence']
                s['last_dmx'] = pkt['dmx_data']

                # Print summary every second
                now = time.time()
                if now - last_print >= 1.0:
                    last_print = now
                    sys.stdout.write("\033[2J\033[H")  # clear screen
                    print(f"sACN Test Receiver — {args.bind}:{args.port}")
                    print(f"{'=' * 60}")
                    for u in sorted(stats.keys()):
                        s = stats[u]
                        elapsed = now - s['first_seen']
                        fps = s['packets'] / elapsed if elapsed > 0 else 0
                        color = "" if args.no_color else f" {color_block(s['last_dmx'])}"
                        print(f"  Universe {u:3d} | {s['source']}")
                        print(f"    {s['packets']:6d} pkts | {fps:5.1f} pkt/s | "
                              f"{s['drops']} seq drops{color}")
                        print(f"    First pixels: {format_dmx_preview(s['last_dmx'])}")
                        print()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
