import socket
import json
import time
from zeroconf import ServiceBrowser, Zeroconf

class WLEDListener:
    def __init__(self):
        self.devices = set()

    def remove_service(self, zeroconf, type, name):
        pass

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info and info.addresses:
            address = socket.inet_ntoa(info.addresses[0])
            port = info.port
            if (address, port) not in self.devices:
                self.devices.add((address, port))
                print(f"Detected new WLED device at {address}:{port}")
                self.fetch_device_info(address, port)

    def fetch_device_info(self, address, port):
        try:
            with socket.create_connection((address, port), timeout=5) as sock:
                sock.sendall(f"GET /json/info HTTP/1.1\r\nHost: {address}\r\n\r\n".encode())
                response = sock.recv(4096).decode()
                json_start = response.find('{')
                if json_start != -1:
                    info = json.loads(response[json_start:])
                    print(f"  Name: {info.get('name', 'Unknown')}")
                    print(f"  Version: {info.get('ver', 'Unknown')}")
                    print(f"  MAC: {info.get('mac', 'Unknown')}")
                    print(f"  Free heap: {info.get('freeheap', 'Unknown')} bytes")
                    print(f"  Uptime: {info.get('uptime', 'Unknown')} seconds")
        except Exception as e:
            print(f"  Error fetching device info: {e}")
        print()

def enumerate_wled_devices():
    zeroconf = Zeroconf()
    listener = WLEDListener()
    browser = ServiceBrowser(zeroconf, "_wled._tcp.local.", listener)

    print("Searching for WLED devices (will stop after 10 seconds)...")
    start_time = time.time()

    try:
        while time.time() - start_time < 10:
            time.sleep(0.1)  # Small sleep to prevent busy waiting
    finally:
        zeroconf.close()

    print("\nSearch completed. Total WLED devices found:", len(listener.devices))

if __name__ == "__main__":
    enumerate_wled_devices()