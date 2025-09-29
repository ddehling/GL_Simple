#!/usr/bin/env python3
import time
import random
import sacn  # Import the module, not a specific class

# Configuration
TARGET_IP = "192.168.68.111"  # Change this to your device's IP address
UNIVERSE = 1                 # sACN universe to use
NUM_PIXELS = 10              # Number of pixels to control
FPS = 30                     # Frames per second
CHANNEL_COUNT = NUM_PIXELS * 3  # 3 channels per pixel (RGB)

def main():
    # Create a sender - the correct way to instantiate the sender
    sender = sacn.sACNsender()  # Note: it's sACNsender, not sACN
    
    # Setup the sender
    sender.start()
    
    # Set the name of the sender
    sender.source_name = "Python sACN Example"
    
    # Create a new output universe
    sender.activate_output(universe=UNIVERSE)
    
    # Set destination IP
    sender[UNIVERSE].destination = TARGET_IP  # Note: it's 'destination', not 'destinations'
    
    # Set up packet data
    sender[UNIVERSE].multicast = False  # Use unicast
    
    print(f"Sending data to {TARGET_IP} on universe {UNIVERSE}")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Create a byte array for all channels (RGB values for each pixel)
            dmx_data = bytearray(CHANNEL_COUNT)
            
            # Fill with some example data (random colors for each pixel)
            for pixel in range(NUM_PIXELS):
                # Calculate the starting index for this pixel
                idx = pixel * 3
                
                # Set random RGB values
                dmx_data[idx] = random.randint(0, 255)     # Red
                dmx_data[idx + 1] = random.randint(0, 255) # Green
                dmx_data[idx + 2] = random.randint(0, 255) # Blue
            
            # Send the data
            sender[UNIVERSE].dmx_data = dmx_data
            
            # Wait for next frame
            time.sleep(1/FPS)
            
    except KeyboardInterrupt:
        print("Stopping sender...")
    finally:
        # Stop the sender
        sender.stop()

if __name__ == "__main__":
    main()