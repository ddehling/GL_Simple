#!/usr/bin/env python3
import configparser
from pathlib import Path

# Test loading the config file
config = configparser.ConfigParser()
config_path = Path('config.ini')

if not config_path.exists():
    print(f"ERROR: {config_path} not found!")
    exit(1)

config.read(config_path)

print("✓ Config file loaded successfully!")
print(f"\nSections found: {', '.join(config.sections())}")
print("\n--- Configuration Values ---")
print(f"Show Rendering Window: {config.getboolean('Display', 'show_rendering_window')}")
print(f"Show FPS Counter: {config.getboolean('Display', 'show_fps')}")
print(f"Target FPS: {config.getint('Display', 'target_fps')}")
print(f"Magnification: {config.getint('Display', 'magnification')}")
print(f"Web Control Enabled: {config.getboolean('WebControl', 'enable_web_control')}")
print(f"Web Port: {config.getint('WebControl', 'web_port')}")
print(f"Microphone Device: {config.get('Audio', 'microphone_device')}")
print(f"Startup Weather Set: {config.get('Startup', 'startup_weather_set')}")
print(f"Startup Weather State: {config.get('Startup', 'startup_weather_state')}")
print("\n✓ All configuration options validated!")
