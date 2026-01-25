# GL_Simple - Windows Setup Guide

This is an OpenGL-based DMX lighting control system with weather effects, shader rendering, audio analysis, and web control.

## Quick Start (Easiest Method)

1. **Double-click `setup_and_run.bat`**
   - This will automatically:
     - Check your Python installation
     - Create a virtual environment
     - Install all dependencies
     - Launch the application

2. **Access the Web Control Panel**
   - Open your browser to: `http://localhost:5000`
   - Control weather effects, shaders, and more!

## Alternative Launch Methods

### After First Setup
Once you've run the setup script once, you can use the quick launcher:
- **Double-click `quick_run.bat`** for faster startup

### Manual Launch
If you prefer manual control:
```batch
# Activate virtual environment
venv\Scripts\activate

# Run the application
python Stories_OGL.py
```

### PowerShell Direct
```powershell
# Run the setup script directly
powershell -ExecutionPolicy Bypass -File setup_and_run.ps1
```

## Requirements

- **Python 3.8+** (3.10 or higher recommended)
- **Windows 10/11**
- **OpenGL-compatible graphics card**
- **Audio input device** (optional, for audio-reactive effects)

### Installing Python
If you don't have Python installed:
1. Download from: https://www.python.org/downloads/
2. During installation, **check "Add Python to PATH"**
3. Restart your computer after installation

## What Gets Installed

The setup script installs these Python packages:
- **glfw** - Window management
- **PyOpenGL** - OpenGL bindings
- **numpy, scipy** - Math and scientific computing
- **opencv-python** - Image processing
- **sounddevice, librosa, soundfile** - Audio processing
- **Flask** - Web control interface
- **sacn** - DMX lighting control (sACN/E1.31)
- **pygame** - Audio playback
- **numba** - Performance optimization
- And more...

## Features

### Visual Effects
- 🌤️ Dynamic weather system (rain, fog, storms, aurora, etc.)
- ⭐ Celestial bodies (sun, moon, planets)
- 🌊 Ocean effects (waves, kelp, bioluminescence)
- 🔥 Special effects (lightning, meteors, fireflies)
- 🎨 40+ shader effects

### Control Systems
- 🌐 **Web Interface** at `http://localhost:5000`
  - Change weather sets
  - Trigger events
  - Adjust parameters
- 🎹 **MIDI Controller** support
- 🎤 **Audio-reactive** effects
- 📡 **DMX Output** via sACN/E1.31

### Weather Sets
- **Forest** - Woodland environment with seasonal changes
- **Desert** - Sand storms and desert effects
- **Ocean** - Underwater and beach scenes
- **Spooky** - Halloween-themed effects
- **Mountain** - Alpine environment
- **Cyberpunk** - Futuristic urban effects

## Troubleshooting

### "Python not found"
- Make sure Python is installed and added to PATH
- Restart your terminal/command prompt after installing Python
- Try running: `python --version` to verify installation

### "Module not found" errors
- Run the setup script again: `setup_and_run.bat`
- Or manually install: `pip install -r info\requirements.txt`

### Performance Issues
- Close other GPU-intensive applications
- Reduce the magnification setting in `Stories_OGL.py`
- Disable the web control panel (set `enable_web_control = False`)

### OpenGL Errors
- Update your graphics drivers
- Make sure your GPU supports OpenGL 3.3+

### Audio Device Not Found
- Check device name in `Stories_OGL.py` (line 40)
- Run `sound_editor.py` to list available audio devices
- Change or remove the device_name parameter

### Port Already in Use (Web Control)
- Close other applications using port 5000
- Or change the port in `Stories_OGL.py` (line 57)

## Configuration

### Main Application Settings
Edit `Stories_OGL.py`:
- **Line 22-25**: Frame dimensions and magnification
- **Line 40**: Audio input device name
- **Line 49**: Enable/disable web control
- **Line 57**: Web control port
- **Line 639**: Initial weather set
- **Line 640**: Initial weather state

### DMX Configuration
Edit files in `DMXconfig/` folder to configure your DMX universes and fixtures.

### Web Control Password
Change the admin password in `Stories_OGL.py` line 59:
```python
admin_password="your_secure_password"
```

## Project Structure

```
GL_Simple/
├── Stories_OGL.py          # Main application
├── sound_editor.py         # Audio device configuration tool
├── computer.py             # Alternative shader testing app
├── setup_and_run.bat       # Automated setup script (RECOMMENDED)
├── setup_and_run.ps1       # PowerShell setup script
├── quick_run.bat           # Quick launcher (after setup)
├── corefunctions/          # Core system modules
│   ├── shader_renderer.py
│   ├── soundinput.py
│   ├── web_controller.py
│   └── shader_effects/     # 40+ shader effects
├── templates/              # Web interface HTML
├── DMXconfig/              # DMX fixture configurations
├── media/                  # Images and sounds
└── info/                   # Documentation
    └── requirements.txt    # Python dependencies
```

## Support

For more information, see the documentation files in the `info/` directory:
- `WEATHER_EDITOR_README.md` - Weather system guide
- `WEB_CONTROL_README.md` - Web interface documentation
- `MIDI_README.md` - MIDI controller setup
- `EVENT_MAP_SYSTEM.md` - Event scheduling system

## License

Check the repository for license information.

---

**Enjoy your lighting control system!** 🎭✨
