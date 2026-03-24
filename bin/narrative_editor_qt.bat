@echo off
cd /d "%~dp0.."

REM Activate venv
call venv\Scripts\activate.bat

REM Install deps if missing
python -c "import PySide6" 2>nul || (
    echo Installing PySide6...
    pip install PySide6
)
python -c "import qtpy" 2>nul || (
    echo Installing qtpy...
    pip install qtpy
)
python -c "import NodeGraphQt" 2>nul || (
    echo Installing NodeGraphQt...
    pip install NodeGraphQt
)

REM Launch editor (optional: pass a script.json as argument)
if "%~1"=="" (
    python tools\narrative_editor_qt.py
) else (
    python tools\narrative_editor_qt.py "%~1"
)

pause
