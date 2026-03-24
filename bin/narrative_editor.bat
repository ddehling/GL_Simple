@echo off
cd /d "%~dp0.."

REM Activate venv
call venv\Scripts\activate.bat

REM Install narrative editor deps if missing
python -c "import dearpygui" 2>nul || (
    echo Installing dearpygui...
    pip install dearpygui
)

REM Launch editor (optional: pass a script.json as argument)
if "%~1"=="" (
    python tools\narrative_editor.py
) else (
    python tools\narrative_editor.py "%~1"
)

pause
