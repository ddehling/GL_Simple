@echo off
cd /d "%~dp0.."
REM Quick Run Script - assumes setup has already been completed
REM Use this after running setup_and_run.bat at least once

echo Starting GL_Simple...
echo.
echo Web control panel: http://localhost:5000
echo Press Ctrl+C to stop
echo.

REM Activate virtual environment and run
call venv\Scripts\activate.bat
python Stories_OGL.py

pause
