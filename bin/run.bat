@echo off
REM GL_Simple everyday launch (Windows). Wraps run.ps1.

where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell not found. Run bin\run.ps1 directly or install PowerShell.
    pause
    exit /b 1
)

powershell -ExecutionPolicy Bypass -File "%~dp0run.ps1"

if %ERRORLEVEL% NEQ 0 (
    pause
)
