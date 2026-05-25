@echo off
REM GL_Simple Setup launcher for Windows (calls setup.ps1)

where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell not found. Run bin\setup.ps1 directly or install PowerShell.
    pause
    exit /b 1
)

powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

if %ERRORLEVEL% NEQ 0 (
    pause
)
