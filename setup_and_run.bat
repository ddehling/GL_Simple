@echo off
REM GL_Simple Setup and Run Script for Windows (Batch version)
REM This is a simpler alternative that calls the PowerShell script

echo =====================================
echo   GL_Simple Setup ^& Launcher
echo =====================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell not found!
    echo Please run the setup_and_run.ps1 script directly or install PowerShell
    pause
    exit /b 1
)

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1"

REM Keep window open if there was an error
if %ERRORLEVEL% NEQ 0 (
    pause
)
