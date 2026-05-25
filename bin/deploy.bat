@echo off
REM GL_Simple Deploy launcher for Windows (Batch wrapper that calls PowerShell)

echo =====================================
echo   GL_Simple Deploy
echo =====================================
echo.

where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell not found!
    echo Run bin\deploy.ps1 directly or install PowerShell.
    pause
    exit /b 1
)

powershell -ExecutionPolicy Bypass -File "%~dp0deploy.ps1"

if %ERRORLEVEL% NEQ 0 (
    pause
)
