# GL_Simple Setup and Run Script for Windows
# This script will:
# 1. Check if Python is installed
# 2. Create a virtual environment if it doesn't exist
# 3. Install all required dependencies
# 4. Run the application

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  GL_Simple Setup & Launcher" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Function to install Python using winget
function Install-Python {
    Write-Host "  Attempting to install Python automatically using winget..." -ForegroundColor Cyan
    Write-Host "  This may take a few minutes..." -ForegroundColor Yellow
    
    try {
        # Install Python 3.12 using winget
        $installResult = & winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements 2>&1
        
        if ($LASTEXITCODE -eq 0 -or $installResult -like "*Successfully installed*") {
            Write-Host "  Python installed successfully!" -ForegroundColor Green
            Write-Host "  Refreshing environment variables..." -ForegroundColor Yellow
            
            # Refresh PATH environment variable
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            
            Write-Host "  Please close and reopen this terminal, then run the script again." -ForegroundColor Cyan
            Read-Host "Press Enter to exit"
            exit 0
        } else {
            throw "Installation failed"
        }
    } catch {
        Write-Host "  Automatic installation failed." -ForegroundColor Red
        Write-Host "" -ForegroundColor Red
        Write-Host "  Please install Python manually:" -ForegroundColor Yellow
        Write-Host "    1. Go to: https://www.python.org/downloads/" -ForegroundColor White
        Write-Host "    2. Download Python 3.10 or higher" -ForegroundColor White
        Write-Host "    3. Run installer and CHECK 'Add Python to PATH'" -ForegroundColor White
        Write-Host "    4. Restart this script" -ForegroundColor White
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check if Python is installed
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow

# Try multiple Python commands and locations
$pythonCmd = $null
$pythonCmds = @("python", "python3", "py")

foreach ($cmd in $pythonCmds) {
    try {
        $testVersion = & $cmd --version 2>&1 | Out-String
        if ($testVersion -match "Python \d+\.\d+\.\d+" -and $testVersion -notlike "*Microsoft Store*") {
            $pythonCmd = $cmd
            $pythonVersion = $testVersion
            break
        }
    } catch {
        continue
    }
}

# If not found in PATH, try common installation locations
if (-not $pythonCmd) {
    $commonPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            try {
                $testVersion = & $path --version 2>&1 | Out-String
                if ($testVersion -match "Python \d+\.\d+\.\d+") {
                    $pythonCmd = $path
                    $pythonVersion = $testVersion
                    Write-Host "  Found Python at: $path" -ForegroundColor Cyan
                    break
                }
            } catch {
                continue
            }
        }
    }
}

try {
    # Check if this is the Windows Store redirect or not found
    if (-not $pythonCmd -or $pythonVersion -like "*Microsoft Store*" -or $pythonVersion -like "*was not found*") {
        Write-Host "  Python is not installed" -ForegroundColor Yellow
        Write-Host "" -ForegroundColor Yellow
        
        # Check if winget is available
        try {
            $wingetCheck = & winget --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  Found winget package manager - can install automatically!" -ForegroundColor Green
                $response = Read-Host "  Install Python automatically? (Y/n)"
                
                if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
                    Install-Python
                } else {
                    Write-Host "  Manual installation required:" -ForegroundColor Yellow
                    Write-Host "    Download from: https://www.python.org/downloads/" -ForegroundColor White
                    Read-Host "Press Enter to exit"
                    exit 1
                }
            } else {
                throw "winget not available"
            }
        } catch {
            Write-Host "  winget not available - manual installation required:" -ForegroundColor Yellow
            Write-Host "    1. Download from: https://www.python.org/downloads/" -ForegroundColor White
            Write-Host "    2. Check 'Add Python to PATH' during install" -ForegroundColor White
            Write-Host "    3. Restart this script" -ForegroundColor White
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
    
    Write-Host "  Found: $($pythonVersion.Trim())" -ForegroundColor Green
    Write-Host "  Using command: $pythonCmd" -ForegroundColor Cyan
} catch {
    Write-Host "  ERROR: Python not found!" -ForegroundColor Red
    Write-Host "  Install from: https://www.python.org/" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Store the Python command for later use
$global:PYTHON_CMD = $pythonCmd

# Check Python version
$versionString = $pythonVersion.Trim() -replace 'Python ', ''
try {
    $majorVersion = [int]($versionString.Split('.')[0])
    $minorVersion = [int]($versionString.Split('.')[1])

    if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 8)) {
        Write-Host "  ERROR: Python 3.8 or higher is required!" -ForegroundColor Red
        Write-Host "  Current version: $($pythonVersion.Trim())" -ForegroundColor Red
        Write-Host "  Please upgrade Python to 3.10+" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
} catch {
    Write-Host "  ERROR: Unable to parse Python version!" -ForegroundColor Red
    Write-Host "  Please ensure Python 3.8+ is properly installed" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if it doesn't exist
Write-Host ""
Write-Host "[2/5] Setting up virtual environment..." -ForegroundColor Yellow
$venvPath = ".\venv"
if (Test-Path $venvPath) {
    Write-Host "  Virtual environment already exists" -ForegroundColor Green
} else {
    Write-Host "  Creating new virtual environment..." -ForegroundColor Cyan
    & $global:PYTHON_CMD -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to create virtual environment!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "  Virtual environment created successfully" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "[3/5] Activating virtual environment..." -ForegroundColor Yellow
$activateScript = ".\venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "  Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Activation script not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Upgrade pip
Write-Host ""
Write-Host "[4/5] Upgrading pip..." -ForegroundColor Yellow
& $global:PYTHON_CMD -m pip install --upgrade pip --quiet
Write-Host "  pip upgraded successfully" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "[5/5] Installing dependencies..." -ForegroundColor Yellow
$requirementsPath = ".\info\requirements.txt"
if (Test-Path $requirementsPath) {
    Write-Host "  Installing packages from requirements.txt..." -ForegroundColor Cyan
    Write-Host "  This may take several minutes..." -ForegroundColor Yellow
    Write-Host ""
    
    & pip install -r $requirementsPath
    $installExitCode = $LASTEXITCODE
    
    Write-Host ""
    
    if ($installExitCode -ne 0) {
        Write-Host "  WARNING: Some packages may have failed to install" -ForegroundColor Yellow
        Write-Host "  Attempting to verify critical packages..." -ForegroundColor Yellow
        
        # Try to verify numpy is installed
        & $global:PYTHON_CMD -c "import numpy" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ERROR: Critical packages failed to install!" -ForegroundColor Red
            Write-Host "  Trying to install packages one more time..." -ForegroundColor Yellow
            & pip install -r $requirementsPath --no-cache-dir
            
            # Check again
            & $global:PYTHON_CMD -c "import numpy" 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  ERROR: Installation failed. Please check the errors above." -ForegroundColor Red
                Read-Host "Press Enter to exit"
                exit 1
            }
        }
        Write-Host "  Core packages verified successfully" -ForegroundColor Green
    } else {
        Write-Host "  All dependencies installed successfully" -ForegroundColor Green
    }
} else {
    Write-Host "  ERROR: requirements.txt not found at $requirementsPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Setup complete - Launch application
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting GL_Simple application..." -ForegroundColor Yellow
Write-Host "The application will open in OpenGL windows" -ForegroundColor Cyan
Write-Host "Web control panel will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

# Small delay before launching
Start-Sleep -Seconds 2

# Run the main application
try {
    & $global:PYTHON_CMD Stories_OGL.py
} catch {
    Write-Host ""
    Write-Host "Application terminated" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Goodbye!" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
