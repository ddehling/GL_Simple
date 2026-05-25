# GL_Simple all-in-one setup for Windows.
#
# Run from the repo root after `git clone`. Does everything needed to
# get from a freshly cloned engine to a running app:
#
#   1. Installs git + gh (GitHub CLI) via winget if missing.
#   2. Signs you into GitHub via gh's browser-based device flow if
#      not already signed in. gh wires itself as git's credential
#      helper, so private project repos clone without any PAT pasting.
#   3. Reads deploy/catalog.yaml, probes each project repo to mark it
#      [ACCESSIBLE] or [NO ACCESS] under your GitHub auth.
#   4. Prompts you to pick which project(s) to install and which is
#      the primary (the one that runs on launch).
#   5. Clones the selected project repos into projects/<id>/.
#   6. Writes the primary id to config.yaml's `project:` field.
#   7. Installs Python via winget if missing.
#   8. Creates .\venv and installs requirements.txt.
#   9. Launches Stories_OGL.py.
#
# Re-running is fine - idempotent (skips installed bits and existing
# clones).
#
# Cross-platform: the Linux equivalent is bin/setup.sh - they share
# behaviour and the same catalog.

Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  GL_Simple Setup (Windows)" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

function Update-PathFromMachineAndUser {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

function Test-Command($name) {
    try { & $name --version 2>&1 | Out-Null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}

function Assert-Winget {
    if (-not (Test-Command winget)) {
        Write-Host "ERROR: winget not available. Install App Installer from the Microsoft Store, then re-run." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 1
    }
}

# ---------------------------------------------------------------------
# 1. git
# ---------------------------------------------------------------------
if (-not (Test-Command git)) {
    Assert-Winget
    Write-Host "[1/8] Installing git via winget..." -ForegroundColor Yellow
    & winget install --id Git.Git -e --silent --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: git install failed. Install from https://git-scm.com/download/win and re-run." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 1
    }
    Update-PathFromMachineAndUser
} else {
    Write-Host "[1/8] git OK ($(& git --version))" -ForegroundColor Green
}

# ---------------------------------------------------------------------
# 2. gh + auth
# ---------------------------------------------------------------------
if (-not (Test-Command gh)) {
    Assert-Winget
    Write-Host "[2/8] Installing GitHub CLI (gh) via winget..." -ForegroundColor Yellow
    & winget install --id GitHub.cli -e --silent --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: gh install failed. Install from https://cli.github.com/ and re-run." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 1
    }
    Update-PathFromMachineAndUser
} else {
    Write-Host "[2/8] gh OK ($(& gh --version | Select-Object -First 1))" -ForegroundColor Green
}

& gh auth status 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "      Not signed in to GitHub yet." -ForegroundColor Yellow
    Write-Host "      gh will print a URL and a one-time code. Open the URL" -ForegroundColor Cyan
    Write-Host "      on any device, enter the code, sign in, and grant access." -ForegroundColor Cyan
    Write-Host "      The script will resume automatically." -ForegroundColor Cyan
    Write-Host ""
    & gh auth login --hostname github.com --web
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: gh auth login failed." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 1
    }
}
$ghUser = (& gh api user -q .login 2>$null)
if (-not $ghUser) { $ghUser = "?" }
Write-Host "      Signed in to GitHub as: $ghUser" -ForegroundColor Green
& gh auth setup-git 2>&1 | Out-Null

# ---------------------------------------------------------------------
# 3. Load catalog
# ---------------------------------------------------------------------
$catalogPath = "deploy/catalog.yaml"
if (-not (Test-Path $catalogPath)) {
    Write-Host "ERROR: catalog file not found at $catalogPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}

$projects = @()
$current = $null
foreach ($line in (Get-Content $catalogPath)) {
    if ($line -match '^  ([A-Za-z_][A-Za-z0-9_]*):\s*$') {
        if ($current) { $projects += $current }
        $current = @{ Id = $Matches[1]; Url = ""; Name = "" }
    } elseif ($null -ne $current) {
        if ($line -match '^    repo:\s*(.+?)\s*$') {
            $current.Url = $Matches[1]
        } elseif ($line -match '^    display_name:\s*(.+?)\s*$') {
            $current.Name = ($Matches[1] -replace '^"','' -replace '"$','')
        }
    }
}
if ($current) { $projects += $current }
if ($projects.Count -eq 0) {
    Write-Host "ERROR: no projects found in $catalogPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}
Write-Host "[3/8] Loaded $($projects.Count) projects from $catalogPath" -ForegroundColor Green

# ---------------------------------------------------------------------
# 4. Probe access
# ---------------------------------------------------------------------
Write-Host "[4/8] Checking access to each project repo..." -ForegroundColor Yellow
foreach ($p in $projects) {
    if (-not $p.Url) { $p.Access = "NO_URL"; continue }
    $env:GIT_TERMINAL_PROMPT = "0"
    & git ls-remote $p.Url HEAD 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $p.Access = "OK" } else { $p.Access = "DENIED" }
}

Write-Host ""
Write-Host "Available projects:" -ForegroundColor Cyan
Write-Host ""
for ($i = 0; $i -lt $projects.Count; $i++) {
    $p = $projects[$i]
    $name = if ($p.Name) { $p.Name } else { $p.Id }
    switch ($p.Access) {
        "OK"     { $status = "[ACCESSIBLE]"; $color = "Green" }
        "DENIED" { $status = "[NO ACCESS] "; $color = "Yellow" }
        default  { $status = "[NO URL]    "; $color = "Yellow" }
    }
    Write-Host (("  {0}) {1,-22} {2}  {3}" -f ($i + 1), $p.Id, $status, $name)) -ForegroundColor $color
}
Write-Host ""
Write-Host "  NO ACCESS = your GitHub account doesn't have access to that repo." -ForegroundColor DarkGray
Write-Host "  Ask the project owner to add you, then re-run this script." -ForegroundColor DarkGray
Write-Host ""

# ---------------------------------------------------------------------
# 5. Pick projects + primary
# ---------------------------------------------------------------------
$chosenIdx = @()
while ($chosenIdx.Count -eq 0) {
    $selection = Read-Host "Enter project numbers to install (space-separated, e.g. '1 2')"
    $chosenIdx = @()
    $valid = $true
    foreach ($tok in ($selection -split '\s+' | Where-Object { $_ -ne "" })) {
        if ($tok -notmatch '^\d+$') {
            Write-Host "  '$tok' is not a number." -ForegroundColor Red
            $valid = $false; break
        }
        $idx = [int]$tok - 1
        if ($idx -lt 0 -or $idx -ge $projects.Count) {
            Write-Host "  $tok is out of range." -ForegroundColor Red
            $valid = $false; break
        }
        if ($projects[$idx].Access -ne "OK") {
            Write-Host "  $tok ($($projects[$idx].Id)) is not accessible." -ForegroundColor Red
            $valid = $false; break
        }
        $chosenIdx += $idx
    }
    if (-not $valid) { $chosenIdx = @() }
}

$chosenIds = $chosenIdx | ForEach-Object { $projects[$_].Id }
Write-Host "  Will install: $($chosenIds -join ', ')" -ForegroundColor Green

$primaryId = $chosenIds[0]
if ($chosenIds.Count -gt 1) {
    Write-Host ""
    Write-Host "Which project is the primary (runs on app launch)?" -ForegroundColor Cyan
    for ($i = 0; $i -lt $chosenIds.Count; $i++) {
        Write-Host (("  {0}) {1}" -f ($i + 1), $chosenIds[$i]))
    }
    while ($true) {
        $primaryInput = Read-Host "Primary [1]"
        if (-not $primaryInput) { $primaryInput = "1" }
        if ($primaryInput -match '^\d+$' -and [int]$primaryInput -ge 1 -and [int]$primaryInput -le $chosenIds.Count) {
            $primaryId = $chosenIds[[int]$primaryInput - 1]
            break
        }
        Write-Host "  Invalid choice." -ForegroundColor Red
    }
}
Write-Host "  Primary: $primaryId" -ForegroundColor Green

# ---------------------------------------------------------------------
# 6. Clone selected
# ---------------------------------------------------------------------
Write-Host ""
Write-Host "[5/8] Cloning selected projects..." -ForegroundColor Yellow
foreach ($i in $chosenIdx) {
    $p = $projects[$i]
    $dest = "projects/$($p.Id)"
    if (Test-Path "$dest\.git") {
        Write-Host "      $($p.Id) already deployed at $dest (skipping)" -ForegroundColor DarkGray
        continue
    }
    Write-Host "      cloning $($p.Id) from $($p.Url)" -ForegroundColor Cyan
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    & git clone $p.Url $dest
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: clone of $($p.Id) failed." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 1
    }
}

# ---------------------------------------------------------------------
# 7. Write config.yaml's project: field
# ---------------------------------------------------------------------
Write-Host "[6/8] Setting active project to '$primaryId' in config.yaml..." -ForegroundColor Yellow
if (-not (Test-Path "config.yaml")) {
    Write-Host "ERROR: config.yaml not found at repo root." -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}
$configLines = Get-Content "config.yaml"
$replaced = $false
$newLines = foreach ($line in $configLines) {
    if ($line -match '^project:') { $replaced = $true; "project: $primaryId" } else { $line }
}
if (-not $replaced) { $newLines += "project: $primaryId" }
Set-Content "config.yaml" -Value $newLines -Encoding ASCII

# ---------------------------------------------------------------------
# 8. Python + venv + pip install
# ---------------------------------------------------------------------
Write-Host "[7/8] Installing Python dependencies..." -ForegroundColor Yellow

# Find a Python 3.10+ - prefer 3.12 via py launcher (3.13 drops distutils).
$pythonCmd = $null
$pyArgs = @()
try {
    $v = & py -3.12 --version 2>&1 | Out-String
    if ($v -match 'Python 3\.12') { $pythonCmd = "py"; $pyArgs = @("-3.12") }
} catch {}
if (-not $pythonCmd) {
    foreach ($cmd in @("python","python3","py")) {
        try {
            $v = & $cmd --version 2>&1 | Out-String
            if ($v -match 'Python 3\.(1[0-2])' -and $v -notlike '*Microsoft Store*') {
                $pythonCmd = $cmd; break
            }
        } catch {}
    }
}
if (-not $pythonCmd) {
    Assert-Winget
    Write-Host "      Installing Python 3.12 via winget..." -ForegroundColor Yellow
    & winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
    Update-PathFromMachineAndUser
    Write-Host "      Python installed. Please close this terminal, reopen, and re-run setup.ps1." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"; exit 0
}

if (-not (Test-Path ".\venv")) {
    & $pythonCmd @pyArgs -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: venv creation failed." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 1
    }
}
& .\venv\Scripts\Activate.ps1
& python -m pip install --upgrade pip setuptools wheel --quiet
& pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "      pip install reported errors; retrying with --no-cache-dir..." -ForegroundColor Yellow
    & pip install -r requirements.txt --no-cache-dir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: pip install failed." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 1
    }
}

# ---------------------------------------------------------------------
# 9. Launch
# ---------------------------------------------------------------------
Write-Host ""
Write-Host "[8/8] Launching application..." -ForegroundColor Green
Write-Host "      Web control panel: http://localhost:5000" -ForegroundColor Cyan
Write-Host "      Press Ctrl+C to stop." -ForegroundColor Cyan
Write-Host ""
try { & python Stories_OGL.py } catch { Write-Host "Application terminated" -ForegroundColor Yellow }
