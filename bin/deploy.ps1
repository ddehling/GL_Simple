# GL_Simple Deploy: interactive multi-project deployment for Windows.
#
# On a fresh machine:
#   1. Ensures git is installed (winget on Windows).
#   2. Reads deploy/catalog.yaml for the list of available projects.
#   3. Probes each catalog entry to see which the current GitHub
#      credentials can access (via `git ls-remote`).
#   4. Prompts the operator to pick which project(s) to install and
#      which one is the primary (auto-runs on app launch).
#   5. Clones chosen projects into projects/<id>/.
#   6. Writes config.yaml's `project:` field to the primary.
#   7. Hands off to bin/setup_and_run.ps1.
#
# The Linux equivalent is bin/deploy.sh - keep their UX identical.

Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  GL_Simple Deploy (Windows)" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# --- 1. Ensure git is present ---
function Install-Git {
    Write-Host "  Installing Git via winget..." -ForegroundColor Cyan
    try {
        & winget install --id Git.Git -e --silent --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            Write-Host "  Git installed" -ForegroundColor Green
            return $true
        }
    } catch {}
    return $false
}

try { & git --version 2>&1 | Out-Null } catch { $LASTEXITCODE = 1 }
if ($LASTEXITCODE -ne 0) {
    Write-Host "[1/6] git not found." -ForegroundColor Yellow
    try {
        & winget --version 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "winget missing" }
    } catch {
        Write-Host "ERROR: winget not available. Install git manually from https://git-scm.com/download/win and re-run." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 1
    }
    $response = Read-Host "  Install Git automatically? (Y/n)"
    if ($response -eq "" -or $response -match "^[Yy]") {
        if (-not (Install-Git)) {
            Write-Host "ERROR: automatic Git install failed. Install from https://git-scm.com/download/win and re-run." -ForegroundColor Red
            Read-Host "Press Enter to exit"; exit 1
        }
    } else {
        Read-Host "Press Enter to exit"; exit 1
    }
} else {
    Write-Host "[1/6] git OK ($(& git --version))" -ForegroundColor Green
}

# --- 2. Load catalog ---
$catalogPath = "deploy/catalog.yaml"
if (-not (Test-Path $catalogPath)) {
    Write-Host "ERROR: catalog file not found at $catalogPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}

# Parse YAML by hand. Mirrors the bash parser in bin/deploy.sh.
# Format expected:
#   projects:
#     <id>:
#       repo: <url>
#       display_name: <name>
$projects = @()  # list of @{ Id; Url; Name }
$current = $null
foreach ($line in (Get-Content $catalogPath)) {
    if ($line -match '^  ([A-Za-z_][A-Za-z0-9_]*):\s*$') {
        if ($current) { $projects += $current }
        $current = @{ Id = $Matches[1]; Url = ""; Name = "" }
    } elseif ($current -ne $null) {
        if ($line -match '^    repo:\s*(.+?)\s*$') {
            $current.Url = $Matches[1]
        } elseif ($line -match '^    display_name:\s*(.+?)\s*$') {
            $name = $Matches[1] -replace '^"', '' -replace '"$', ''
            $current.Name = $name
        }
    }
}
if ($current) { $projects += $current }

if ($projects.Count -eq 0) {
    Write-Host "ERROR: no projects found in $catalogPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}

Write-Host "[2/6] Loaded $($projects.Count) projects from $catalogPath" -ForegroundColor Green

# --- 3. Probe access ---
Write-Host "[3/6] Checking access to each project repo..." -ForegroundColor Yellow
foreach ($p in $projects) {
    if (-not $p.Url) {
        $p.Access = "NO_URL"
        continue
    }
    $env:GIT_TERMINAL_PROMPT = "0"
    & git ls-remote $p.Url HEAD 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $p.Access = "OK"
    } else {
        $p.Access = "DENIED"
    }
}

# --- 4. Show picker ---
Write-Host ""
Write-Host "Available projects:" -ForegroundColor Cyan
Write-Host ""
for ($i = 0; $i -lt $projects.Count; $i++) {
    $num = $i + 1
    $p = $projects[$i]
    $name = if ($p.Name) { $p.Name } else { $p.Id }
    switch ($p.Access) {
        "OK"     { $status = "[ACCESSIBLE]"; $color = "Green" }
        "DENIED" { $status = "[NO ACCESS] "; $color = "Yellow" }
        default  { $status = "[NO URL]    "; $color = "Yellow" }
    }
    $line = ("  {0}) {1,-22} {2}  {3}" -f $num, $p.Id, $status, $name)
    Write-Host $line -ForegroundColor $color
}
Write-Host ""
Write-Host "  NO ACCESS = current GitHub credentials can't read that repo." -ForegroundColor DarkGray
Write-Host "  Set up a Personal Access Token or SSH key (see docs/DEPLOYMENT.md)" -ForegroundColor DarkGray
Write-Host "  then re-run this script." -ForegroundColor DarkGray
Write-Host ""

# --- 5. Read selection ---
$chosenIdx = @()
while ($chosenIdx.Count -eq 0) {
    $input = Read-Host "Enter project numbers to install (space-separated, e.g. '1 2')"
    $chosenIdx = @()
    $valid = $true
    foreach ($tok in ($input -split '\s+' | Where-Object { $_ -ne "" })) {
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
            Write-Host "  $tok ($($projects[$idx].Id)) is not accessible with current credentials." -ForegroundColor Red
            $valid = $false; break
        }
        $chosenIdx += $idx
    }
    if (-not $valid) { $chosenIdx = @() }
}

$chosenIds = $chosenIdx | ForEach-Object { $projects[$_].Id }
Write-Host "  Will install: $($chosenIds -join ', ')" -ForegroundColor Green

# --- 6. Choose primary if more than one ---
$primaryId = $chosenIds[0]
if ($chosenIds.Count -gt 1) {
    Write-Host ""
    Write-Host "Which project is the primary (runs on app launch)?" -ForegroundColor Cyan
    for ($i = 0; $i -lt $chosenIds.Count; $i++) {
        Write-Host ("  {0}) {1}" -f ($i + 1), $chosenIds[$i])
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

# --- 7. Clone selected projects ---
Write-Host ""
Write-Host "[4/6] Cloning selected projects..." -ForegroundColor Yellow
foreach ($i in $chosenIdx) {
    $p = $projects[$i]
    $dest = "projects/$($p.Id)"
    if (Test-Path "$dest\.git") {
        Write-Host "  - $($p.Id) already deployed at $dest (skipping)" -ForegroundColor DarkGray
        continue
    }
    Write-Host "  - cloning $($p.Id) from $($p.Url)" -ForegroundColor Cyan
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    & git clone $p.Url $dest
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: clone of $($p.Id) failed." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 1
    }
}

# --- 8. Write config.yaml's project: field ---
Write-Host ""
Write-Host "[5/6] Setting active project to '$primaryId' in config.yaml..." -ForegroundColor Yellow
if (-not (Test-Path "config.yaml")) {
    Write-Host "ERROR: config.yaml not found at repo root." -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}
$configLines = Get-Content "config.yaml"
$replaced = $false
$newLines = foreach ($line in $configLines) {
    if ($line -match '^project:') {
        $replaced = $true
        "project: $primaryId"
    } else {
        $line
    }
}
if (-not $replaced) {
    $newLines += "project: $primaryId"
}
Set-Content "config.yaml" -Value $newLines -Encoding ASCII
Write-Host "  Done." -ForegroundColor Green

# --- 9. Hand off ---
Write-Host ""
Write-Host "[6/6] Deployment complete." -ForegroundColor Green
Write-Host ""
Write-Host "Run the app with:    bin\setup_and_run.bat   (or bin\setup_and_run.ps1)" -ForegroundColor Cyan
Write-Host "Switch primary with: edit config.yaml's 'project:' field, then setup_and_run." -ForegroundColor Cyan
Write-Host ""
