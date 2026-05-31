# GL_Simple everyday launch (Windows).
#
# Pulls latest engine + all deployed projects from their remotes,
# then launches the app. Offline-tolerant: any pull that can't reach
# its remote (no DNS, no internet, auth failure, hung connection) is
# logged and skipped, and the app launches with whatever's on disk.
#
# Use bin\windows-install.ps1 for the first install. Use bin\windows-run.ps1 every
# subsequent launch.

Set-Location (Split-Path -Parent $PSScriptRoot)

# Aborts a stalled HTTPS pull in ~5s. Plus a 15s outer timeout per
# repo so a hung connection can't make the operator wait forever.
$env:GIT_TERMINAL_PROMPT = "0"
$env:GIT_HTTP_LOW_SPEED_LIMIT = "1000"
$env:GIT_HTTP_LOW_SPEED_TIME = "5"

function Invoke-Pull {
    param([string]$Label, [string]$Dir)
    if (-not (Test-Path "$Dir\.git")) {
        Write-Host "  ${Label}: not a git checkout, skipping" -ForegroundColor DarkGray
        return
    }
    $proc = Start-Process -FilePath git -ArgumentList "-C", $Dir, "pull", "--ff-only" `
                          -NoNewWindow -PassThru `
                          -RedirectStandardOutput "$env:TEMP\gl_pull_$PID.out" `
                          -RedirectStandardError "$env:TEMP\gl_pull_$PID.err"
    if (-not $proc.WaitForExit(15000)) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  ${Label}: pull timed out - keeping local state" -ForegroundColor Yellow
        return
    }
    $out = (Get-Content "$env:TEMP\gl_pull_$PID.out" -ErrorAction SilentlyContinue) -join "`n"
    $err = (Get-Content "$env:TEMP\gl_pull_$PID.err" -ErrorAction SilentlyContinue) -join "`n"
    Remove-Item "$env:TEMP\gl_pull_$PID.out","$env:TEMP\gl_pull_$PID.err" -ErrorAction SilentlyContinue
    if ($proc.ExitCode -eq 0) {
        if ($out) { ($out -split "`n") | ForEach-Object { Write-Host "  ${Label}: $_" -ForegroundColor DarkGray } }
    } else {
        Write-Host "  ${Label}: pull failed (offline / auth / conflict?) - keeping local state" -ForegroundColor Yellow
        if ($err) { Write-Host "      $err" -ForegroundColor DarkGray }
    }
}

Write-Host "[1/2] Pulling latest engine + deployed projects..." -ForegroundColor Cyan
Invoke-Pull "engine" "."
Get-ChildItem projects -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    if (Test-Path "$($_.FullName)\.git") {
        Invoke-Pull $_.Name $_.FullName
    }
}

Write-Host ""
Write-Host "[2/2] Launching application..." -ForegroundColor Cyan
if (-not (Test-Path ".\venv")) {
    Write-Host "ERROR: .\venv not found. Run bin\windows-install.ps1 first." -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}
& .\venv\Scripts\python.exe Stories_OGL.py
