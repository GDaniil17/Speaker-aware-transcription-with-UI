# One-command start for Windows (PowerShell 5.1+). Usually started via run.bat.
# uv, Python and ffmpeg are fetched into this folder; nothing is installed
# system-wide and no administrator rights are needed.
#
#   .\run.bat            Quick start: Whisper on the CPU, no account or token needed.
#   .\run.bat --full     Everything: speaker identification, all ASR engines, CLI.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$full = $false
foreach ($arg in $args) {
    switch -Regex ($arg) {
        '^(--full|-full|/full)$' { $full = $true }
        '^(--cpu|-cpu|/cpu)$' { }  # the quick start on Windows always uses the CPU
        '^(--help|-help|-h|/\?)$' {
            Write-Host 'Usage: run.bat [--full]'
            Write-Host ''
            Write-Host '  (no option)  Quick start: Whisper transcription on the CPU, no token needed.'
            Write-Host '  --full       Everything: speaker identification (pyannote), all ASR'
            Write-Host '               engines and the transcribe CLI. Downloads PyTorch (a few GB).'
            exit 0
        }
        default {
            Write-Host "Unknown argument: $arg. Use --help for usage." -ForegroundColor Red
            exit 2
        }
    }
}

Set-Location -LiteralPath $PSScriptRoot
$projectDir = $PSScriptRoot
$runtimeDir = Join-Path $projectDir '.cache\quickstart'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
$env:UV_CACHE_DIR = Join-Path $runtimeDir 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $runtimeDir 'python'

function Fail([string]$message) {
    Write-Host $message -ForegroundColor Red
    Write-Host 'Setup stopped. Check the error above and your internet connection, then run the same command again.' -ForegroundColor Red
    exit 1
}

# Run a native command and stop when it fails (PowerShell 5.1 does not).
function Invoke-Checked([string]$exe, [string[]]$arguments, [string]$what) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit code $LASTEXITCODE)." }
}

# --- uv: reuse an installed one, else fetch it into .cache\quickstart ----------
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uv = $uvCommand.Source
} else {
    $uv = Join-Path $runtimeDir 'bin\uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        Write-Host 'Installing uv locally (no administrator rights or PATH changes)...'
        $env:UV_UNMANAGED_INSTALL = Join-Path $runtimeDir 'bin'
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1' | Invoke-Expression
        } catch {
            Fail "Could not install uv: $_"
        } finally {
            Remove-Item Env:UV_UNMANAGED_INSTALL -ErrorAction SilentlyContinue
        }
        if (-not (Test-Path -LiteralPath $uv)) { Fail 'uv was not installed.' }
    }
}

# A free local port: 8501 unless another program answers there.
function Get-FreePort([string]$python) {
    $code = @'
import socket

def busy(port):
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family) as sock:
                sock.settimeout(0.3)
                if sock.connect_ex((host, port)) == 0:
                    return True
        except OSError:
            pass
    return False

print(next(port for port in range(8501, 8600) if not busy(port)))
'@
    $port = (& $python -c $code).Trim()
    if ($LASTEXITCODE -ne 0) { Fail 'Could not find a free port.' }
    return $port
}

function Show-Address([string]$port) {
    if ($port -ne '8501') {
        Write-Host "Port 8501 is used by another program, so this app uses port $port."
    }
    Write-Host "Open http://127.0.0.1:$port if the browser does not open. Stop with Ctrl+C."
}

function Get-StreamlitArgs([string]$port) {
    return @('--server.address', '127.0.0.1', '--server.port', $port,
        '--server.headless', 'false', '--browser.gatherUsageStats', 'false')
}

# --- full setup: the locked project environment in .venv -----------------------
if ($full) {
    Write-Host 'Preparing the full environment (.venv): Whisper, speaker identification, all engines...'
    # Every engine: Parakeet and GigaAM are core, Qwen3-ASR and SenseVoice are
    # extras. --inexact keeps anything else installed earlier.
    Invoke-Checked $uv @('sync', '--inexact', '--extra', 'qwen3-asr', '--extra', 'sensevoice') 'Installing the project'
    if (-not (Test-Path -LiteralPath '.env')) {
        Copy-Item '.env.example' '.env'
        Write-Host 'Created .env. Speaker identification needs a Hugging Face token once: see README.md.'
    }
    $port = Get-FreePort (Join-Path $projectDir '.venv\Scripts\python.exe')
    Write-Host 'Starting the full app.'
    Show-Address $port
    & $uv run --no-sync streamlit run app.py @(Get-StreamlitArgs $port)
    exit $LASTEXITCODE
}

# --- quick start: only the Whisper backend, in .venv-whisper -------------------
$venvDir = Join-Path $projectDir '.venv-whisper'
$python = Join-Path $venvDir 'Scripts\python.exe'
$requirements = 'requirements\whisper.txt'

# Rebuild the venv when the requirement files change.
$setupId = (Get-Content 'requirements\ui.txt', $requirements -Raw) -join "`n"
$stamp = Join-Path $venvDir '.run-setup'
Write-Host 'Preparing Python 3.12 and the Whisper environment...'
$current = $null
if (Test-Path -LiteralPath $stamp) { $current = Get-Content -LiteralPath $stamp -Raw }
if (-not (Test-Path -LiteralPath $python) -or $current -ne $setupId) {
    if (Test-Path -LiteralPath $venvDir) { Remove-Item -Recurse -Force -LiteralPath $venvDir }
    Invoke-Checked $uv @('venv', '--python', '3.12', $venvDir) 'Creating the Python environment'
}
Invoke-Checked $uv @('pip', 'install', '--python', $python, '--only-binary', ':all:',
    '-r', $requirements) 'Installing Whisper'
Set-Content -LiteralPath $stamp -Value $setupId -NoNewline

$srcDir = Join-Path $projectDir 'src'
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$srcDir;$env:PYTHONPATH" } else { $env:PYTHONPATH = $srcDir }
$env:SPEAKER_TRANSCRIBE_BASIC = '1'
$env:SPEAKER_TRANSCRIBE_BACKEND = 'faster-whisper'

$port = Get-FreePort $python
Write-Host 'Starting Whisper large-v3-turbo (faster-whisper, CPU). No Hugging Face token needed.'
Show-Address $port
Write-Host 'The model downloads on your first transcription; keep this window open.'
& $python -m streamlit run app.py @(Get-StreamlitArgs $port)
exit $LASTEXITCODE
