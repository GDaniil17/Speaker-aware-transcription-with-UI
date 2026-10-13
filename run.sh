#!/bin/bash
# One-command start for macOS and Linux. Needs only bash and curl (or wget):
# uv, Python and ffmpeg are fetched into this folder, nothing is installed
# system-wide. Works with the Bash 3.2 shipped with macOS.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./run.sh [--full] [--cpu]

Install what the app needs into this folder and open it in the browser.

  (no option)  Quick start: Whisper transcription, no account or token needed.
               Apple Silicon Macs (macOS 14+) use the GPU via MLX, other
               machines the CPU via faster-whisper.
  --full       Everything: speaker identification (pyannote), all ASR engines
               and the `transcribe` CLI. Downloads PyTorch (a few GB).
  --cpu        Quick start on the CPU even on Apple Silicon.
  -h, --help   Show this help.
EOF
}

full=0
cpu=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --full) full=1 ;;
        --cpu) cpu=1 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "Unknown argument: $1. Use --help for usage." >&2; exit 2 ;;
    esac
    shift
done
if [ "$full" -eq 1 ] && [ "$cpu" -eq 1 ]; then
    echo "--cpu applies to the quick start only; --full picks the device itself." >&2
    exit 2
fi
os="$(uname -s)"
case "$os" in
    Darwin | Linux) ;;
    *) echo "This launcher is for macOS and Linux. On Windows, run run.bat." >&2; exit 1 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")"
project_dir="$PWD"
runtime_dir="$project_dir/.cache/quickstart"
mkdir -p "$runtime_dir"
export UV_CACHE_DIR="$runtime_dir/uv-cache"
export UV_PYTHON_INSTALL_DIR="$runtime_dir/python"

trap 'echo "Setup stopped. Check the error above and your internet connection, then run the same command again." >&2' ERR

# --- uv: reuse an installed one, else fetch it into .cache/quickstart ----------
if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
else
    uv_bin="$runtime_dir/bin/uv"
    if [ ! -x "$uv_bin" ]; then
        echo "Installing uv locally (no sudo or shell profile changes)…"
        installer="$runtime_dir/install-uv.sh"
        if command -v curl >/dev/null 2>&1; then
            curl --fail --location --silent --show-error --retry 3 \
                https://astral.sh/uv/install.sh -o "$installer"
        elif command -v wget >/dev/null 2>&1; then
            wget --quiet -O "$installer" https://astral.sh/uv/install.sh
        else
            echo "Neither curl nor wget is installed. Install one of them and retry." >&2
            exit 1
        fi
        UV_UNMANAGED_INSTALL="$runtime_dir/bin" sh "$installer"
    fi
fi

# Print a free local port: 8501 unless another program answers there, over
# IPv4 or IPv6 ("localhost" may resolve to ::1, where another app on 8501
# would otherwise answer instead of this one).
free_port() {
    "$1" -c '
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
'
}

announce() {
    if [ "$1" != 8501 ]; then
        echo "Port 8501 is used by another program, so this app uses port $1."
    fi
    echo "Open http://127.0.0.1:$1 if the browser does not open. Stop with Ctrl+C."
}

streamlit_args() {
    echo "--server.address 127.0.0.1 --server.port $1 --server.headless false"
    echo "--browser.gatherUsageStats false"
}

# --- full setup: the locked project environment in .venv -----------------------
if [ "$full" -eq 1 ]; then
    echo "Preparing the full environment (.venv): Whisper, speaker identification, all engines…"
    # Every engine: Parakeet and GigaAM are core, Qwen3-ASR and SenseVoice are
    # extras. --inexact keeps anything else installed earlier (e.g. parakeet-mlx).
    "$uv_bin" sync --inexact --extra qwen3-asr --extra sensevoice
    if [ ! -f .env ]; then
        cp .env.example .env
        echo "Created .env. Speaker identification needs a Hugging Face token once: see README.md."
    fi
    port="$(free_port .venv/bin/python)"
    echo "Starting the full app."
    announce "$port"
    # shellcheck disable=SC2046  # the arguments are meant to split
    exec "$uv_bin" run --no-sync streamlit run app.py $(streamlit_args "$port")
fi

# --- quick start: only the Whisper backend, in .venv-whisper -------------------
venv_dir="$project_dir/.venv-whisper"
want_mlx=0
if [ "$cpu" -eq 0 ] && [ "$os" = Darwin ] && [ "$(uname -m)" = arm64 ] \
    && [ "$(sw_vers -productVersion | cut -d. -f1)" -ge 14 ]; then
    want_mlx=1
fi
if [ "$want_mlx" -eq 1 ]; then
    requirements=requirements/mlx.txt
else
    requirements=requirements/whisper.txt
fi

# Install only what the chosen backend needs. The venv is rebuilt when the
# backend or the requirement files change, so switching to/from --cpu leaves
# no unused packages behind.
setup_id="$(printf 'mlx=%s\n' "$want_mlx"; cat requirements/ui.txt "$requirements")"
stamp="$venv_dir/.run-setup"
echo "Preparing Python 3.12 and the Whisper environment…"
if [ ! -x "$venv_dir/bin/python" ] || [ "$(cat "$stamp" 2>/dev/null)" != "$setup_id" ]; then
    rm -rf "$venv_dir"
    "$uv_bin" venv --python 3.12 "$venv_dir"
fi
uv_install() {
    "$uv_bin" pip install --python "$venv_dir/bin/python" --only-binary :all: "$@"
}

backend=faster-whisper
if [ "$want_mlx" -eq 1 ]; then
    echo "Installing Apple Silicon GPU support (MLX)…"
    # mlx-whisper's torch requirement only converts checkpoints; skip it.
    if uv_install -r requirements/mlx.txt && uv_install --no-deps 'mlx-whisper>=0.4,<0.5'; then
        backend=mlx
    else
        echo "MLX installation failed; continuing with Whisper on the CPU."
        uv_install -r requirements/whisper.txt
    fi
else
    uv_install -r requirements/whisper.txt
fi
printf '%s' "$setup_id" > "$stamp"

export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export SPEAKER_TRANSCRIBE_BASIC=1
export SPEAKER_TRANSCRIBE_BACKEND="$backend"

port="$(free_port "$venv_dir/bin/python")"
echo "Starting Whisper large-v3-turbo ($backend). No Hugging Face token needed."
announce "$port"
echo "The model downloads on your first transcription; keep this terminal open."
# shellcheck disable=SC2046  # the arguments are meant to split
exec "$venv_dir/bin/python" -m streamlit run app.py $(streamlit_args "$port")
