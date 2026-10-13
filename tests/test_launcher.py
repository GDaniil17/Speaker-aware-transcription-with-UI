"""run.sh: first and repeated start-up, without downloads or opening a browser."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="run.sh is for macOS and Linux")


def executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\nset -eu\n" + body)
    path.chmod(0o755)


def make_machine(tmp_path: Path, system: str, arch: str, version: str, mlx_fails: bool) -> dict:
    """A project copy and a PATH with mock uv/curl/python and nothing else."""
    project = tmp_path / "Project with spaces"
    project.mkdir()
    shutil.copy(ROOT / "run.sh", project)
    shutil.copy(ROOT / ".env.example", project)
    shutil.copytree(ROOT / "requirements", project / "requirements")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Only expose shell utilities: simulate a machine without uv, Python or
    # ffmpeg even when the host running this test has all three installed.
    for command in ("mkdir", "cp", "cat", "cut", "dirname", "rm", "sh"):
        (bin_dir / command).symlink_to(shutil.which(command))
    log = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": str(bin_dir),
        "TEST_LOG": str(log),
        "TEST_BIN": str(bin_dir),
        "TEST_MLX_FAILS": str(int(mlx_fails)),
    }
    executable(bin_dir / "uname", f'[ "$1" = -s ] && echo {system} || echo {arch}')
    executable(bin_dir / "sw_vers", f"echo {version}")
    # A fresh machine has no uv. Return a local mock of its official installer.
    executable(
        bin_dir / "curl",
        """
echo installer-download >> "$TEST_LOG"
while [ "$1" != -o ]; do shift; done
cat > "$2" <<'INSTALLER'
mkdir -p "$UV_UNMANAGED_INSTALL"
cp "$TEST_BIN/mock-uv" "$UV_UNMANAGED_INSTALL/uv"
INSTALLER
""",
    )
    executable(
        bin_dir / "mock-uv",
        """
echo "uv $*" >> "$TEST_LOG"
if [ "$1" = venv ]; then
    mkdir -p "$4/bin"
    cp "$TEST_BIN/mock-python" "$4/bin/python"
elif [ "$1" = sync ]; then
    mkdir -p .venv/bin
    cp "$TEST_BIN/mock-python" .venv/bin/python
elif [ "$1" = run ]; then
    echo "launch-full $*" >> "$TEST_LOG"
elif [[ "$*" == *mlx-whisper* ]] && [ "$TEST_MLX_FAILS" = 1 ]; then
    exit 1
fi
""",
    )
    executable(
        bin_dir / "mock-python",
        """
if [ "$1" = -c ]; then
    echo 8501
else
    echo "launch $SPEAKER_TRANSCRIBE_BASIC $SPEAKER_TRANSCRIBE_BACKEND $*" >> "$TEST_LOG"
    test "$PWD/src" = "${PYTHONPATH%%:*}"
fi
""",
    )
    return {"project": project, "env": env, "log": log}


def launch(machine: dict, *args: str) -> None:
    result = subprocess.run(
        ["/bin/bash", str(machine["project"] / "run.sh"), *args],
        cwd=machine["project"].parent,
        env=machine["env"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("system", "arch", "version", "args", "mlx_fails", "expected"),
    [
        ("Darwin", "arm64", "15.0", [], False, "mlx"),
        ("Darwin", "arm64", "15.0", ["--cpu"], False, "faster-whisper"),
        ("Darwin", "x86_64", "15.0", [], False, "faster-whisper"),
        ("Darwin", "arm64", "13.0", [], False, "faster-whisper"),
        ("Darwin", "arm64", "15.0", [], True, "faster-whisper"),
        ("Linux", "x86_64", "", [], False, "faster-whisper"),
        ("Linux", "aarch64", "", [], False, "faster-whisper"),
    ],
)
def test_quick_start_first_and_repeat_launch(
    tmp_path, system, arch, version, args, mlx_fails, expected
):
    machine = make_machine(tmp_path, system, arch, version, mlx_fails)
    for _ in range(2):
        launch(machine, *args)

    calls = machine["log"].read_text()
    assert calls.count("installer-download") == 1
    assert calls.count("uv venv") == 1  # the second start reuses the environment
    assert calls.count(f"launch 1 {expected} -m streamlit run app.py") == 2
    assert "--server.address 127.0.0.1 --server.port 8501" in calls
    assert "pyannote" not in calls
    assert "uv sync" not in calls
    if expected == "mlx" or mlx_fails:
        # mlx-whisper's torch dependency is skipped; faster-whisper only on fallback.
        assert "--no-deps mlx-whisper" in calls
        assert ("requirements/whisper.txt" in calls) == mlx_fails
    else:
        assert "mlx" not in calls


def test_full_setup_syncs_the_project_and_creates_env_file(tmp_path):
    machine = make_machine(tmp_path, "Linux", "x86_64", "", mlx_fails=False)
    launch(machine, "--full")

    calls = machine["log"].read_text()
    assert "uv sync --inexact" in calls
    assert "launch-full run --no-sync streamlit run app.py" in calls
    assert "--server.port 8501" in calls
    assert "uv venv" not in calls
    assert (machine["project"] / ".env").read_text() == (ROOT / ".env.example").read_text()


@pytest.mark.parametrize("args", [["--typo"], ["--full", "--cpu"]])
def test_bad_arguments_fail_before_setup(args):
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "run.sh"), *args], text=True, capture_output=True
    )
    assert result.returncode == 2
    assert result.stderr
