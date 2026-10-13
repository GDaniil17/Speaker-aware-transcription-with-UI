"""Streamlit entry point: `streamlit run app.py` (or ./run.sh, run.bat, make run)."""

from pathlib import Path

from speaker_transcribe.ui.app import main

main(app_dir=Path(__file__).resolve().parent)
