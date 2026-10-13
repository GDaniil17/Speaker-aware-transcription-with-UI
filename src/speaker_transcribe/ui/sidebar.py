"""Settings sidebar: engine, model, language, speakers, audio range, advanced."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field

import streamlit as st

from speaker_transcribe.catalog import (
    DEFAULT_WHISPER_MODEL,
    DIARIZATION_MODEL,
    ENGINE_LANGUAGE_LISTS,
    ENGINES,
    GIGAAM_MODELS,
    GIGAAM_VERSIONS,
    HARDWARE_LABELS,
    MODEL_GUIDANCE,
    WHISPER_LANGUAGES,
    WHISPER_MODELS,
    gigaam_version,
    is_english_only_model,
)
from speaker_transcribe.chunking import (
    DEFAULT_CHUNK_LADDER,
    DEFAULT_STALL_TIMEOUT,
    format_ladder,
    parse_ladder,
)
from speaker_transcribe.config import Settings, get_hf_token
from speaker_transcribe.engine_install import engine_install_hint, install_command, install_engine
from speaker_transcribe.hub import diarization_model_cached, remote_size
from speaker_transcribe.model_downloads import (
    ModelDownloadTask,
    active_model_download_task,
    download_size,
    get_model_download_task,
    model_is_cached,
    start_model_download,
)
from speaker_transcribe.runtime import (
    PARAKEET_BACKEND_LABELS,
    describe_runtime,
    detect_backend,
    detect_device,
    engine_available,
    faster_whisper_available,
    hardware_profile,
    mlx_available,
    parakeet_backend,
)
from speaker_transcribe.ui.formatting import format_duration, format_size
from speaker_transcribe.ui.i18n import locale_key, progress_text, tr

TEMPERATURE_PRESETS: dict[str, float | tuple[float, ...]] = {
    "Fallback schedule": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "0.0 deterministic": 0.0,
    "0.2": 0.2,
    "0.4": 0.4,
    "0.6": 0.6,
    "0.8": 0.8,
    "1.0 creative": 1.0,
}
TEMPERATURE_OPTIONS = [*TEMPERATURE_PRESETS, "Custom value"]


@dataclass
class Choices:
    """Everything the sidebar decided, plus whether a run can start."""

    engine: str
    engine_ok: bool
    model_ready: bool
    diarize: bool
    diar_cached: bool
    hf_token: str
    ladder_error: str = ""
    # Settings fields other than input_path and hf_token.
    options: dict[str, object] = field(default_factory=dict)

    @property
    def can_run(self) -> bool:
        return self.engine_ok and self.model_ready and not self.ladder_error

    @property
    def needs_token(self) -> bool:
        """Diarization is on but its model is neither cached nor downloadable."""
        return self.diarize and not self.diar_cached and not self.hf_token

    def settings_for(self, input_path: str) -> Settings:
        return Settings(input_path=input_path, hf_token=self.hf_token or None, **self.options)


def render_sidebar(basic_mode: bool) -> Choices:
    """Draw the settings sidebar and return what was chosen."""
    with st.sidebar:
        _header()
        engine, engine_ok = _engine_picker(basic_mode)
        options, resolved_backend = _model_and_language(engine, basic_mode)
        model_ready = _model_download(
            engine, str(options["whisper_model"]), resolved_backend, engine_ok
        )
        st.divider()
        diarize, diar_cached, hf_token, speakers = _speakers(basic_mode)
        options.update(speakers, diarize=diarize)
        options.update(_audio_range())
        recovery, ladder_error = _advanced(engine, resolved_backend)
        options.update(recovery)
        _guidance()
    return Choices(
        engine=engine,
        engine_ok=engine_ok,
        model_ready=model_ready,
        diarize=diarize,
        diar_cached=diar_cached,
        hf_token=hf_token,
        ladder_error=ladder_error,
        options=options,
    )


def _header() -> None:
    heading, locale = st.columns([3, 2], vertical_alignment="center")
    heading.html(
        '<div role="heading" aria-level="2" '
        'style="height: 40px; display: flex; align-items: center; '
        'font-size: 20px; font-weight: 600; line-height: 24px">'
        f"{tr('Settings')}</div>",
    )
    locale.selectbox(
        "Interface language",
        ["EN", "RU"],
        key="ui_language",
        label_visibility="collapsed",
        help="Interface language / Язык интерфейса",
    )
    st.caption(tr("Choose a model, then download it before processing a file."))


def _engine_picker(basic_mode: bool) -> tuple[str, bool]:
    engine = st.selectbox(
        tr("Engine"),
        ["whisper"] if basic_mode else list(ENGINES),
        format_func=lambda key: ENGINES[key].label,
        help=tr("Choose the transcription engine. Whisper is the default."),
        key="engine_select",
    )
    engine_ok = engine_available(engine)
    if st.session_state.pop("_engine_installed", None) == engine:
        st.success(tr("Engine installed. Download its model next."))
    if not engine_ok:
        _install_engine_prompt(engine)
    return engine, engine_ok


def _install_engine_prompt(engine: str) -> None:
    st.warning(tr("This engine needs an optional package."))
    if install_command(engine) is None:
        st.caption(tr("Run this app from the project environment to install here."))
        st.code(engine_install_hint(engine), language="bash")
        return
    if not st.button(
        f"↓ {tr('Install engine')} · {ENGINES[engine].label}",
        key=f"install_engine_{engine}",
        use_container_width=True,
        help=f"{tr('Install from a terminal instead:')} {engine_install_hint(engine)}",
    ):
        return
    with st.status(tr("Installing engine…"), expanded=True) as status:
        output = st.empty()
        recent: list[str] = []

        def show_install_line(line: str) -> None:
            recent.append(line)
            output.code("\n".join(recent[-8:]), language="text")

        try:
            install_engine(engine, show_install_line)
            importlib.invalidate_caches()
            if not engine_available(engine):
                raise RuntimeError(tr("Restart the app to use the installed engine."))
        except Exception as exc:
            status.update(label=tr("Engine installation failed"), state="error")
            st.error(f"{tr('Engine installation failed')}: {exc}")
        else:
            status.update(label=tr("Engine installed. Download its model next."), state="complete")
            st.session_state["_engine_installed"] = engine
            st.rerun()


def _language_picker(
    engine_key: str, languages: dict[str, str] | None, title_case: bool
) -> str | None:
    """Recording-language selector; None means auto-detect."""
    if languages is None:  # the engine detects the language itself
        st.selectbox(
            tr("Recording language"),
            ["auto"],
            format_func=lambda _: tr("Auto-detect"),
            disabled=True,
            key=locale_key(f"language_{engine_key}", ["auto"]),
        )
        return None
    if len(languages) == 1:
        (code,) = languages
        st.selectbox(
            tr("Recording language"),
            [code],
            format_func=lambda c: f"{languages[c]} ({c})",
            disabled=True,
            key=locale_key(f"language_{engine_key}", [code]),
        )
        return code

    def name(code: str) -> str:
        return languages[code].title() if title_case else languages[code]

    codes = ["auto"] + sorted(languages, key=lambda c: languages[c])
    choice = st.selectbox(
        tr("Recording language"),
        codes,
        format_func=lambda c: tr("Auto-detect") if c == "auto" else f"{name(c)} ({c})",
        key=locale_key(f"language_{engine_key}", codes),
    )
    return None if choice == "auto" else choice


def _model_and_language(engine: str, basic_mode: bool) -> tuple[dict[str, object], str]:
    """Backend, model and recording language. Returns (options, resolved backend)."""
    device, compute_type = detect_device()
    backend = "auto"
    if engine == "whisper":
        backend_options = ["Auto"]
        if mlx_available():
            backend_options.append("mlx")
        if faster_whisper_available():
            backend_options.append("faster-whisper")
        # The quick-start launcher tells the UI which backend it installed.
        default_backend = os.environ.get("SPEAKER_TRANSCRIBE_BACKEND", "Auto")
        backend_label = st.selectbox(
            tr("Backend"),
            backend_options,
            index=backend_options.index(default_backend)
            if basic_mode and default_backend in backend_options
            else 0,
            format_func=tr,
            help=tr("MLX uses the Apple Silicon GPU. faster-whisper uses CUDA or CPU."),
            key=locale_key("backend_select", backend_options),
        )
        backend = "auto" if backend_label == "Auto" else backend_label
        resolved_backend = detect_backend() if backend == "auto" else backend
        runtime_label = describe_runtime(resolved_backend, device, compute_type)
    else:
        resolved_backend = "auto"
        runtime_label = ENGINES[engine].label
        if engine == "parakeet" and (parakeet := parakeet_backend()):
            runtime_label += f" · {PARAKEET_BACKEND_LABELS[parakeet]}"
    st.caption(f"{tr('Transcription')}: **{runtime_label}**")

    if engine == "whisper":
        models = list(WHISPER_MODELS)
        model = st.selectbox(
            tr("Model"), models, index=models.index(DEFAULT_WHISPER_MODEL), key="model_whisper"
        )
        if is_english_only_model(model):
            language = _language_picker("whisper_en", {"en": "English"}, title_case=False)
        else:
            language = _language_picker("whisper", WHISPER_LANGUAGES, title_case=True)
    else:
        eng = ENGINES[engine]
        if engine == "gigaam":
            model = _gigaam_model_picker()
        else:
            model = st.selectbox(tr("Model"), list(eng.models), key=f"model_{engine}")
        language = _language_picker(engine, eng.languages, title_case=False)

    options = {"engine": engine, "whisper_model": model, "language": language, "backend": backend}
    return options, resolved_backend


def _gigaam_model_picker() -> str:
    """GigaAM version first (v3 or v2), then that version's model variants."""
    versions = list(GIGAAM_VERSIONS)
    version = st.radio(
        tr("GigaAM version"),
        versions,
        format_func=lambda v: GIGAAM_VERSIONS[v],
        horizontal=True,
        help=tr("v3 is newer and more accurate. Its e2e models add punctuation and capitals."),
        key="gigaam_version",
    )
    models = [m for m in GIGAAM_MODELS if gigaam_version(m) == version]
    recommended = next(iter(GIGAAM_MODELS))
    model = st.selectbox(
        tr("Model"),
        models,
        format_func=lambda m: f"{m} · {tr('recommended')}" if m == recommended else m,
        key=f"model_gigaam_{version}",
    )
    st.caption(tr(GIGAAM_MODELS[model]))
    return model


@st.fragment(run_every=0.5)
def _download_progress(task: ModelDownloadTask) -> None:
    """Follow a server-owned download, including after the page is refreshed."""
    stage, fraction, eta, _error, done, elapsed = task.snapshot()
    if done:
        st.rerun()
    repo, _, step = stage.partition(" · ")
    localized_stage = f"{repo} · {progress_text(step)}" if step else tr(stage)
    label = f"{localized_stage} · {fraction:.0%} · {format_duration(elapsed)}"
    if eta is not None and eta > 0:
        label += f" / ~{format_duration(eta)}"
    st.progress(fraction, text=label)


def _model_download(engine: str, model: str, resolved_backend: str, engine_ok: bool) -> bool:
    """Model status and the download button. Returns True when the model is ready."""
    download_task = get_model_download_task(engine, model, resolved_backend)
    active_download = active_model_download_task()
    downloading_this = active_download is not None and active_download is download_task
    download_error = download_task.snapshot()[3] if download_task is not None else None
    model_ready = not downloading_this and model_is_cached(engine, model, resolved_backend)
    if model_ready:
        st.success(f"✓ {tr('Downloaded and ready')}")
        return True
    if not engine_ok:
        return False

    size_label = ""  # what still has to be downloaded, when the Hub can tell us
    if downloading_this:
        st.info(f"↓ {tr('Downloading model…')} · **{model}**")
    elif active_download is not None:
        st.info(f"{tr('Another model is downloading.')} · **{active_download.model}**")
    else:
        size = download_size(engine, model, resolved_backend)
        if size:
            size_label = f" · {format_size(size)}"
        st.info(f"↓ {tr('Download required')} · **{model}**{size_label}")
        if download_error:
            st.error(f"{tr('Model download failed')}: {download_error}")
    if active_download is not None:
        button_label = tr("Downloading model…")
    elif download_error:
        button_label = tr("Retry download")
    else:
        button_label = tr("Download model") + size_label
    if st.button(
        f"↓ {button_label}",
        key="download_selected_model",
        use_container_width=True,
        disabled=active_download is not None,
    ):
        try:
            start_model_download(engine, model, resolved_backend)
            st.rerun()
        except Exception as exc:
            st.error(f"{tr('Model download failed')}: {exc}")
    if active_download is not None:
        _download_progress(active_download)
    return False


def _speakers(basic_mode: bool) -> tuple[bool, bool, str, dict[str, object]]:
    """Speaker identification. Returns (diarize, model cached, token, options)."""
    diar_cached = not basic_mode and diarization_model_cached(DIARIZATION_MODEL)
    env_token = get_hf_token() or ""
    hf_token = env_token
    diarize = st.toggle(
        tr("Identify speakers"),
        value=not basic_mode and (diar_cached or bool(hf_token.strip())),
        disabled=basic_mode,
        help=tr("Automatically assign transcript lines to speakers using pyannote."),
        key="diarize",
    )
    if basic_mode:
        st.caption(tr("Speaker identification needs the full setup (start with --full)."))
    elif diarize and diar_cached:
        st.caption(f"✓ {tr('Ready offline')}")
    elif diarize:
        speaker_model_size = remote_size(DIARIZATION_MODEL)
        st.caption(
            tr("A Hugging Face token is needed once to download the speaker model.")
            + (f" ({format_size(speaker_model_size)})" if speaker_model_size else "")
        )
        st.caption(
            f"[Hugging Face token](https://huggingface.co/settings/tokens) · "
            f"[Model terms](https://huggingface.co/{DIARIZATION_MODEL})"
        )
        typed_token = st.text_input(
            tr("Hugging Face token"),
            type="password",
            placeholder="HF_TOKEN from .env" if env_token else "hf_…",
            key="hf_token_input",
        )
        hf_token = typed_token.strip() or env_token
    if diarize:
        col_a, col_b = st.columns(2)
        min_speakers = col_a.number_input(tr("Min speakers"), 1, 20, 2, key="min_speakers")
        max_speakers = col_b.number_input(tr("Max speakers"), 1, 20, 5, key="max_speakers")
    else:
        min_speakers, max_speakers = 1, 5
        st.caption(tr("Transcription only; you can edit speaker labels later."))
    options = {"min_speakers": int(min_speakers), "max_speakers": int(max_speakers)}
    return diarize, diar_cached, hf_token.strip(), options


def _audio_range() -> dict[str, object]:
    modes = ["Whole file", "First N seconds", "Time range"]
    with st.expander(tr("Audio range")):
        mode = st.radio(
            tr("Audio range"),
            modes,
            format_func=tr,
            label_visibility="collapsed",
            key=locale_key("fragment_mode", modes),
        )
        start, duration = 0.0, None
        if mode == "First N seconds":
            duration = float(st.number_input(tr("Seconds"), 5, 36000, 60, step=5, key="first_n"))
        elif mode == "Time range":
            start = float(st.number_input(tr("Start (s)"), 0, 36000, 0, step=5, key="range_start"))
            end = float(st.number_input(tr("End (s)"), 1, 36000, 120, step=5, key="range_end"))
            duration = max(0.0, end - start)
    return {"start": start, "duration": duration}


def _advanced(engine: str, resolved_backend: str) -> tuple[dict[str, object], str]:
    """Stall recovery and decoder parameters. Returns (options, chunk-ladder error)."""
    with st.expander(tr("Advanced settings")):
        options, ladder_error = _stall_recovery()
        if engine == "whisper":
            options.update(_whisper_parameters(resolved_backend))
    return options, ladder_error


def _stall_recovery() -> tuple[dict[str, object], str]:
    ladder: tuple[float, ...] = DEFAULT_CHUNK_LADDER
    ladder_error = ""
    with st.expander(tr("Recovery for long files")):
        st.caption(tr("If progress stops, retry the recording in smaller pieces."))
        chunk_fallback = st.checkbox(
            tr("Split into chunks when a run stalls"), value=True, key="chunk_fallback"
        )
        stall_timeout = st.number_input(
            tr("Stall timeout (seconds)"),
            min_value=0,
            max_value=7200,
            value=int(DEFAULT_STALL_TIMEOUT),
            step=30,
            key="stall_timeout",
        )
        ladder_text = st.text_input(
            tr("Chunk ladder (minutes)"),
            value=format_ladder(DEFAULT_CHUNK_LADDER),
            disabled=not chunk_fallback,
            key="chunk_ladder",
        )
        try:
            ladder = parse_ladder(ladder_text)
        except ValueError as exc:
            ladder_error = str(exc)
            st.error(f"{tr('Chunk ladder (minutes)')}: {ladder_error}")
        whole_first = st.checkbox(tr("Try the whole file first"), value=True, key="whole_first")
        fail_on_stall = st.checkbox(
            tr("Fail if some audio stalls at every chunk size"), value=False, key="fail_on_stall"
        )
    options = {
        "stall_timeout": float(stall_timeout),
        "chunk_fallback": chunk_fallback,
        "chunk_ladder": ladder,
        "try_whole_file_first": whole_first,
        "skip_failed_chunks": not fail_on_stall,
    }
    return options, ladder_error


def _whisper_parameters(resolved_backend: str) -> dict[str, object]:
    beam_options = ["Auto", 1, 2, 3, 4, 5]
    beam_choice = st.selectbox(
        tr("Beam size"),
        beam_options,
        format_func=lambda value: tr(value) if isinstance(value, str) else str(value),
        key=locale_key("beam_size", beam_options),
    )
    options: dict[str, object] = {
        "beam_size": None if beam_choice == "Auto" else int(beam_choice),
        "vad_filter": st.checkbox(tr("VAD filter (skip silence)"), value=True, key="vad_filter"),
    }
    with st.expander(tr("Advanced Whisper parameters")):
        temperature_label = st.selectbox(
            tr("Temperature"),
            TEMPERATURE_OPTIONS,
            format_func=tr,
            key=locale_key("temperature", TEMPERATURE_OPTIONS),
        )
        if temperature_label == "Custom value":
            options["temperature"] = float(
                st.number_input(
                    tr("Custom value"),
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key="custom_temperature",
                )
            )
        else:
            options["temperature"] = TEMPERATURE_PRESETS[temperature_label]
        options["condition_on_previous_text"] = st.checkbox(
            tr("Condition on previous text"), value=True, key="condition_previous"
        )
        prompt = st.text_area(tr("Initial prompt"), height=80, key="initial_prompt").strip()
        options["initial_prompt"] = prompt or None
        options["chunk_length"] = int(
            st.slider(tr("Chunk length (s)"), 15, 120, 60, 15, key="chunk_length")
        )
        options["compression_ratio_threshold"] = float(
            st.number_input(
                tr("Compression ratio threshold"), 0.0, 10.0, 2.4, 0.1, key="compression_ratio"
            )
        )
        options["log_prob_threshold"] = float(
            st.number_input(
                tr("Log probability threshold"), -10.0, 0.0, -1.0, 0.1, key="log_probability"
            )
        )
        options["no_speech_threshold"] = float(
            st.slider(tr("No-speech threshold"), 0.0, 1.0, 0.6, 0.05, key="no_speech")
        )
        if st.checkbox(tr("Hallucination silence guard"), value=False, key="hallucination_guard"):
            options["hallucination_silence_threshold"] = float(
                st.slider(
                    tr("Silence guard threshold (s)"),
                    0.5,
                    10.0,
                    2.0,
                    0.5,
                    key="silence_guard_threshold",
                )
            )
        if resolved_backend != "mlx":
            options["batch_size"] = int(st.slider(tr("Batch size"), 1, 32, 8, 1, key="batch_size"))
    return options


def _guidance() -> None:
    with st.expander(tr("Which model should I choose?")):
        profile = hardware_profile()
        st.caption(HARDWARE_LABELS[profile])
        for eng_label, rec_model, why in MODEL_GUIDANCE[profile]:
            st.markdown(f"- **{eng_label}** · `{rec_model}` — {tr(why)}")
        with st.expander(tr("Supported languages")):
            eng_filter = st.selectbox(
                tr("Engine"),
                list(ENGINE_LANGUAGE_LISTS),
                format_func=lambda k: ENGINES[k].label,
                key="lang_list_engine",
            )
            langs = ENGINE_LANGUAGE_LISTS[eng_filter]
            st.caption(f"{len(langs)} {tr('languages')}")
            rows = "".join(
                f"| {name.title()} | `{code}` |\n"
                for code, name in sorted(langs.items(), key=lambda kv: kv[1])
            )
            st.markdown(f"| {tr('Language')} | {tr('Code')} |\n| --- | --- |\n{rows}")
