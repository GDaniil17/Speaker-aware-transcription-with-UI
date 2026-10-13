"""Interface translations (English and Russian).

English strings are the keys; `tr` returns the Russian text when the interface
language is RU. The interface language never changes the language a recording
is transcribed in.
"""

from __future__ import annotations

import streamlit as st

RU: dict[str, str] = {
    "Settings": "Настройки",
    "Engine": "Движок",
    "Model": "Модель",
    "Language": "Язык записи",
    "Auto-detect": "Определить автоматически",
    "Backend": "Вычисления",
    "Auto": "Автоматически",
    "Identify speakers": "Определять говорящих",
    "Min speakers": "Мин. говорящих",
    "Max speakers": "Макс. говорящих",
    "Audio range": "Фрагмент записи",
    "Whole file": "Весь файл",
    "First N seconds": "Первые N секунд",
    "Time range": "Интервал",
    "Seconds": "Секунды",
    "Start (s)": "Начало (с)",
    "End (s)": "Конец (с)",
    "Advanced settings": "Дополнительные настройки",
    "Recovery for long files": "Если обработка застряла",
    "Try the whole file first": "Сначала обработать файл целиком",
    "Split into chunks when a run stalls": "При зависании разбить на фрагменты",
    "Stall timeout (seconds)": "Таймаут без прогресса (с)",
    "Chunk ladder (minutes)": "Размеры фрагментов (мин)",
    "Fail if some audio stalls at every chunk size": "Ошибка, если фрагмент не обработан",
    "Beam size": "Размер луча",
    "VAD filter (skip silence)": "Пропускать тишину (VAD)",
    "Advanced Whisper parameters": "Параметры Whisper",
    "Temperature": "Температура",
    "Fallback schedule": "Постепенное повышение при сбое",
    "Custom value": "Своё значение",
    "0.0 deterministic": "0.0 · точнее",
    "1.0 creative": "1.0 · свободнее",
    "Condition on previous text": "Учитывать предыдущий текст",
    "Initial prompt": "Подсказка для распознавания",
    "Chunk length (s)": "Длина блока (с)",
    "Compression ratio threshold": "Порог повтора текста",
    "Log probability threshold": "Порог уверенности",
    "No-speech threshold": "Порог тишины",
    "Hallucination silence guard": "Защита от текста на тишине",
    "Silence guard threshold (s)": "Порог защиты (с)",
    "Batch size": "Размер пакета",
    "Upload file": "Загрузить файл",
    "Local path": "Файл на компьютере",
    "Audio or video file": "Аудио или видео",
    "Browse files on this machine": "Выбрать файл на компьютере",
    "Path to an existing file on this machine": "Путь к файлу на этом компьютере",
    "Browse": "Просмотр",
    "Permission denied for this folder.": "Нет доступа к этой папке.",
    "File not found.": "Файл не найден.",
    "Transcribe": "Начать обработку",
    "Transcript": "Расшифровка",
    "Working…": "Обработка…",
    "Done": "Готово",
    "Total time": "Общее время",
    "Failed": "Ошибка",
    "Preparing…": "Подготовка…",
    "elapsed": "прошло",
    "left": "осталось",
    "Downloaded and ready": "Скачана и готова к работе",
    "Download model": "Скачать модель",
    "Downloading model…": "Скачивание модели…",
    "Another model is downloading.": "Сейчас скачивается другая модель.",
    "Retry download": "Повторить скачивание",
    "Download required": "Нужно скачать модель",
    "Choose a model, then download it before processing a file.": (
        "Выберите модель и скачайте её до начала обработки файла."
    ),
    "Model download failed": "Не удалось скачать модель",
    "Model is ready": "Модель готова",
    "Install the engine first": "Сначала установите движок",
    "This engine needs an optional package.": "Для этого движка нужен дополнительный пакет.",
    "Install engine": "Установить движок",
    "Installing engine…": "Установка движка…",
    "Engine installation failed": "Не удалось установить движок",
    "Engine installed. Download its model next.": (
        "Движок установлен. Теперь скачайте его модель."
    ),
    "Restart the app to use the installed engine.": (
        "Перезапустите приложение, чтобы использовать установленный движок."
    ),
    "Install from a terminal instead:": "Или установите из терминала:",
    "Run this app from the project environment to install here.": (
        "Запустите приложение из окружения проекта, чтобы установить движок здесь."
    ),
    "Recording language": "Язык записи",
    "Transcription": "Распознавание",
    "Speaker identification": "Определение говорящих",
    "off": "выключено",
    "No token needed": "Токен не нужен",
    "Hugging Face token": "Токен Hugging Face",
    "Ready offline": "Работает без сети",
    "A Hugging Face token is needed once to download the speaker model.": (
        "Чтобы скачать модель для разделения говорящих, один раз потребуется токен Hugging Face."
    ),
    "Which model should I choose?": "Какую модель выбрать?",
    "Supported languages": "Поддерживаемые языки",
    "languages": "языков",
    "Code": "Код",
    "No speech detected in the selected fragment.": "В выбранном фрагменте речь не найдена.",
    "Speaker-aware transcription": "Расшифровка с разделением говорящих",
    "Transcribe locally and identify speakers. Your files stay on this machine.": (
        "Распознавайте речь и разделяйте говорящих локально. Файлы остаются на вашем компьютере."
    ),
    "Upload a file, then start transcription. Download the model first.": (
        "Добавьте аудио или видео и запустите обработку. Сначала скачайте выбранную модель."
    ),
    "Add a recording": "Добавьте запись",
    "Source": "Источник",
    "Download the selected model in Settings to start processing.": (
        "Скачайте выбранную модель в настройках, чтобы начать обработку."
    ),
    "Choose the transcription engine. Whisper is the default.": (
        "Выберите движок распознавания. По умолчанию используется Whisper."
    ),
    "MLX uses the Apple Silicon GPU. faster-whisper uses CUDA or CPU.": (
        "MLX использует графику Apple Silicon; faster-whisper — CUDA или процессор."
    ),
    "Automatically assign transcript lines to speakers using pyannote.": (
        "Автоматически назначать говорящих репликам с помощью pyannote."
    ),
    "Speaker identification needs the full setup (start with --full).": (
        "Для определения говорящих нужна полная установка (запуск с --full)."
    ),
    "Transcription only; you can edit speaker labels later.": (
        "Только распознавание; говорящих можно назначить позже."
    ),
    "If progress stops, retry the recording in smaller pieces.": (
        "Если обработка остановится, повторить её по небольшим фрагментам."
    ),
    "— select a folder or file —": "— выберите папку или файл —",
    "⬆️  .. (up one level)": "⬆️  .. (на уровень выше)",
    "Click a timestamp to play from there. Edit speakers and lines below.": (
        "Нажмите на время, чтобы перейти к нужному месту. Ниже можно править реплики и говорящих."
    ),
    "Edit speakers and lines below, then download your transcript.": (
        "Исправьте реплики и говорящих, затем скачайте расшифровку."
    ),
    "Paste the token in Settings, or turn off speaker identification.": (
        "Вставьте токен в настройках или отключите определение говорящих."
    ),
    "This step reports time but no live percentage.": (
        "Для этого шага видно время, но нет процента выполнения."
    ),
    "Last run": "Последний запуск",
    "Best multilingual quality; float16 on CUDA keeps large models fast.": (
        "Высокое качество для разных языков; на CUDA крупные модели работают быстрее."
    ),
    "Very fast for the 25 EU languages, language auto-detected.": (
        "Быстро распознаёт 25 европейских языков, язык определяет сама."
    ),
    "Strong accuracy across 30 languages; the heaviest, wants GPU VRAM.": (
        "Точная модель для 30 языков; потребуется много памяти видеокарты."
    ),
    "Near-large quality at high speed on the Apple GPU.": (
        "Близкое к large качество при высокой скорости на графике Apple."
    ),
    "Fast EU-language transcription; faster still with parakeet-mlx on the GPU.": (
        "Быстро распознаёт европейские языки; ещё быстрее с parakeet-mlx на графике."
    ),
    "Good for Chinese, Cantonese, English, Japanese, and Korean audio.": (
        "Подходит для китайского, кантонского, английского, японского и корейского."
    ),
    "Russian audio with punctuation, very fast on the CPU cores.": (
        "Русская речь с пунктуацией, очень быстро на процессоре."
    ),
    "Best speed/quality balance on CPU (int8); avoid large-v3 (slow).": (
        "Хороший баланс скорости и качества на процессоре (int8)."
    ),
    "Fastest option for English-only audio.": "Самый быстрый вариант для английской речи.",
    "~15× faster than Whisper for its 5 languages — great on CPU.": (
        "Быстрая модель для пяти языков; хорошо работает на процессоре."
    ),
    "Best choice for Russian on CPU: fast ONNX model, no PyTorch needed.": (
        "Удобный выбор для русской речи на процессоре: быстрая ONNX-модель."
    ),
    # GigaAM version and model picker
    "GigaAM version": "Версия GigaAM",
    "v3 is newer and more accurate. Its e2e models add punctuation and capitals.": (
        "v3 новее и точнее. Модели e2e расставляют знаки препинания и заглавные буквы."
    ),
    "recommended": "рекомендуется",
    "Russian · punctuation and capitals, RNN-T decoder, best accuracy": (
        "Русский · с пунктуацией и заглавными буквами, декодер RNN-T, лучшая точность"
    ),
    "Russian · punctuation and capitals, CTC decoder, slightly faster": (
        "Русский · с пунктуацией и заглавными буквами, декодер CTC, чуть быстрее"
    ),
    "Russian · lowercase, no punctuation, RNN-T decoder": (
        "Русский · строчные буквы без пунктуации, декодер RNN-T"
    ),
    "Russian · lowercase, no punctuation, CTC decoder": (
        "Русский · строчные буквы без пунктуации, декодер CTC"
    ),
    # Transcript editor (player.js)
    "Speakers:": "Говорящие:",
    "Add speaker": "Добавить говорящего",
    "Speaker": "Говорящий",
    "New speaker…": "Новый говорящий…",
    "New speaker name for this line:": "Имя нового говорящего для этой реплики:",
    "Jump to": "Перейти к",
    "Speaker labels": "Имена говорящих",
    "Include speaker names in the downloaded file": "Добавить имена говорящих в скачанный файл",
    "This browser can't play this video format. Preparing an MP3 preview of the audio…": (
        "Браузер не может воспроизвести этот формат видео. Готовим MP3 со звуком…"
    ),
    "This browser can't play this video format, so an MP3 of its audio plays instead.": (
        "Браузер не может воспроизвести этот формат видео, поэтому играет MP3 со звуком."
    ),
    "No preview for media over 150 MB: this server doesn't serve static files. "
    "Restart the app so it picks up enableStaticServing from .streamlit/config.toml.": (
        "Нет предпросмотра для файлов больше 150 МБ: сервер не раздаёт статические файлы. "
        "Перезапустите приложение, чтобы применить enableStaticServing из .streamlit/config.toml."
    ),
}


def is_russian() -> bool:
    """True when the interface language selector is set to RU."""
    return st.session_state.get("ui_language", "EN") == "RU"


def tr(label: str) -> str:
    """`label` in the current interface language."""
    return RU.get(label, label) if is_russian() else label


def locale_key(name: str, choices: list | tuple) -> str:
    """Widget key per interface language, carrying the selection across a switch.

    Option labels are translated, so a widget needs a fresh key when the
    language changes; this copies the selected value over to the new key.
    """
    language = st.session_state.get("ui_language", "EN")
    key = f"{name}_{language}"
    other = f"{name}_{'EN' if language == 'RU' else 'RU'}"
    switching = st.session_state.get("_locale_seen") != language
    if (switching or key not in st.session_state) and st.session_state.get(other) in choices:
        st.session_state[key] = st.session_state[other]
    return key


def remember_locale() -> None:
    """Record the language this run rendered in; call at the end of every run."""
    st.session_state["_locale_seen"] = st.session_state.get("ui_language", "EN")


def progress_text(message: str) -> str:
    """Translate pipeline status text without changing engine diagnostics."""
    if not is_russian():
        return message
    prefixes = {
        "Preparing audio": "Подготовка аудио",
        "Loading diarization model (local copy)": "Загрузка модели говорящих (локальная копия)",
        "Downloading diarization model": "Скачивание модели говорящих",
        "Diarizing on": "Определение говорящих на",
        "Diarizing:": "Определение говорящих:",
        "Diarizing ·": "Определение говорящих ·",
        "Loading Whisper model": "Загрузка модели Whisper",
        "Loading Parakeet model": "Загрузка модели Parakeet",
        "Loading Qwen3-ASR model": "Загрузка модели Qwen3-ASR",
        "Loading SenseVoice model": "Загрузка модели SenseVoice",
        "Loading GigaAM model": "Загрузка модели GigaAM",
        "Transcribing whole file": "Распознавание всего файла",
        "Transcribing with": "Распознавание через",
        "Transcribing": "Распознавание",
        "Downloading model": "Скачивание модели",
        "Skipping diarization — transcription only": (
            "Определение говорящих пропущено (только распознавание)"
        ),
        "Done": "Готово",
    }
    for source, translated in prefixes.items():
        if message.startswith(source):
            return translated + message[len(source) :]
    return message
