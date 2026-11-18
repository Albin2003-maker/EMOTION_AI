"""
EMOTION AI - The Voice Emotion Intelligence System
==================================================

Gradio 6 frontend for the frozen HuBERT + noise-augmented Logistic Regression pipeline.

This file only *presents* results. All prediction work happens in
``src.inference.predict_emotion(audio_path)``, which is imported once at
startup (so the models load once, not on every request) and is never
modified or re-implemented here.

Run from the project root with the virtual environment active:

    python app.py
"""
from __future__ import annotations

import html
import math
import os
import sys
import traceback
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
# inference.py may use paths relative to the project root; make that true no
# matter which folder the app was launched from.
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402  (after sys.path / cwd setup)

REQUIRED_FILES = [
    "src/inference.py",
    "models/hubert_noise_augmented_classifier.joblib",
]
SAMPLE_CLIP = ROOT / "dataset" / "Actor_01" / "03-01-01-01-01-01-01.wav"
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".webm",
}

MISSING_FILES = [p for p in REQUIRED_FILES if not (ROOT / p).exists()]

# --------------------------------------------------------------------------- #
# Existing ML pipeline (imported once, outside any callback)
# --------------------------------------------------------------------------- #
IMPORT_ERROR: str | None = None
try:
    from src.inference import predict_emotion  # type: ignore
except Exception as exc:  # keep the UI alive and explain what is wrong
    predict_emotion = None  # type: ignore
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    print("[EMOTION AI] Could not import src.inference.predict_emotion", file=sys.stderr)
    traceback.print_exc()

# --------------------------------------------------------------------------- #
# Emotion colour system (inspired by Inside Out's emotion-colour associations;
# all shapes below are original)
# --------------------------------------------------------------------------- #
EMOTIONS: dict[str, dict[str, str]] = {
    "Happy": {
        "color": "#FFD93D",
        "glyph": (
            '<circle cx="12" cy="12" r="4"/>'
            '<path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3'
            'M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/>'
        ),
    },
    "Sad": {
        "color": "#4D96FF",
        "glyph": '<path d="M12 3.2c3.3 4.1 5.6 6.9 5.6 10.1a5.6 5.6 0 0 1-11.2 0C6.4 10.100 8.700 7.300 12 3.200z"/>',
    },
    "Angry": {
        "color": "#FF4D4D",
        "glyph": (
            '<path d="M12 2.800c.9 3.700 5.300 5.700 5.300 10.400a5.300 5.300 0 0 1-10.600 0'
            'c0-2.100 1-3.400 2.100-4.500.2 1.300.8 2.100 1.700 2.500C9.800 8.400 10.900 5.600 12 2.800z"/>'
        ),
    },
    "Fearful": {
        "color": "#9B72CF",
        "glyph": (
            '<path d="M2.500 12S6 5.500 12 5.500 21.500 12 21.500 12 18 18.500 12 18.500 2.500 12 2.500 12z"/>'
            '<circle cx="12" cy="12" r="2.800"/>'
        ),
    },
    "Disgust": {
        "color": "#6BCB77",
        "glyph": '<path d="M12 12a1.500 1.500 0 0 1 3 0 3.500 3.500 0 0 1-7 0 5.500 5.500 0 0 1 11 0 7.500 7.500 0 0 1-15 0"/>',
    },
    "Calm": {
        "color": "#4ECDC4",
        "glyph": (
            '<path d="M3 8.500c3-2.600 6 2.600 9 0s6-2.600 9 0"/>'
            '<path d="M3 15.500c3-2.600 6 2.600 9 0s6-2.600 9 0"/>'
        ),
    },
    "Neutral": {
        "color": "#AAB2BD",
        "glyph": '<circle cx="12" cy="12" r="8"/><path d="M8.500 12h7"/>',
    },
    "Surprised": {
        "color": "#FF9F43",
        "glyph": '<path d="M12 3c.6 4.800 1.700 7.200 9 9-7.300 1.800-8.400 4.200-9 9-.6-4.800-1.700-7.200-9-9 7.300-1.800 8.400-4.200 9-9z"/>',
    },
}

# Order of the arcs around the wheel (clockwise from the top).
WHEEL_ORDER = ["Happy", "Surprised", "Angry", "Fearful", "Sad", "Calm", "Disgust", "Neutral"]

# Only used if a pipeline returns scores as a bare list/array instead of a
# {emotion: score} mapping. This is the order given in the project brief
# (alphabetical). A dict return value makes this unnecessary.
SEQUENCE_SCORE_ORDER = ["Angry", "Calm", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

_NAME_LOOKUP = {name.lower(): name for name in EMOTIONS}
_NAME_LOOKUP.update(
    {"anger": "Angry", "fear": "Fearful", "surprise": "Surprised", "disgusted": "Disgust", "sadness": "Sad", "happiness": "Happy"}
)

MUTED = "#8794AB"
IDLE_GLYPH = '<path d="M5 10v4M9 7v10M13 9v6M17 6v12M21 11v2" transform="translate(-1 0)"/>'


def _canon(name: object) -> str | None:
    return _NAME_LOOKUP.get(str(name).strip().lower())


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------- #
# SVG building blocks
# --------------------------------------------------------------------------- #
def glyph_svg(name: str, size: int = 20) -> str:
    return (
        f'<svg class="glyph" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{EMOTIONS[name]["glyph"]}</svg>'
    )


def wheel_svg(active: str | None = None, state: str = "idle", size: int = 200) -> str:
    """Eight-arc emotion wheel: the logo mark and the large result indicator."""
    r, stroke, gap = 80.0, 16.0, 14.0
    step = 360 / len(WHEEL_ORDER)
    circumference = 2 * math.pi * r
    cap_deg = math.degrees((stroke / 2) / r)
    visible_deg = step - gap
    dash = r * math.radians(visible_deg) - stroke  # round caps add `stroke` in total

    arcs = []
    for i, name in enumerate(WHEEL_ORDER):
        start = -90 + i * step + gap / 2 + cap_deg
        on = " on" if name == active else ""
        arcs.append(
            f'<circle class="arc{on}" style="--i:{i};--c:{EMOTIONS[name]["color"]}" '
            f'cx="100" cy="100" r="{r}" fill="none" stroke="{EMOTIONS[name]["color"]}" '
            f'stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-dasharray="{dash:.2f} {circumference - dash:.2f}" '
            f'transform="rotate({start:.2f} 100 100)"/>'
        )

    if active in EMOTIONS:
        centre_color = EMOTIONS[active]["color"]
        centre_markup = EMOTIONS[active]["glyph"]
    else:
        centre_color = MUTED
        centre_markup = IDLE_GLYPH
    label = f"Emotion wheel showing {active}" if active else "Emotion wheel showing the eight emotion colours"
    return (
        f'<svg class="wheel is-{state}" width="{size}" height="{size}" viewBox="0 0 200 200" role="img" '
        f'aria-label="{_esc(label)}">{"".join(arcs)}'
        f'<svg x="72" y="72" width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" style="color:{centre_color}" '
        f'class="wheel-glyph">{centre_markup}</svg></svg>'
    )


# --------------------------------------------------------------------------- #
# HTML renderers
# --------------------------------------------------------------------------- #
def _pred_shell(wheel: str, inner: str, state: str, accent: str = MUTED) -> str:
    return (
        f'<section class="pred is-{state}" style="--accent:{accent};--glow:{_rgba(accent, 0.20)}" aria-live="polite">'
        f'<div class="pred-wheel">{wheel}</div><div class="pred-text">{inner}</div></section>'
    )


def idle_prediction() -> str:
    inner = (
        '<h2 class="pred-title">No analysis yet</h2>'
        '<p class="pred-body">Add a recording, then choose Analyze voice. The predicted emotion '
        "and all eight model scores will appear here.</p>"
    )
    return _pred_shell(wheel_svg(None, "idle"), inner, "idle")


def loading_prediction() -> str:
    inner = (
        '<h2 class="pred-title">Analyzing voice</h2>'
        '<p class="pred-body">Extracting a HuBERT speech representation and running the emotion classifier. '
        "This usually takes a few seconds.</p>"
    )
    return _pred_shell(wheel_svg(None, "loading"), inner, "loading")


def notice_prediction(title: str, message: str, fix: str = "", detail: str = "") -> str:
    parts = [f'<h2 class="pred-title">{_esc(title)}</h2>', f'<p class="pred-body">{_esc(message)}</p>']
    if fix:
        parts.append(f'<p class="pred-fix">{_esc(fix)}</p>')
    if detail:
        parts.append(
            '<details class="pred-detail"><summary>Technical details</summary>'
            f"<pre>{_esc(detail)}</pre></details>"
        )
    return _pred_shell(wheel_svg(None, "idle"), "".join(parts), "notice")


def result_prediction(emotion: str, model_name: object = None) -> str:
    accent = EMOTIONS[emotion]["color"]
    display_name = str(model_name or "HuBERT + Logistic Regression")
    chips_html = (
        '<div class="ea-chips">'
        f'<span class="ea-chip" style="--c:{accent}">'
        f'<i class="ea-dot"></i>{_esc(display_name)}</span></div>'
    )
    inner = (
        '<p class="pred-label">Predicted emotion</p>'
        f'<h2 class="pred-word" style="color:{accent}">{emotion}</h2>'
        '<p class="pred-body">The vocal pattern in this clip most resembles '
        f"<strong>{emotion.lower()}</strong> speech. EMOTION AI estimates patterns in how a voice sounds; "
        "it does not know how the speaker actually feels.</p>"
        f"{chips_html}"
    )
    return _pred_shell(wheel_svg(emotion, "result"), inner, "result", accent)


def _scores_shell(rows: str, state: str) -> str:
    return (
        f'<section class="scores is-{state}">'
        '<div class="scores-head"><h3>Model scores</h3>'
        "<p>Classifier probability estimates across the eight trained vocal-expression labels. "
        "These describe patterns in the audio; they are not certainty about how a person feels.</p></div>"
        f'<ul class="sc-list">{rows}</ul></section>'
    )


def idle_scores(loading: bool = False) -> str:
    rows = []
    for name in WHEEL_ORDER:
        rows.append(
            f'<li class="sc-row" style="--c:{EMOTIONS[name]["color"]}">'
            f'<span class="row-ico">{glyph_svg(name)}</span><span class="row-name">{name}</span>'
            '<span class="sc-track"><span class="sc-fill"></span></span>'
            '<span class="row-val">&ndash;</span></li>'
        )
    return _scores_shell("".join(rows), "loading" if loading else "idle")


def result_scores(pct: dict[str, float | None], raw: dict[str, float | None], top: str) -> str:
    order = sorted(EMOTIONS, key=lambda n: (pct.get(n) is None, -(pct.get(n) or 0.0)))
    rows = []
    for name in order:
        value = pct.get(name)
        width = 0.0 if value is None else max(0.0, min(100.0, value))
        shown = "n/a" if value is None else f"{value:.1f}%"
        meter = (
            f'role="meter" aria-label="{name} model score" aria-valuemin="0" aria-valuemax="100" '
            f'aria-valuenow="{width:.1f}"'
        )
        title = "No score returned" if raw.get(name) is None else f"Raw model score: {raw[name]}"
        rows.append(
            f'<li class="sc-row{" sc-top" if name == top else ""}" '
            f'style="--c:{EMOTIONS[name]["color"]};--w:{width:.2f}%" title="{_esc(title)}">'
            f'<span class="row-ico">{glyph_svg(name)}</span><span class="row-name">{name}</span>'
            f'<span class="sc-track" {meter}><span class="sc-fill"></span></span>'
            f'<span class="row-val">{shown}</span></li>'
        )
    return _scores_shell("".join(rows), "result")


def header_html() -> str:
    return (
        '<header class="hdr">'
        f'<div class="hdr-mark">{wheel_svg(None, "idle", 76)}</div>'
        '<div class="hdr-copy"><h1>EMOTION AI</h1>'
        "<p class=\"hdr-sub\">The Voice Emotion Intelligence System</p></div></header>"
        '<p class="intro">Record or upload a short clip of speech. A pretrained HuBERT speech encoder '
        "turns the voice into a 768-dimensional representation, and a noise-augmented Logistic Regression "
        "classifier estimates how closely it matches eight acted vocal-expression labels: angry, calm, "
        "disgust, fearful, happy, neutral, sad and surprised.</p>"
    )


def input_intro_html() -> str:
    return (
        '<h2 class="panel-title">Your recording</h2>'
        '<p class="panel-note">Speak naturally for 3 to 10 seconds with one voice and little background '
        "noise. WAV files work best.</p>"
    )


def disclaimer_html() -> str:
    return (
        '<p class="disclaimer"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/>'
        '<path d="M12 11v5M12 7.800v.2"/></svg>'
        "<span>The models were trained on RAVDESS, a dataset of acted speech. Scores describe vocal patterns "
        "only. They cannot establish a person's true emotional state, intentions, personality or mental "
        "health.</span></p>"
    )


def startup_notice_html() -> str:
    problems = []
    if MISSING_FILES:
        problems.append("Missing project files: " + ", ".join(MISSING_FILES))
    if IMPORT_ERROR:
        problems.append("Could not import predict_emotion: " + IMPORT_ERROR)
    if not problems:
        return ""
    items = "".join(f"<li>{_esc(p)}</li>" for p in problems)
    return (
        '<div class="startup-notice" role="alert"><strong>EMOTION AI can&rsquo;t analyze audio yet.</strong>'
        f"<ul>{items}</ul><p>Start the app from the project root with your virtual environment active, "
        "and check that these files exist.</p></div>"
    )


# --------------------------------------------------------------------------- #
# Result parsing (presentation only: nothing is recomputed or re-ranked)
# --------------------------------------------------------------------------- #
class UnexpectedOutput(ValueError):
    pass


def _unpack(result: object) -> tuple[object, object, object, object]:
    if isinstance(result, dict):
        return (
            result.get("predicted_emotion"),
            result.get("scores"),
            result.get("model_name", "HuBERT + Logistic Regression"),
            None,
        )
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        padded = list(result) + [None] * (4 - len(result))
        return tuple(padded[:4])  # type: ignore[return-value]
    raise UnexpectedOutput(
        f"predict_emotion returned {type(result).__name__}; expected a dict "
        "containing predicted_emotion and scores."
    )


def _parse_scores(raw: object) -> dict[str, float]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, (list, tuple)) and len(raw) == len(SEQUENCE_SCORE_ORDER):
        items = list(zip(SEQUENCE_SCORE_ORDER, raw))
    else:
        raise UnexpectedOutput("`scores` must be a dict of emotion name to score.")
    parsed: dict[str, float] = {}
    for key, value in items:
        name = _canon(key)
        if name is None:
            continue
        try:
            parsed[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise UnexpectedOutput(f"Score for {key!r} is not a number: {value!r}") from exc
    if not parsed:
        raise UnexpectedOutput("None of the returned score names matched the eight emotions.")
    return parsed


def render_result(result: object) -> tuple[str, str]:
    predicted, scores_raw, model_name, _unused = _unpack(result)
    emotion = _canon(predicted) if isinstance(predicted, str) else None
    if emotion is None:
        raise UnexpectedOutput(f"Unrecognized predicted_emotion: {predicted!r}")
    raw = _parse_scores(scores_raw)
    # Scores in 0-1 are shown as percentages; scores already above 1 are
    # assumed to be on a 0-100 scale. The raw value stays in each row's tooltip.
    scale = 100.0 if max(abs(v) for v in raw.values()) <= 1.0 + 1e-9 else 1.0
    pct = {name: (raw[name] * scale if name in raw else None) for name in EMOTIONS}
    raw_full = {name: raw.get(name) for name in EMOTIONS}
    return result_prediction(emotion, model_name), result_scores(pct, raw_full, emotion)


# --------------------------------------------------------------------------- #
# Gradio callbacks
# --------------------------------------------------------------------------- #
def _fail(title: str, message: str, fix: str = "", detail: str = "") -> tuple[str, str]:
    return notice_prediction(title, message, fix, detail), idle_scores()


def analyze(audio_path: str | None):
    """Generator: shows a loading state, then the result or a clear error."""
    if not audio_path:
        yield _fail(
            "Add a recording first",
            "There is no audio to analyze yet.",
            "Record with the microphone or upload a clip, then choose Analyze voice.",
        )
        return

    yield loading_prediction(), idle_scores(loading=True)

    if predict_emotion is None:
        yield _fail(
            "The models didn't load",
            "EMOTION AI couldn't import its prediction code, so it can't analyze audio.",
            "Run the app from the project root with your virtual environment active, and check "
            "that src/inference.py and the files in models/ exist.",
            IMPORT_ERROR or "",
        )
        return

    path = Path(audio_path)
    if not path.is_file():
        yield _fail("That recording is gone", "The audio file couldn't be found.", "Record or upload it again.")
        return
    if path.stat().st_size == 0:
        yield _fail("That recording is empty", "The audio file has no data in it.", "Record a few seconds of speech, or upload a different file.")
        return
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        yield _fail(
            "That file type isn't supported",
            f"Files ending in {path.suffix or 'no extension'} can't be analyzed.",
            "Upload a WAV, MP3, FLAC, OGG or M4A file.",
        )
        return

    try:
        result = predict_emotion(str(path))
    except Exception as exc:  # inference must never crash the app
        traceback.print_exc()
        yield _fail(
            "The models couldn't analyze that clip",
            "Something went wrong while reading or processing the audio.",
            "Try a WAV recording of 3 to 10 seconds with clear speech. If it keeps failing, check the terminal for the full error.",
            f"{type(exc).__name__}: {exc}",
        )
        return

    try:
        yield render_result(result)
    except UnexpectedOutput as exc:
        traceback.print_exc()
        yield _fail(
            "The models returned something unexpected",
            "The analysis ran, but the result couldn't be displayed.",
            "Check that predict_emotion returns predicted_emotion and a scores dict covering all eight emotions.",
            str(exc),
        )


def reset_outputs():
    return idle_prediction(), idle_scores()


def clear_all():
    return (None, *reset_outputs())


def load_sample():
    return str(SAMPLE_CLIP)


# --------------------------------------------------------------------------- #
# Styling (kept inside the Gradio app: passed to launch())
# --------------------------------------------------------------------------- #
HEAD = """
<meta name="color-scheme" content="dark">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Instrument+Sans:wght@400..700&display=swap" rel="stylesheet">
"""

FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"

CSS = """
:root, .gradio-container {
  --ea-bg: #0B1220; --ea-panel: #101A2E; --ea-raise: #16233A; --ea-line: #25324A; --ea-line-soft: #1D2A42;
  --ea-text: #E8ECF4; --ea-muted: #9AA6BC; --ea-dim: #8794AB;
  --ea-display: "Bricolage Grotesque", "Instrument Sans", ui-sans-serif, system-ui, sans-serif;
  --ea-body: "Instrument Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}

/* Re-skin Gradio's own variables so every built-in component matches. */
.gradio-container {
  --body-background-fill: var(--ea-bg);
  --body-text-color: var(--ea-text);
  --body-text-color-subdued: var(--ea-muted);
  --background-fill-primary: var(--ea-panel);
  --background-fill-secondary: #0E1829;
  --block-background-fill: #0E1829;
  --block-border-color: var(--ea-line);
  --block-border-width: 1px;
  --block-radius: 16px;
  --block-label-background-fill: var(--ea-raise);
  --block-label-text-color: var(--ea-muted);
  --border-color-primary: var(--ea-line);
  --border-color-accent: var(--ea-text);
  --input-background-fill: #0E1829;
  --input-border-color: var(--ea-line);
  --color-accent: var(--ea-text);
  --color-accent-soft: var(--ea-raise);
  --button-primary-background-fill: #EEF2FA;
  --button-primary-background-fill-hover: #FFFFFF;
  --button-primary-text-color: #0B1220;
  --button-primary-text-color-hover: #0B1220;
  --button-primary-text-color-active: #0B1220;
  --button-primary-background-fill-active: #DDE4F2;
  --button-primary-border-color: transparent;
  --button-secondary-background-fill: transparent;
  --button-secondary-background-fill-hover: var(--ea-raise);
  --button-secondary-text-color: var(--ea-text);
  --button-secondary-border-color: #33415F;
  --shadow-drop: none;
  --shadow-drop-lg: none;
  --font: var(--ea-body);

  width: 100% !important;
  max-width: 1180px !important;
  box-sizing: border-box;
  margin-inline: auto !important;
  padding: 32px 24px 48px !important;
  font-family: var(--ea-body);
  color: var(--ea-text);
}
body, gradio-app { background: var(--ea-bg) !important; }
/* Gradio's inner wrapper adds 32px of side padding of its own; the container padding above is the page margin. */
.main.fillable { padding-inline: 0 !important; }
footer { display: none !important; }

/* Header */
.hdr { display: flex; align-items: center; gap: 22px; }
.hdr-mark { flex: none; line-height: 0; }
.hdr h1 {
  margin: 0; font-family: var(--ea-display); font-weight: 800; font-size: clamp(2.1rem, 5vw, 3.2rem);
  line-height: 1; letter-spacing: -0.02em; color: var(--ea-text);
}
.hdr-sub { margin: 8px 0 0; font-family: var(--ea-display); font-weight: 500; font-size: clamp(1rem, 2.2vw, 1.25rem); color: var(--ea-muted); }
.intro { max-width: 64ch; margin: 22px 0 6px; color: var(--ea-muted); font-size: 1.02rem; line-height: 1.65; }

/* Gradio 6 rows are CSS grids: set the two-column layout explicitly. */
.main-row { display: grid !important; grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); gap: 24px !important; align-items: start; }
.main-row > .column { min-width: 0 !important; width: auto !important; }

/* Panels */
.ea-panel {
  background: linear-gradient(180deg, #121E35 0%, #0F192D 100%) !important;
  border: 1px solid var(--ea-line) !important;
  border-radius: 24px !important;
  padding: 26px !important;
  box-shadow: 0 24px 60px -30px rgba(0, 0, 0, 0.7);
  gap: 18px !important;
}
.panel-title, .scores h3 { margin: 0; font-family: var(--ea-display); font-weight: 700; font-size: 1.35rem; letter-spacing: -0.01em; color: var(--ea-text); }
.panel-note { margin: 8px 0 0; max-width: 46ch; color: var(--ea-muted); line-height: 1.6; font-size: 0.97rem; }

/* Buttons */
#analyze-btn, #analyze-btn:hover, #analyze-btn:active, #analyze-btn:focus { color: #0B1220 !important; }
#analyze-btn { font-family: var(--ea-display); font-weight: 700; font-size: 1.05rem; min-height: 52px; border-radius: 14px; letter-spacing: 0.005em; transition: transform .15s ease, background .15s ease; }
#analyze-btn:hover { transform: translateY(-1px); }
#analyze-btn:focus-visible, #clear-btn:focus-visible, #sample-btn:focus-visible { outline: 2px solid var(--ea-text); outline-offset: 3px; }
#clear-btn { min-height: 52px; border-radius: 14px; font-weight: 600; border: 1px solid #33415F !important; }
#sample-btn { background: transparent !important; border: none !important; color: var(--ea-muted) !important; text-decoration: underline; text-underline-offset: 3px; font-weight: 500; justify-content: flex-start; padding: 0 !important; min-height: 0; box-shadow: none !important; }
#sample-btn:hover { color: var(--ea-text) !important; }
.button-row { gap: 12px !important; }

/* Prediction tile */
.pred {
  position: relative; display: flex; align-items: center; gap: 28px; padding: 26px; border-radius: 20px;
  border: 1px solid var(--ea-line-soft);
  background: radial-gradient(120% 140% at 8% 20%, var(--glow), transparent 62%), #0D1729;
  transition: background .5s ease;
}
.pred-wheel { flex: none; line-height: 0; }
.pred-text { min-width: 0; }
.pred-label { margin: 0 0 4px; color: var(--ea-muted); font-size: 0.95rem; }
.pred-word { margin: 0; font-family: var(--ea-display); font-weight: 800; font-size: clamp(2.6rem, 6vw, 4rem); line-height: 1; letter-spacing: -0.025em; }
.pred-title { margin: 0; font-family: var(--ea-display); font-weight: 700; font-size: 1.55rem; letter-spacing: -0.01em; color: var(--ea-text); }
.pred-body { margin: 12px 0 0; max-width: 52ch; color: var(--ea-muted); line-height: 1.6; font-size: 0.98rem; }
.pred-body strong { color: var(--ea-text); font-weight: 600; }
.pred-fix { margin: 12px 0 0; max-width: 52ch; color: var(--ea-text); line-height: 1.55; font-size: 0.98rem; }
.pred.is-notice { border-color: #3A4A68; }
.pred-detail { margin-top: 14px; color: var(--ea-dim); font-size: 0.88rem; }
.pred-detail summary { cursor: pointer; }
.pred-detail pre { margin: 8px 0 0; padding: 10px 12px; background: #0A1220; border-radius: 10px; white-space: pre-wrap; word-break: break-word; color: var(--ea-muted); font-size: 0.82rem; }
.ea-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.ea-chip { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px; border: 1px solid var(--ea-line); font-size: 0.86rem; color: var(--ea-text); background: rgba(255,255,255,0.02); }
.ea-chip .ea-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--c); display: inline-block; }

/* Wheel */
.wheel { overflow: visible; }
.wheel .arc { opacity: .62; transition: opacity .4s ease; }
.wheel.is-result .arc { opacity: .16; animation: ea-settle .7s ease-out both; }
.wheel.is-result .arc.on { opacity: 1; filter: drop-shadow(0 0 9px var(--c)); animation: ea-light .9s ease-out both; }
.wheel.is-loading .arc { animation: ea-chase 1.5s ease-in-out infinite; animation-delay: calc(var(--i) * 0.14s); }
@keyframes ea-chase { 0%, 100% { opacity: .18; } 40% { opacity: 1; } }
@keyframes ea-settle { from { opacity: .62; } to { opacity: .16; } }
@keyframes ea-light { 0% { opacity: .16; filter: drop-shadow(0 0 0 var(--c)); } 100% { opacity: 1; filter: drop-shadow(0 0 9px var(--c)); } }

/* Scores */
.scores { padding: 6px 2px 0; }
.scores-head p { margin: 8px 0 0; max-width: 60ch; color: var(--ea-muted); font-size: 0.92rem; line-height: 1.55; }
.sc-list { list-style: none; margin: 20px 0 0; padding: 0; display: grid; gap: 14px; }
.sc-row { display: grid; grid-template-columns: 22px 92px minmax(0, 1fr) 58px; align-items: center; gap: 12px; }
.row-ico { color: var(--c); display: inline-flex; }
.row-name { color: var(--ea-muted); font-weight: 500; }
.sc-row.sc-top .row-name { color: var(--ea-text); font-weight: 700; }
.sc-track { position: relative; display: block; height: 10px; border-radius: 999px; background: #1A2842; overflow: hidden; }
.sc-fill { display: block; height: 100%; width: var(--w, 0%); border-radius: inherit; background: var(--c); }
.scores.is-result .sc-fill { animation: ea-grow .9s cubic-bezier(.2,.8,.2,1) both; }
.scores.is-result .sc-row.sc-top .sc-fill { box-shadow: 0 0 14px -2px var(--c); }
.scores.is-loading .sc-track::after { content: ""; position: absolute; inset: 0; background: linear-gradient(90deg, transparent, rgba(255,255,255,.10), transparent); transform: translateX(-100%); animation: ea-sheen 1.3s ease-in-out infinite; }
@keyframes ea-grow { from { width: 0; } }
@keyframes ea-sheen { to { transform: translateX(100%); } }
.row-val { text-align: right; font-variant-numeric: tabular-nums; color: var(--ea-text); font-weight: 600; font-size: 0.95rem; }
.sc-row:not(.sc-top) .row-val { color: var(--ea-muted); font-weight: 500; }
.scores.is-idle .row-name, .scores.is-loading .row-name { color: var(--ea-muted); }

/* Footer + notices */
.disclaimer { display: flex; align-items: flex-start; gap: 10px; max-width: 78ch; margin: 28px 0 0; color: var(--ea-dim); font-size: 0.86rem; line-height: 1.6; }
.disclaimer svg { flex: none; margin-top: 3px; }
.startup-notice { margin: 18px 0 0; padding: 16px 18px; border: 1px solid #3A4A68; border-radius: 14px; background: #101A2E; color: var(--ea-text); line-height: 1.55; }
.startup-notice ul { margin: 8px 0; padding-left: 20px; color: var(--ea-muted); }
.startup-notice p { margin: 0; color: var(--ea-muted); }

/* Responsive */
@media (max-width: 860px) {
  .main-row { grid-template-columns: minmax(0, 1fr) !important; }
  .main-row > .column { min-width: 0 !important; width: 100% !important; }
  .ea-panel { padding: 20px !important; border-radius: 20px !important; }
}
@media (max-width: 560px) {
  .hdr { gap: 16px; }
  .hdr-mark svg { width: 52px; height: 52px; }
  .hdr h1 { font-size: 2rem; }
  .pred { flex-direction: column; align-items: flex-start; gap: 18px; padding: 20px; }
  .sc-row { grid-template-columns: 20px 78px minmax(0, 1fr) 52px; gap: 10px; }
}

@media (prefers-reduced-motion: reduce) {
  .wheel .arc, .wheel.is-result .arc, .wheel.is-result .arc.on, .wheel.is-loading .arc,
  .scores.is-result .sc-fill, .scores.is-loading .sc-track::after, #analyze-btn { animation: none !important; transition: none !important; }
  .wheel.is-loading .arc { opacity: .6; }
}
"""

# Fonts are applied through CSS (--font and the families loaded in HEAD); passing
# plain strings to the theme's `font=` argument crashes launch() in Gradio 6.31.0.
THEME = gr.themes.Base(
    primary_hue="slate",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
)

# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
with gr.Blocks(title="EMOTION AI | Voice Emotion Intelligence") as demo:
    gr.HTML(header_html(), padding=False, container=False, apply_default_css=False)
    if MISSING_FILES or IMPORT_ERROR:
        gr.HTML(startup_notice_html(), padding=False, container=False, apply_default_css=False)

    with gr.Row(equal_height=False, elem_classes=["main-row"]):
        with gr.Column(scale=5, min_width=320, elem_classes=["ea-panel"]):
            gr.HTML(input_intro_html(), padding=False, container=False, apply_default_css=False)
            audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                format="wav",
                show_label=False,
                buttons=["download"],
                elem_id="audio-input",
                waveform_options=gr.WaveformOptions(
                    waveform_color="#5C6B88", waveform_progress_color="#E8ECF4", show_recording_waveform=True
                ),
            )
            with gr.Row(elem_classes=["button-row"]):
                analyze_btn = gr.Button("Analyze voice", variant="primary", elem_id="analyze-btn", scale=3)
                clear_btn = gr.Button("Clear", variant="secondary", elem_id="clear-btn", scale=1, min_width=90)
            sample_btn = None
            if SAMPLE_CLIP.is_file():
                sample_btn = gr.Button("Load a sample clip", elem_id="sample-btn", size="sm")

        with gr.Column(scale=7, min_width=360, elem_classes=["ea-panel"]):
            prediction_out = gr.HTML(idle_prediction(), padding=False, container=False, apply_default_css=False)
            scores_out = gr.HTML(idle_scores(), padding=False, container=False, apply_default_css=False)

    gr.HTML(disclaimer_html(), padding=False, container=False, apply_default_css=False)

    outputs = [prediction_out, scores_out]
    analyze_btn.click(analyze, inputs=audio, outputs=outputs, show_progress="hidden", concurrency_limit=1)
    clear_btn.click(clear_all, outputs=[audio, *outputs], show_progress="hidden")
    audio.clear(reset_outputs, outputs=outputs, show_progress="hidden")
    if sample_btn is not None:
        sample_btn.click(load_sample, outputs=audio, show_progress="hidden")


# --------------------------------------------------------------------------- #
# Startup self-check
# --------------------------------------------------------------------------- #
def _startup_report() -> None:
    print("[EMOTION AI] Project root:", ROOT)
    for rel in REQUIRED_FILES:
        print(f"[EMOTION AI]   {'ok     ' if (ROOT / rel).exists() else 'MISSING'} {rel}")
    if predict_emotion is None:
        print("[EMOTION AI] predict_emotion could not be imported; the UI will show an error on analysis.")
        return
    if SAMPLE_CLIP.is_file():
        # Also warms up any lazily-loaded model so the first user request is fast.
        try:
            result = predict_emotion(str(SAMPLE_CLIP))
            predicted = _unpack(result)[0]
            print(f"[EMOTION AI] Models ready. Warm-up on sample clip predicted: {predicted}")
        except Exception as exc:
            print(f"[EMOTION AI] Warm-up on sample clip failed: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    _startup_report()
    demo.queue().launch(
        server_name="127.0.0.1",
        theme=THEME,
        css=CSS,
        head=HEAD,
        js=FORCE_DARK_JS,
    )
