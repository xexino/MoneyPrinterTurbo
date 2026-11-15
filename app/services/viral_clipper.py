"""Viral video clipper: turn a long online video into vertical short-form clips.

Usage (from the project root)::

    uv run --python 3.11 python -m app.services.viral_clipper --url "VIDEO_URL" --count 2 --layout both
    uv run --python 3.11 python -m app.services.viral_clipper --file path/to/video.mp4 --count 2

Pipeline:
    1. Download the source with yt-dlp (or use a local file).
    2. Transcribe with faster-whisper, keeping word-level timestamps.
    3. Pick the strongest, non-overlapping moments (LLM if configured, else heuristics).
    4. Render 1080x1920 clips with ffmpeg in "full" and/or "split" layouts, burning in
       word-by-word yellow (#FFD700) subtitles with a spring "pop" animation (ASS/libass).
    5. Verify every MP4 (ffprobe + yellow-pixel check on a subtitle frame).
    6. Write VIRAL_PACK.md and append to the clipper history (Markdown + JSON).

Everything is written under ``<PROJECT_ROOT>/storage/clipper/``; no machine-specific
paths are used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIPPER_ROOT = PROJECT_ROOT / "storage" / "clipper"
SESSIONS_DIR = CLIPPER_ROOT / "sessions"
HISTORY_MD = CLIPPER_ROOT / "CONVERSATION_AND_CLIP_HISTORY.md"
HISTORY_JSON = CLIPPER_ROOT / "CONVERSATION_AND_CLIP_HISTORY.json"
FONTS_DIR = PROJECT_ROOT / "resource" / "fonts"

OUT_W, OUT_H, OUT_FPS = 1080, 1920, 30
SUB_COLOR_HEX = "#FFD700"
# ASS colours are &HAABBGGRR -> #FFD700 = R:FF G:D7 B:00
SUB_COLOR_ASS = "&H0000D7FF"
SUB_FONT = "Be Vietnam Pro"

PLATFORM_HOSTS = {
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "tiktok.com": "TikTok",
    "instagram.com": "Instagram",
    "twitter.com": "X/Twitter",
    "x.com": "X/Twitter",
    "twitch.tv": "Twitch",
    "facebook.com": "Facebook",
    "fb.watch": "Facebook",
    "vimeo.com": "Vimeo",
    "reddit.com": "Reddit",
    "dailymotion.com": "Dailymotion",
    "kick.com": "Kick",
    "rumble.com": "Rumble",
    "ted.com": "TED",
}

SENSITIVE_QUERY_KEYS = re.compile(
    r"(token|key|sig|signature|auth|session|password|pass|secret|cookie|expires|policy)",
    re.I,
)

HOOK_WORDS = {
    "secret", "never", "always", "nobody", "everyone", "truth", "crazy", "insane",
    "why", "how", "what", "mistake", "wrong", "best", "worst", "money", "million",
    "billion", "stop", "listen", "imagine", "actually", "honestly", "literally",
    "shocking", "biggest", "first", "last", "only", "real", "hate", "love", "fear",
    "rich", "broke", "fail", "win", "lose", "changed", "life", "you", "your",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on",
    "for", "with", "at", "by", "from", "is", "are", "was", "were", "be", "been", "it",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "me",
    "my", "your", "our", "their", "his", "her", "its", "not", "no", "yes", "do",
    "does", "did", "have", "has", "had", "just", "like", "really", "very", "can",
    "will", "would", "could", "should", "what", "when", "where", "who", "how", "why",
    "there", "here", "about", "into", "out", "up", "down", "all", "any", "some",
    "more", "most", "one", "know", "think", "going", "get", "got", "yeah", "um",
    "uh", "oh", "okay", "gonna", "wanna", "it's", "i'm", "that's", "don't", "dont",
    "im", "thats", "because", "also", "than", "too", "them", "him", "us", "said",
    "say", "see", "want", "make", "right", "well", "thing", "things", "people",
    # filler / intensifiers: good hook signals, useless as topic keywords
    "actually", "honestly", "literally", "basically", "everything", "something",
    "anything", "nothing", "forever", "still", "never", "always", "after", "before",
    "today", "start", "every", "even", "much", "many", "only", "being", "these",
}


# --------------------------------------------------------------------------- data


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class ClipPlan:
    index: int
    start: float
    end: float
    score: float
    text: str
    hook: str = ""
    titles: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class RenderedFile:
    clip_index: int
    layout: str
    path: str
    relative_path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    has_audio: bool = False
    subtitles_verified: bool | None = None
    verification_note: str = ""


@dataclass
class SessionResult:
    timestamp: str
    user_request: str
    source_url: str
    platform: str
    video_title: str
    status: str = "failed"
    session_dir: str = ""
    clips: list[dict] = field(default_factory=list)
    files: list[RenderedFile] = field(default_factory=list)
    viral_pack: str = ""
    transcription: str = ""
    selection_method: str = ""
    viral_pack_method: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------ helpers


def log(msg: str) -> None:
    print(f"[viral_clipper] {msg}", file=sys.stderr, flush=True)


def rel(path: Path | str) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError("ffmpeg not found (install ffmpeg or imageio-ffmpeg)") from exc


def ffprobe_bin() -> str | None:
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    sibling = Path(ffmpeg_bin()).with_name(
        "ffprobe.exe" if os.name == "nt" else "ffprobe"
    )
    return str(sibling) if sibling.exists() else None


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"command failed ({cmd[0]}): {tail}")
    return proc


def sanitize_url(url: str) -> str:
    """Drop query parameters that look like credentials before storing a URL."""
    try:
        parts = urlparse(url)
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                 if not SENSITIVE_QUERY_KEYS.search(k)]
        return urlunparse(parts._replace(query=urlencode(query)))
    except Exception:
        return url


def detect_platform(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    for domain, name in PLATFORM_HOSTS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return host or "unknown"


def new_session_dir() -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    base = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate, n = SESSIONS_DIR / base, 1
    while candidate.exists():
        n += 1
        candidate = SESSIONS_DIR / f"{base}_{n}"
    candidate.mkdir(parents=True)
    return candidate


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


# ----------------------------------------------------------------------- download


def download_video(url: str, dest: Path) -> tuple[Path, dict]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed (uv sync installs it)") from exc

    dest.mkdir(parents=True, exist_ok=True)
    opts = {
        "outtmpl": str(dest / "source.%(ext)s"),
        "format": "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info.get("_type") == "playlist" and info.get("entries"):
            info = next(e for e in info["entries"] if e)
    files = sorted(dest.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    files = [f for f in files if f.suffix not in {".part", ".ytdl", ".json"}]
    if not files:
        raise RuntimeError("download finished but no media file was produced")
    meta = {
        "title": info.get("title") or "Untitled video",
        "uploader": info.get("uploader") or info.get("channel") or "",
        "duration": info.get("duration"),
        "extractor": info.get("extractor_key") or "",
        "webpage_url": info.get("webpage_url") or url,
    }
    return files[0], meta


def probe(path: Path) -> dict:
    exe = ffprobe_bin()
    if not exe:
        return {}
    out = run([exe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    return json.loads(out.stdout or "{}")


def media_duration(path: Path) -> float:
    info = probe(path)
    try:
        return float(info.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        return 0.0


# -------------------------------------------------------------------- transcribe


def transcribe(video: Path, workdir: Path, model_size: str | None) -> list[Word]:
    from faster_whisper import WhisperModel

    try:
        from app.config import config

        wcfg = config.whisper
    except Exception:
        wcfg = {}
    size = model_size or wcfg.get("model_size") or "small"
    device = wcfg.get("device", "cpu")
    compute_type = wcfg.get("compute_type", "int8")
    local = PROJECT_ROOT / "models" / f"whisper-{size}"
    model_path = str(local) if (local / "model.bin").is_file() else size

    audio = workdir / "audio_16k.wav"
    run([ffmpeg_bin(), "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(audio)])

    log(f"transcribing with whisper '{model_path}' ({device}/{compute_type})")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(str(audio), word_timestamps=True, vad_filter=True)
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            text = (w.word or "").strip()
            if text:
                words.append(Word(float(w.start), float(w.end), text))
    audio.unlink(missing_ok=True)
    return words


def load_words_json(path: Path) -> list[Word]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Word(float(w["start"]), float(w["end"]), str(w["text"]).strip())
            for w in data if str(w.get("text", "")).strip()]


# ---------------------------------------------------------------- clip selection


def sentences_from_words(words: list[Word]) -> list[list[Word]]:
    sents, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        gap = words[i + 1].start - w.end if i + 1 < len(words) else 99
        if re.search(r"[.!?]['\")]?$", w.text) or gap > 1.2:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def keywords_for(text: str, n: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    for tok in tokenize(text):
        if len(tok) > 3 and tok not in STOPWORDS and not tok.isdigit():
            counts[tok] = counts.get(tok, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def score_window(sents: list[list[Word]]) -> float:
    words = [w for s in sents for w in s]
    if not words:
        return 0.0
    text = " ".join(w.text for w in words)
    toks = tokenize(text)
    dur = max(1.0, words[-1].end - words[0].start)
    pace = len(words) / dur  # words per second, energetic speech ~2.5-3.5
    first = " ".join(w.text for w in sents[0]).lower()
    hook = sum(1 for t in tokenize(first) if t in HOOK_WORDS) * 1.5
    hook += 2.0 if "?" in first else 0.0
    emphasis = sum(1 for t in toks if t in HOOK_WORDS) / max(1, len(toks)) * 20
    numbers = min(3, len(re.findall(r"\d", text))) * 0.5
    exclaim = text.count("!") * 0.5 + text.count("?") * 0.4
    ends_clean = 1.0 if re.search(r"[.!?]['\")]?$", words[-1].text) else 0.0
    return round(pace * 1.2 + hook + emphasis + numbers + exclaim + ends_clean, 3)


def heuristic_select(words: list[Word], count: int, min_s: float, max_s: float,
                     total: float) -> list[ClipPlan]:
    sents = sentences_from_words(words)
    candidates: list[ClipPlan] = []
    for i in range(len(sents)):
        for j in range(i, len(sents)):
            start, end = sents[i][0].start, sents[j][-1].end
            dur = end - start
            if dur > max_s:
                break
            if dur >= min_s:
                window = sents[i:j + 1]
                text = " ".join(w.text for s in window for w in s)
                candidates.append(ClipPlan(0, max(0.0, start - 0.15),
                                           min(total or end + 0.4, end + 0.4),
                                           score_window(window), text))
    if not candidates:  # short video or sparse speech: fall back to even windows
        return even_windows(words, count, min_s, max_s, total)
    candidates.sort(key=lambda c: -c.score)
    chosen: list[ClipPlan] = []
    for c in candidates:
        if all(c.end <= o.start or c.start >= o.end for o in chosen):
            chosen.append(c)
        if len(chosen) == count:
            break
    chosen.sort(key=lambda c: c.start)
    for i, c in enumerate(chosen, 1):
        c.index = i
    return chosen


def even_windows(words: list[Word], count: int, min_s: float, max_s: float,
                 total: float) -> list[ClipPlan]:
    total = total or (words[-1].end if words else 0)
    length = min(max_s, max(min_s, total / max(1, count)))
    length = min(length, total / max(1, count)) if total else length
    plans = []
    for i in range(count):
        start = i * (total / count) if total else i * length
        end = min(total, start + length) if total else start + length
        if end - start < 3:
            continue
        text = " ".join(w.text for w in words if start <= w.start < end)
        plans.append(ClipPlan(len(plans) + 1, start, end, 0.0, text))
    return plans


def llm_available() -> bool:
    try:
        from app.config import config
        from app.models.llm_provider import get_llm_provider

        provider_id = str(config.app.get("llm_provider", "")).lower()
        provider = get_llm_provider(provider_id)
        if provider is None:
            return False
        key = config.app.get(provider.config_key("api_key"), "")
        return bool(key) or provider_id in {"ollama", "claude_code", "codex", "chatgpt"}
    except Exception:
        return False


def llm_json(prompt: str):
    from app.services import llm

    raw = llm._generate_response(prompt)
    if not raw or raw.startswith("Error"):
        raise RuntimeError(raw or "empty LLM response")
    raw = llm._strip_code_fence(raw)
    m = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
    return json.loads(m.group(1) if m else raw)


def llm_select(words: list[Word], count: int, min_s: float, max_s: float,
               total: float, title: str) -> list[ClipPlan]:
    lines = []
    for s in sentences_from_words(words):
        lines.append(f"[{s[0].start:.1f}-{s[-1].end:.1f}] " + " ".join(w.text for w in s))
    transcript = "\n".join(lines)[:24000]
    prompt = (
        "You are a viral short-form video editor. From the timestamped transcript of "
        f"the video \"{title}\", choose the {count} most viral, self-contained moments. "
        f"Each must be between {min_s:.0f} and {max_s:.0f} seconds, must not overlap, "
        "must start on a strong hook and end on a complete thought. Use the given "
        "timestamps. Reply with ONLY JSON: {\"clips\": [{\"start\": float, \"end\": "
        "float, \"hook\": str, \"titles\": [str, str, str]}]}\n\nTRANSCRIPT:\n" + transcript
    )
    data = llm_json(prompt)
    plans: list[ClipPlan] = []
    for item in (data.get("clips") if isinstance(data, dict) else data) or []:
        start, end = float(item["start"]), float(item["end"])
        end = min(end, total or end)
        if end - start < max(5.0, min_s * 0.5) or end - start > max_s * 1.3:
            continue
        if any(not (end <= p.start or start >= p.end) for p in plans):
            continue
        text = " ".join(w.text for w in words if start <= w.start < end)
        plans.append(ClipPlan(0, max(0.0, start - 0.1), min(total or end + 0.3, end + 0.3),
                              0.0, text, hook=str(item.get("hook", "")),
                              titles=[str(t) for t in item.get("titles", [])][:3]))
        if len(plans) == count:
            break
    if not plans:
        raise RuntimeError("LLM returned no usable clips")
    plans.sort(key=lambda c: c.start)
    for i, p in enumerate(plans, 1):
        p.index = i
    return plans


# --------------------------------------------------------------------- subtitles


def ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def chunk_words(words: list[Word], max_words: int = 3, max_chars: int = 18) -> list[list[Word]]:
    chunks, cur = [], []
    for i, w in enumerate(words):
        if cur and (len(cur) >= max_words
                    or len(" ".join(x.text for x in cur + [w])) > max_chars
                    or w.start - cur[-1].end > 0.6):
            chunks.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?,;:]$", w.text):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def build_ass(words: list[Word], layout: str, path: Path) -> int:
    """Write an ASS file with word-by-word yellow captions and a spring pop.

    Each word gets its own event; the active word overshoots (60% -> 125% -> 92% ->
    100%) like a damped spring, while the rest of the short phrase stays visible.
    Returns the number of dialogue events written.
    """
    if layout == "split":
        font_size, margin_v, alignment = 90, OUT_H // 2 - 50, 2  # straddles the seam
    else:
        font_size, margin_v, alignment = 104, 520, 2  # lower-middle, clear of UI chrome
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Viral,{SUB_FONT},{font_size},{SUB_COLOR_ASS},{SUB_COLOR_ASS},&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,7,3,{alignment},60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    spring = (r"{\fscx60\fscy60\t(0,70,\fscx125\fscy125)\t(70,140,\fscx92\fscy92)"
              r"\t(140,210,\fscx100\fscy100)}")
    settle = r"{\fscx100\fscy100}"
    events = []
    chunks = chunk_words(words)
    for ci, chunk in enumerate(chunks):
        next_start = chunks[ci + 1][0].start if ci + 1 < len(chunks) else None
        chunk_end = chunk[-1].end + 0.25
        if next_start is not None:
            chunk_end = min(chunk_end, next_start)
        for wi, w in enumerate(chunk):
            start = w.start
            end = chunk[wi + 1].start if wi + 1 < len(chunk) else chunk_end
            if end - start < 0.05:
                end = start + 0.05
            parts = []
            for k, other in enumerate(chunk):
                token = ass_escape(other.text.upper())
                parts.append(f"{spring}{token}{settle}" if k == wi else token)
            events.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Viral,,0,0,0,,{' '.join(parts)}"
            )
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return len(events)


# ------------------------------------------------------------------------ render


def filter_graph(layout: str, ass_name: str | None, fonts_rel: str) -> str:
    if layout == "split":
        half = OUT_H // 2
        graph = (
            "[0:v]split=2[a][b];"
            f"[a]scale={OUT_W}:{half}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{half},setsar=1[top];"
            # Bottom panel: tighter punch-in, biased upward so heads in wide shots stay in frame.
            f"[b]crop=iw/1.8:ih/1.8:(iw-iw/1.8)/2:(ih-ih/1.8)/5,"
            f"scale={OUT_W}:{half}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{half},setsar=1[bot];"
            "[top][bot]vstack=inputs=2,"
            f"drawbox=x=0:y={half - 3}:w={OUT_W}:h=6:color=black@0.9:t=fill"
        )
    else:
        graph = (
            f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},setsar=1"
        )
    graph += f",fps={OUT_FPS}"
    if ass_name:
        graph += f",ass={ass_name}:fontsdir={fonts_rel}"
    return graph + "[v]"


def render_clip(src: Path, plan: ClipPlan, layout: str, words: list[Word],
                out_dir: Path, has_audio: bool) -> tuple[Path, int]:
    out = out_dir / f"clip_{plan.index:02d}_{layout}.mp4"
    clip_words = [Word(w.start - plan.start, w.end - plan.start, w.text)
                  for w in words if w.start >= plan.start and w.end <= plan.end + 0.05]
    ass_name, n_events = None, 0
    if clip_words:
        ass_name = f"clip_{plan.index:02d}_{layout}.ass"
        n_events = build_ass(clip_words, layout, out_dir / ass_name)
    # Run ffmpeg from the session directory with relative paths so the ass filter
    # never has to escape drive letters or spaces (portable across OSes).
    fonts_rel = os.path.relpath(FONTS_DIR, out_dir).replace("\\", "/")
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{plan.start:.3f}", "-t", f"{plan.duration:.3f}", "-i", str(src),
           "-filter_complex", filter_graph(layout, ass_name, fonts_rel),
           "-map", "[v]"]
    if has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", out.name]
    run(cmd, cwd=out_dir)
    return out, n_events


def yellow_pixel_ratio(video: Path, t: float, region: tuple[int, int]) -> float:
    """Fraction of pixels close to #FFD700 in a horizontal band at time t."""
    y0, y1 = region
    proc = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
         "-vf", f"crop={OUT_W}:{y1 - y0}:0:{y0},scale={OUT_W // 4}:-1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    )
    data = proc.stdout
    if not data:
        return 0.0
    hits = 0
    for i in range(0, len(data) - 2, 3):
        r, g, b = data[i], data[i + 1], data[i + 2]
        if r > 200 and 170 < g < 235 and b < 90:
            hits += 1
    return hits / (len(data) / 3)


def verify_file(rf: RenderedFile, plan: ClipPlan, words: list[Word], layout: str) -> None:
    path = Path(rf.path)
    if not path.is_file() or path.stat().st_size < 10_000:
        raise RuntimeError(f"{path.name} missing or too small")
    info = probe(path)
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    rf.has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    if vstreams:
        rf.width, rf.height = int(vstreams[0]["width"]), int(vstreams[0]["height"])
    rf.duration = round(float(info.get("format", {}).get("duration") or 0), 2)
    if (rf.width, rf.height) != (OUT_W, OUT_H) or rf.duration <= 0:
        raise RuntimeError(f"{path.name}: unexpected stream {rf.width}x{rf.height} {rf.duration}s")
    # Subtitle check: sample frames well after a word pops in (spring settled).
    clip_words = [w for w in words if w.start >= plan.start and w.end <= plan.end]
    if not clip_words:
        rf.subtitles_verified = None
        rf.verification_note = "no speech detected; no subtitles burned"
        return
    band = (OUT_H // 2 - 330, OUT_H // 2 + 120) if layout == "split" else (1100, 1560)
    samples = [clip_words[len(clip_words) * k // 4] for k in (0, 1, 2, 3)]
    ratios = [yellow_pixel_ratio(path, (w.start - plan.start) + 0.25, band) for w in samples]
    best = max(ratios)
    rf.subtitles_verified = best > 0.004
    rf.verification_note = f"yellow #FFD700 pixel ratio in caption band: {best:.3%}"


# ------------------------------------------------------------------- viral pack


PLATFORM_TAGS = {
    "TikTok": ["#fyp", "#foryou", "#viral", "#tiktok"],
    "Instagram Reels": ["#reels", "#reelsinstagram", "#explore", "#viral"],
    "Snapchat Spotlight": ["#spotlight", "#snapchat", "#viral"],
    "YouTube Shorts": ["#shorts", "#youtubeshorts", "#viral"],
    "Facebook Reels": ["#reels", "#facebookreels", "#viral"],
}


def first_sentence(text: str, limit: int = 110) -> str:
    m = re.split(r"(?<=[.!?])\s+", text.strip())
    s = (m[0] if m else text).strip()
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "..."


def template_copy(plan: ClipPlan, title: str) -> dict:
    kws = plan.keywords or keywords_for(plan.text) or keywords_for(title) or ["this"]
    k1 = kws[0].capitalize()
    hook = plan.hook or (first_sentence(plan.text) if plan.text else f"Wait for the end of this {k1} moment...")
    titles = plan.titles or [
        f"Nobody Talks About This {k1} Moment",
        f"{first_sentence(plan.text, 60) or title[:60]}",
        f"The {k1} Clip Everyone Needs To See",
    ]
    tags = ["#" + re.sub(r"[^a-z0-9]", "", k) for k in kws[:4] if re.sub(r"[^a-z0-9]", "", k)]
    captions = {}
    for platform, ptags in PLATFORM_TAGS.items():
        if platform == "TikTok":
            body = f"{hook} 🤯 Watch till the end 👀"
        elif platform == "Instagram Reels":
            body = f"{hook}\n\nSave this & share it with someone who needs it 💡"
        elif platform == "Snapchat Spotlight":
            body = f"{first_sentence(hook, 70)} 😳"
        elif platform == "YouTube Shorts":
            body = f"{titles[0]} | {hook}"
        else:
            body = f"{hook}\n\nWhat do you think? Comment below 👇"
        captions[platform] = {"caption": body, "hashtags": tags + ptags}
    return {"hook": hook, "titles": titles[:3], "captions": captions}


def llm_copy(plan: ClipPlan, title: str) -> dict:
    prompt = (
        "Write viral short-form social copy for this clip from the video "
        f"\"{title}\". Clip transcript:\n\"\"\"{plan.text[:3000]}\"\"\"\n\n"
        "Reply with ONLY JSON: {\"hook\": str (on-screen first-2-seconds hook), "
        "\"titles\": [3 alternative titles], \"captions\": {\"TikTok\": {\"caption\": "
        "str, \"hashtags\": [str]}, \"Instagram Reels\": {...}, \"Snapchat Spotlight\": "
        "{...}, \"YouTube Shorts\": {...}, \"Facebook Reels\": {...}}}. Hashtags start "
        "with #, 5-8 per platform. Never invent facts that are not in the transcript."
    )
    data = llm_json(prompt)
    if not isinstance(data, dict) or not data.get("captions"):
        raise RuntimeError("LLM copy missing captions")
    base = template_copy(plan, title)
    base["hook"] = str(data.get("hook") or base["hook"])
    if data.get("titles"):
        base["titles"] = [str(t) for t in data["titles"]][:3]
    for platform in PLATFORM_TAGS:
        c = data["captions"].get(platform)
        if isinstance(c, dict) and c.get("caption"):
            base["captions"][platform] = {
                "caption": str(c["caption"]),
                "hashtags": [h if str(h).startswith("#") else f"#{h}" for h in c.get("hashtags", [])],
            }
    return base


def write_viral_pack(result: SessionResult, plans: list[ClipPlan], copies: dict[int, dict],
                     session: Path) -> Path:
    files_by_clip: dict[int, list[RenderedFile]] = {}
    for f in result.files:
        files_by_clip.setdefault(f.clip_index, []).append(f)
    lines = [
        "# VIRAL PACK",
        "",
        f"- **Original video title:** {result.video_title}",
        f"- **Source:** {result.source_url + ' (' + result.platform + ')' if result.source_url else 'local file'}",
        f"- **Generated:** {result.timestamp}",
        f"- **Clip selection:** {result.selection_method}",
        f"- **Copywriting:** {result.viral_pack_method}",
        f"- **Subtitles:** word-by-word, {SUB_COLOR_HEX} yellow, spring pop animation, bold {SUB_FONT}",
        "",
    ]
    for plan in plans:
        copy = copies[plan.index]
        lines += [
            f"## Clip {plan.index:02d} — {fmt_ts(plan.start)} → {fmt_ts(plan.end)} "
            f"({plan.duration:.0f}s)",
            "",
            "**Files:**",
        ]
        for f in files_by_clip.get(plan.index, []):
            lines.append(f"- `{Path(f.path).name}` ({f.layout}-screen, {f.width}x{f.height}, {f.duration:.1f}s)")
        lines += ["", f"**Viral hook (on-screen, first 2s):** {copy['hook']}", "",
                  "**Alternative titles:**"]
        lines += [f"{i}. {t}" for i, t in enumerate(copy["titles"], 1)]
        lines.append("")
        for platform, c in copy["captions"].items():
            lines += [f"### {platform}", "", c["caption"], "", " ".join(c["hashtags"]), ""]
        if plan.text:
            excerpt = plan.text if len(plan.text) < 600 else plan.text[:600] + "..."
            lines += ["<details><summary>Transcript excerpt</summary>", "", excerpt, "",
                      "</details>", ""]
    lines += [
        "## Recommended posting strategy",
        "",
        "1. **Post the full-screen version first** on TikTok and Instagram Reels; use the "
        "split-screen version on YouTube Shorts and Facebook Reels so each platform gets "
        "a visually distinct upload.",
        "2. **Stagger releases:** one clip per day, 24h apart, so each gets a clean "
        "distribution window. Post Clip 01 first, then Clip 02.",
        "3. **Timing:** test 11:00–13:00 and 19:00–22:00 in your audience's local time; keep "
        "whichever slot wins after a week.",
        "4. **First 60 minutes:** reply to every comment and pin the best one; ask a "
        "question in the caption to drive replies.",
        "5. **Hook on screen:** use the hook text above as the cover/first-frame text and "
        "the first title as the thumbnail line.",
        "6. **Rights:** only post clips you own or have permission to repost; credit the "
        "original creator in the caption.",
        "",
    ]
    out = session / "VIRAL_PACK.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ----------------------------------------------------------------------- history


def append_history(result: SessionResult) -> None:
    CLIPPER_ROOT.mkdir(parents=True, exist_ok=True)
    entry = asdict(result)
    entry["files"] = [asdict(f) if not isinstance(f, dict) else f for f in result.files]

    history: list = []
    if HISTORY_JSON.exists():
        try:
            history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = [history]
        except json.JSONDecodeError:
            backup = HISTORY_JSON.with_suffix(
                f".corrupt-{datetime.now():%Y%m%d%H%M%S}.json")
            HISTORY_JSON.replace(backup)
            result.warnings.append(f"history JSON was unreadable; preserved as {backup.name}")
            history = []
    history.append(entry)
    tmp = HISTORY_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(HISTORY_JSON)

    if not HISTORY_MD.exists():
        HISTORY_MD.write_text(
            "# Conversation and Clip History\n\nAppend-only log of every clipping "
            "request processed by the viral clipper.\n", encoding="utf-8")
    md = [
        "",
        f"## {result.timestamp} — {result.status.upper()}",
        "",
        f"- **Request:** {result.user_request}",
        f"- **Source URL:** {result.source_url or 'n/a'}",
        f"- **Platform:** {result.platform}",
        f"- **Video title:** {result.video_title or 'unknown (source not retrieved)'}",
        f"- **Session:** `{rel(result.session_dir) if result.session_dir else 'n/a'}`",
    ]
    if result.files:
        md.append("- **Generated clips:**")
        for f in result.files:
            sub = {True: "subtitles verified", False: "subtitles NOT detected",
                   None: "no subtitles"}[f.subtitles_verified]
            md.append(f"  - `{f.relative_path}` ({f.layout}, {f.duration:.1f}s, {sub})")
    md.append(f"- **Viral pack:** `{rel(result.viral_pack)}`" if result.viral_pack
              else "- **Viral pack:** not generated")
    for w in result.warnings:
        md.append(f"- ⚠️ Warning: {w}")
    for e in result.errors:
        md.append(f"- ❌ Error: {e}")
    with HISTORY_MD.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")


# --------------------------------------------------------------------------- main


def process(args: argparse.Namespace) -> SessionResult:
    url = sanitize_url(args.url) if args.url else ""
    result = SessionResult(
        timestamp=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        user_request=args.request or (f"Clip {url}" if url else f"Clip local file {args.file}"),
        source_url=url,
        platform=detect_platform(url) if url else "local file",
        video_title="",
    )
    try:
        session = new_session_dir()
        result.session_dir = str(session)
        layouts = ["full", "split"] if args.layout == "both" else [args.layout]

        # 1-2. identify + fetch source
        with tempfile.TemporaryDirectory(prefix="clipper_src_", dir=session) as tmp:
            tmpdir = Path(tmp)
            if args.file:
                src = Path(args.file).expanduser().resolve()
                if not src.is_file():
                    raise FileNotFoundError(f"input file not found: {args.file}")
                result.video_title = args.title or src.stem.replace("_", " ")
            else:
                log(f"downloading {url}")
                src, meta = download_video(args.url, tmpdir)
                result.video_title = meta["title"]
                if meta.get("extractor"):
                    result.platform = PLATFORM_HOSTS.get(
                        meta["extractor"].lower(), result.platform)
            log(f"source: {result.video_title}")

            info = probe(src)
            total = float(info.get("format", {}).get("duration") or 0)
            has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
            if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
                raise RuntimeError("source has no video stream")

            # 3. transcript
            words: list[Word] = []
            if args.words_json:
                words = load_words_json(Path(args.words_json))
                result.transcription = f"provided word timings ({args.words_json})"
            elif has_audio:
                try:
                    words = transcribe(src, tmpdir, args.whisper_model)
                    result.transcription = f"faster-whisper ({len(words)} words)"
                except Exception as exc:
                    result.warnings.append(
                        f"transcription unavailable, clips will have no subtitles: {exc}")
                    result.transcription = "failed"
            else:
                result.warnings.append("source has no audio track; no subtitles")
                result.transcription = "no audio"

            # 4. choose moments
            min_s, max_s = args.min_seconds, args.max_seconds
            if total and total < min_s * args.count:
                min_s = max(5.0, total / args.count * 0.8)
                max_s = max(min_s + 1, total / args.count)
            plans: list[ClipPlan] = []
            if words and not args.no_llm and llm_available():
                try:
                    plans = llm_select(words, args.count, min_s, max_s, total, result.video_title)
                    result.selection_method = "LLM ranking of transcript"
                except Exception as exc:
                    result.warnings.append(f"LLM clip selection failed, used heuristics: {exc}")
            if not plans:
                if words:
                    plans = heuristic_select(words, args.count, min_s, max_s, total)
                    result.selection_method = ("heuristic scoring (hook words, pace, "
                                               "questions, clean endings)")
                else:
                    plans = even_windows(words, args.count, min_s, max_s, total)
                    result.selection_method = "evenly spaced windows (no transcript)"
            if len(plans) < args.count:
                result.warnings.append(
                    f"only {len(plans)} non-overlapping clip(s) fit the source duration")
            for p in plans:
                p.keywords = keywords_for(p.text)

            # 5. render
            for plan in plans:
                for layout in layouts:
                    log(f"rendering clip {plan.index:02d} [{layout}] "
                        f"{fmt_ts(plan.start)}-{fmt_ts(plan.end)}")
                    out, n_events = render_clip(src, plan, layout, words, session, has_audio)
                    rf = RenderedFile(plan.index, layout, str(out), rel(out))
                    verify_file(rf, plan, words, layout)
                    if rf.subtitles_verified is False:
                        result.warnings.append(
                            f"{out.name}: subtitle colour not detected in sampled frames "
                            f"({rf.verification_note})")
                    result.files.append(rf)
                if not args.keep_ass:
                    for a in session.glob(f"clip_{plan.index:02d}_*.ass"):
                        a.unlink()

            result.clips = [
                {"index": p.index, "start": round(p.start, 2), "end": round(p.end, 2),
                 "duration": round(p.duration, 2), "score": p.score,
                 "files": [f.relative_path for f in result.files if f.clip_index == p.index]}
                for p in plans
            ]

        # 6. viral pack
        copies: dict[int, dict] = {}
        use_llm = not args.no_llm and llm_available()
        method = "LLM-written" if use_llm else "template-based from transcript (no LLM configured)"
        for plan in plans:
            copy = None
            if use_llm:
                try:
                    copy = llm_copy(plan, result.video_title)
                except Exception as exc:
                    result.warnings.append(f"LLM copy for clip {plan.index} failed: {exc}")
                    method = "mixed: LLM with template fallback"
            copies[plan.index] = copy or template_copy(plan, result.video_title)
        result.viral_pack_method = method
        result.viral_pack = str(write_viral_pack(result, plans, copies, session))
        result.status = "success" if result.files else "failed"
        if result.files and any(f.subtitles_verified is not True for f in result.files):
            result.status = "partial"
    except Exception as exc:
        result.errors.append(f"{type(exc).__name__}: {exc}")
        if args.debug:
            traceback.print_exc()
        result.status = "partial" if result.files else "failed"
    finally:
        # Don't leave empty session folders behind for requests that produced nothing.
        sd = Path(result.session_dir) if result.session_dir else None
        if sd and sd.is_dir() and not any(sd.iterdir()):
            sd.rmdir()
            result.session_dir = ""
        append_history(result)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Turn an online video into viral vertical clips.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="video URL (YouTube, TikTok, Instagram, X, Twitch, ...)")
    src.add_argument("--file", help="local video file instead of a URL")
    p.add_argument("--count", type=int, default=2, help="number of clips (default 2)")
    p.add_argument("--layout", choices=["full", "split", "both"], default="both")
    p.add_argument("--min-seconds", type=float, default=20.0)
    p.add_argument("--max-seconds", type=float, default=55.0)
    p.add_argument("--whisper-model", help="faster-whisper model size (default: config or 'small')")
    p.add_argument("--words-json", help="use pre-computed word timings [{start,end,text}]")
    p.add_argument("--title", help="title override (local files)")
    p.add_argument("--request", help="original user request, stored in the history")
    p.add_argument("--no-llm", action="store_true", help="never call the configured LLM")
    p.add_argument("--keep-ass", action="store_true", help="keep generated .ass subtitle files")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.count < 1:
        p.error("--count must be >= 1")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = process(args)
    summary = {
        "status": result.status,
        "video_title": result.video_title,
        "platform": result.platform,
        "session_dir": rel(result.session_dir) if result.session_dir else "",
        "files": [{"path": f.relative_path, "layout": f.layout, "duration": f.duration,
                   "resolution": f"{f.width}x{f.height}", "audio": f.has_audio,
                   "subtitles_verified": f.subtitles_verified,
                   "note": f.verification_note} for f in result.files],
        "viral_pack": rel(result.viral_pack) if result.viral_pack else "",
        "history": [rel(HISTORY_MD), rel(HISTORY_JSON)],
        "warnings": result.warnings,
        "errors": result.errors,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if result.status in {"success", "partial"} else 1


if __name__ == "__main__":
    sys.exit(main())
