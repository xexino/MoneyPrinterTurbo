# Viral video clipper


When the user sends a video URL (YouTube, TikTok, Instagram, X/Twitter, Twitch, ...),
act as an autonomous clipping agent and run the engine from the project root:

```bash
uv run --python 3.11 python -m app.services.viral_clipper --url "VIDEO_URL" --count 2 --layout both --request "<the user's request>"
```

- Local file instead of a URL: `--file path/to/video.mp4` (optional `--title`).
- Other flags: `--min-seconds/--max-seconds` (default 20/55), `--whisper-model small`,
  `--no-llm`, `--keep-ass`, `--debug`. See `python -m app.services.viral_clipper -h`.
- If `uv` is missing, use any Python >= 3.11 environment with the project's dependencies.
- The command prints a JSON summary (status, files, subtitle verification, warnings,
  errors). Report only what it verified; never invent output links.

What it produces, under `storage/clipper/` (portable, relative to the project root):

- `sessions/session_YYYYMMDD_HHMMSS/clip_XX_full.mp4`: 1080x1920 full-screen crop
- `sessions/session_YYYYMMDD_HHMMSS/clip_XX_split.mp4`: 1080x1920 split screen (wide
  view on top, punched-in view below)
- `sessions/session_YYYYMMDD_HHMMSS/VIRAL_PACK.md`: hooks, 3 titles per clip, captions and
  hashtags for TikTok, Reels, Spotlight, Shorts and Facebook, and a posting strategy
- `CONVERSATION_AND_CLIP_HISTORY.md` / `.json`: append-only log of every request,
  success or failure (credential-like URL query params are stripped)

Subtitles are word-by-word, #FFD700 yellow, bold Be Vietnam Pro, with a spring pop
(60% -> 125% -> 92% -> 100%) on the active word, burned in with libass. Each output is
checked with ffprobe (1080x1920, duration, audio) and by sampling frames for yellow
caption pixels.

Requirements: ffmpeg built with libass; network access to the video site and, on the
first run, to huggingface.co to download the Whisper model (or place it in
`models/whisper-<size>/`). Clip selection and copy use the LLM configured in
`config.toml` when one is set; otherwise they fall back to transcript heuristics and
templates, and VIRAL_PACK.md says which method was used.

Only process videos the user has the right to use; don't bypass logins, DRM or
geo-blocks. Never delete previous sessions or history.
