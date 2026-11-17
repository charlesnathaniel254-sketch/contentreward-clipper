# ContentReward Clipper

A **Crayo.ai-style AI clip studio** for [ContentReward.com](https://contentreward.com):
paste a long-form video, pick the best moment, and get a **vertical 9:16 clip with
dynamic TikTok-style captions** — ready to post to TikTok, Instagram Reels and
YouTube Shorts as part of a ContentReward campaign.

## How it works

```
Video URL ──► yt-dlp ──► Whisper ──► ASS captions ──► FFmpeg ──► 1080×1920 MP4
              download     word-level   active word      crop / blur-pad
              (section)    timestamps   highlighted      + burned captions
```

| Stage | Tool | What happens |
|---|---|---|
| 1. Download | `yt-dlp` | Downloads **only the section you asked for** (capped at 1080p) |
| 2. Transcribe | OpenAI `whisper` | Speech-to-text with **word-level timestamps** (`word_timestamps=True`) |
| 3. Captions | generated `.ass` file | Words grouped 3–4 at a time; the spoken word is highlighted & "pops" |
| 4. Render | `FFmpeg` | Crops or blur-pads to **9:16 (1080×1920)** and burns captions in |

## Prerequisites

- **Python 3.10+**
- **FFmpeg** on your `PATH` (required by both Whisper and the renderer):
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to PATH

  Check it with `ffmpeg -version`.
- Internet access (for video downloads and the one-time Whisper model download).

## Quick start

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies (note: Whisper installs PyTorch — large, ~2 GB)
pip install -r requirements.txt

# 3. Run the web app
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

> The first clip is slower: Whisper downloads its model on first use
> (~145 MB for the default `base` model) and loads it into memory.

## Using the web studio

1. Paste a video URL (YouTube or any other [yt-dlp-supported site](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)).
2. Set **Start** and **End** — or Start + **length in seconds**.
   Formats: `90`, `1:30`, or `01:02:03`. Keep it under ~60 s for Shorts/Reels.
3. Pick a **caption style** (TikTok yellow highlight, Karaoke green, Boxed, Minimal).
4. Pick a **9:16 fill mode**:
   - **Blur pad** (default) — keeps the full frame, fills the sides with a blurred background. Best for landscape videos.
   - **Centre crop** — fills the whole 9:16 frame by cropping. Best when the subject is centred.
5. Click **Create clip**, watch live progress, preview the result in the browser, then **Download MP4**.
6. Upload the MP4 to your campaign on **ContentReward.com** and post it to TikTok / Reels / Shorts.

## Command-line usage

The pipeline also works without the web UI:

```bash
python processor.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --start 00:01:00 --end 00:01:45 \
    --style tiktok --mode blur --model base
```

Options: `--start`, `--end`, `--duration` (seconds, if no `--end`),
`--model tiny|base|small|medium|large`, `--style tiktok|karaoke|boxed|minimal`,
`--mode crop|blur`, `--no-captions`, `--output-dir`, `--work-dir`.

Finished clips land in `outputs/` (the web app serves them at `/outputs/<id>.mp4`).

## API

Interactive docs: **http://localhost:8000/docs**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dark-mode web studio (Tailwind) |
| `GET` | `/api/health` | Health check + FFmpeg availability |
| `POST` | `/api/clips` | Create a clip job (returns `202` + job) |
| `GET` | `/api/jobs` | List all jobs (newest first) |
| `GET` | `/api/jobs/{id}` | Poll one job's status/progress |
| `GET` | `/api/clips/{id}/download` | Download the finished MP4 |
| `DELETE` | `/api/jobs/{id}` | Delete a job and its files |

Example:

```bash
curl -X POST http://localhost:8000/api/clips \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID","start":"1:00","end":"1:45","caption_style":"tiktok","fill_mode":"blur"}'
# → {"id":"abc123...","status":"queued",...}

curl http://localhost:8000/api/jobs/abc123...
# → {"status":"completed","progress":100,"video_url":"/outputs/abc123....mp4",...}
```

Jobs are processed **one at a time** in the background (Whisper + FFmpeg are
heavy) and stored in memory — restarting the server clears job history, though
finished files stay in `outputs/`.

## Choosing a Whisper model

| Model | Speed | Accuracy | Download |
|---|---|---|---|
| `tiny` | Fastest | Rough — good for a quick test | ~75 MB |
| `base` | Fast | Decent (default) | ~145 MB |
| `small` | Medium | Good | ~465 MB |
| `medium` | Slow | Best caption quality | ~1.5 GB |

Start with `base`; switch to `small`/`medium` if captions have many mistakes.
On an NVIDIA GPU, Whisper will use it automatically via PyTorch.

## Project structure

```
contentreward-clipper/
├── app.py            # FastAPI backend + dark-mode Tailwind frontend (served at /)
├── processor.py      # yt-dlp → Whisper → ASS captions → FFmpeg 9:16 pipeline (+ CLI)
├── requirements.txt  # Python dependencies
├── downloads/        # (generated) temporary downloaded sections — auto-cleaned
└── outputs/          # (generated) finished MP4 clips + .ass caption files
```

## Troubleshooting

- **`FFmpeg was not found on PATH`** — install FFmpeg (see Prerequisites) and restart your terminal/server.
- **`yt-dlp` / `whisper is not installed`** — run `pip install -r requirements.txt` inside your activated venv.
- **Download fails / "Video unavailable"** — update yt-dlp first: `pip install -U yt-dlp`. Some videos are age-restricted, private, or region-locked.
- **Captions look off** — use a bigger Whisper model (`small`/`medium`); very loud music or heavy accents reduce accuracy.
- **Slow processing** — shorten the clip, use `tiny`/`base`, or run on a GPU machine. Jobs run sequentially by design.
- **Port already in use** — use another port: `uvicorn app:app --port 8001`.

## Notes

- Only clip content you **own or have permission to use**, and follow the rules of
  ContentReward, the campaign brand, and each social platform.
- This is a local, single-user tool: no accounts, database, or cloud rendering.
  Job state lives in memory; rendered files live in `outputs/`.

## License

MIT — use it for your own clipping workflow.
