"""
app.py — ContentReward Clipper web app.

A Crayo-style, dark-mode web studio backed by FastAPI:

* Paste a video URL, pick start/end, a caption style and a 9:16 fill mode.
* The clip is processed in the background by :mod:`processor`
  (yt-dlp -> Whisper word-level captions -> FFmpeg 9:16 render).
* Watch live progress, preview the finished vertical clip in the browser
  and download it — ready to upload to a ContentReward.com campaign.

Run it with::

    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000  (interactive API docs at /docs).
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from processor import CAPTION_STYLES, FILL_MODES, ClipProcessor, ProcessingError

# ---------------------------------------------------------------------------
# Config & shared state
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
WORK_DIR = BASE_DIR / "downloads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# One shared processor: the Whisper model is loaded once and reused.
# A lock serialises jobs — Whisper/FFmpeg are heavy and not thread-safe
# enough to run several at once on a typical machine.
processor = ClipProcessor(output_dir=OUTPUT_DIR, work_dir=WORK_DIR)
_processing_lock = threading.Lock()

# In-memory job store: {job_id: job-dict}. Swap for a DB/Redis in production.
jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(fields)


def _public_job(job: dict) -> dict:
    """Strip internal fields before returning a job to the client."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ClipRequest(BaseModel):
    url: str
    start: str = "0"
    end: Optional[str] = None
    duration: Optional[str] = None
    caption_style: str = "tiktok"
    fill_mode: str = "blur"
    whisper_model: str = "base"
    burn_captions: bool = True

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        value = (value or "").strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value

    @field_validator("caption_style")
    @classmethod
    def _check_style(cls, value: str) -> str:
        if value not in CAPTION_STYLES:
            raise ValueError(f"caption_style must be one of {sorted(CAPTION_STYLES)}")
        return value

    @field_validator("fill_mode")
    @classmethod
    def _check_fill(cls, value: str) -> str:
        if value not in FILL_MODES:
            raise ValueError(f"fill_mode must be one of {list(FILL_MODES)}")
        return value

    @field_validator("whisper_model")
    @classmethod
    def _check_model(cls, value: str) -> str:
        allowed = {"tiny", "base", "small", "medium", "large", "turbo"}
        if value not in allowed:
            raise ValueError(f"whisper_model must be one of {sorted(allowed)}")
        return value


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

def run_job(job_id: str, request: ClipRequest) -> None:
    """Execute the full clipping pipeline for one job (background task)."""

    def progress(stage: str, percent: int, message: str) -> None:
        _update_job(job_id, status="processing", stage=stage,
                    progress=percent, message=message)

    _update_job(job_id, status="processing", stage="queued",
                progress=0, message="Queued...")
    try:
        with _processing_lock:
            # Per-job Whisper model choice without losing the shared cache
            # when the model is unchanged.
            if processor.whisper_model != request.whisper_model:
                processor.whisper_model = request.whisper_model
                processor._model = None
            result = processor.process(
                request.url,
                start_time=request.start,
                end_time=request.end,
                duration=request.duration,
                caption_style=request.caption_style,
                fill_mode=request.fill_mode,
                burn_captions=request.burn_captions,
                job_id=job_id,
                callback=progress,
            )
    except ProcessingError as exc:
        _update_job(job_id, status="failed", stage="failed",
                    progress=0, message=str(exc), error=str(exc),
                    finished_at=_now())
        return
    except Exception as exc:  # never leave a job stuck in "processing"
        _update_job(job_id, status="failed", stage="failed",
                    progress=0, message=f"Unexpected error: {exc}",
                    error=str(exc), finished_at=_now())
        return

    _update_job(
        job_id,
        status="completed",
        stage="done",
        progress=100,
        message="Clip ready.",
        output_file=result.output_path.name,
        video_url=f"/outputs/{result.output_path.name}",
        download_url=f"/api/clips/{job_id}/download",
        finished_at=_now(),
    )


# ---------------------------------------------------------------------------
# FastAPI app & API
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ContentReward Clipper",
    description="Turn long videos into captioned 9:16 clips for ContentReward campaigns.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/api/health")
def health() -> dict:
    import shutil
    return {
        "status": "ok",
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "caption_styles": sorted(CAPTION_STYLES),
        "fill_modes": list(FILL_MODES),
        "jobs": len(jobs),
    }


@app.post("/api/clips", status_code=202)
def create_clip(request: ClipRequest, background_tasks: BackgroundTasks) -> dict:
    if request.end is None and request.duration is None:
        raise HTTPException(
            status_code=422,
            detail="Provide an end time or a duration for the clip.",
        )
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "message": "Queued...",
        "url": request.url,
        "start": request.start,
        "end": request.end,
        "duration": request.duration,
        "caption_style": request.caption_style,
        "fill_mode": request.fill_mode,
        "video_url": None,
        "download_url": None,
        "error": None,
        "created_at": _now(),
        "finished_at": None,
    }
    with _jobs_lock:
        jobs[job_id] = job
    background_tasks.add_task(run_job, job_id, request)
    return _public_job(job)


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    with _jobs_lock:
        ordered = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return [_public_job(j) for j in ordered]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _public_job(job)


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    with _jobs_lock:
        job = jobs.pop(job_id, None)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    # Best-effort cleanup of rendered files.
    for suffix in (".mp4", ".ass"):
        try:
            (OUTPUT_DIR / f"{job_id}{suffix}").unlink(missing_ok=True)
        except OSError:
            pass
    return None


@app.get("/api/clips/{job_id}/download")
def download_clip(job_id: str) -> FileResponse:
    with _jobs_lock:
        job = jobs.get(job_id)
    path = OUTPUT_DIR / f"{job_id}.mp4"
    if job is None or not path.exists():
        raise HTTPException(status_code=404, detail="Clip not found or not ready yet.")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"contentreward-clip-{job_id}.mp4",
    )


@app.exception_handler(ProcessingError)
def processing_error_handler(_: Request, exc: ProcessingError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Frontend (dark-mode, Tailwind via CDN) — served from `/`
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>ContentReward Clipper — AI Clip Studio</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = { darkMode: 'class', theme: { extend: {
    colors: { ink: '#0b0b12', panel: '#14141f', edge: '#262636', brand: '#a3e635', violet2: '#8b5cf6' },
    fontFamily: { sans: ['Inter','system-ui','sans-serif'] }
  }}};
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<style> ::selection{background:#a3e635;color:#0b0b12} video{background:#000} </style>
</head>
<body class="bg-ink text-zinc-100 font-sans min-h-screen antialiased">

<!-- Header -->
<header class="border-b border-edge/70 bg-ink/80 backdrop-blur sticky top-0 z-20">
  <div class="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="w-9 h-9 rounded-xl bg-gradient-to-br from-brand to-violet2 grid place-items-center text-ink font-black">✂</div>
      <div>
        <p class="font-extrabold leading-none tracking-tight">ContentReward <span class="text-brand">Clipper</span></p>
        <p class="text-[11px] text-zinc-400 mt-1">Crayo-style AI clips for ContentReward campaigns</p>
      </div>
    </div>
    <a href="/docs" class="text-xs font-semibold text-zinc-300 hover:text-brand border border-edge rounded-full px-3 py-1.5">API docs ↗</a>
  </div>
</header>

<main class="max-w-6xl mx-auto px-4 py-10 grid lg:grid-cols-5 gap-8">
  <!-- Create form -->
  <section class="lg:col-span-3">
    <h1 class="text-4xl md:text-5xl font-black tracking-tight leading-[1.05]">
      Turn long videos into<br><span class="text-transparent bg-clip-text bg-gradient-to-r from-brand to-violet2">viral 9:16 clips.</span>
    </h1>
    <p class="text-zinc-400 mt-4 max-w-xl">Paste a YouTube link, choose the moment, and get a vertical clip with dynamic TikTok-style captions — ready to post to TikTok, Reels &amp; Shorts for your ContentReward campaign.</p>

    <form id="clip-form" class="mt-8 bg-panel border border-edge rounded-3xl p-6 space-y-5 shadow-2xl shadow-black/40">
      <div>
        <label class="text-xs font-bold uppercase tracking-wider text-zinc-400" for="url">Video URL</label>
        <input id="url" required type="url" placeholder="https://www.youtube.com/watch?v=..."
          class="mt-2 w-full bg-ink border border-edge rounded-xl px-4 py-3 text-sm outline-none focus:border-brand placeholder:text-zinc-600" />
      </div>
      <div class="grid grid-cols-3 gap-3">
        <div>
          <label class="text-xs font-bold uppercase tracking-wider text-zinc-400" for="start">Start</label>
          <input id="start" value="0:00" placeholder="0:00"
            class="mt-2 w-full bg-ink border border-edge rounded-xl px-3 py-3 text-sm outline-none focus:border-brand" />
        </div>
        <div>
          <label class="text-xs font-bold uppercase tracking-wider text-zinc-400" for="end">End</label>
          <input id="end" placeholder="0:45"
            class="mt-2 w-full bg-ink border border-edge rounded-xl px-3 py-3 text-sm outline-none focus:border-brand" />
        </div>
        <div>
          <label class="text-xs font-bold uppercase tracking-wider text-zinc-400" for="duration">…or length (s)</label>
          <input id="duration" type="number" min="1" max="600" placeholder="45"
            class="mt-2 w-full bg-ink border border-edge rounded-xl px-3 py-3 text-sm outline-none focus:border-brand" />
        </div>
      </div>
      <p class="-mt-2 text-[11px] text-zinc-500">Times accept <code class="text-zinc-300">90</code>, <code class="text-zinc-300">1:30</code> or <code class="text-zinc-300">01:02:03</code>. Fill End <em>or</em> length. Keep clips under ~60s for Shorts/Reels.</p>
      <div class="grid grid-cols-2 gap-3">
        <div>
          <label class="text-xs font-bold uppercase tracking-wider text-zinc-400" for="style">Caption style</label>
          <select id="style" class="mt-2 w-full bg-ink border border-edge rounded-xl px-3 py-3 text-sm outline-none focus:border-brand">
            <option value="tiktok">TikTok — yellow highlight</option>
            <option value="karaoke">Karaoke — green highlight</option>
            <option value="boxed">Boxed — black bar</option>
            <option value="minimal">Minimal — clean white</option>
          </select>
        </div>
        <div>
          <label class="text-xs font-bold uppercase tracking-wider text-zinc-400" for="mode">9:16 fill mode</label>
          <select id="mode" class="mt-2 w-full bg-ink border border-edge rounded-xl px-3 py-3 text-sm outline-none focus:border-brand">
            <option value="blur">Blur pad — keep full frame</option>
            <option value="crop">Centre crop — fill frame</option>
          </select>
        </div>
      </div>
      <div class="grid grid-cols-2 gap-3 items-end">
        <div>
          <label class="text-xs font-bold uppercase tracking-wider text-zinc-400" for="model">Whisper model</label>
          <select id="model" class="mt-2 w-full bg-ink border border-edge rounded-xl px-3 py-3 text-sm outline-none focus:border-brand">
            <option value="tiny">tiny — fastest</option>
            <option value="base" selected>base — balanced</option>
            <option value="small">small — better</option>
            <option value="medium">medium — best (slow)</option>
          </select>
        </div>
        <label class="flex items-center gap-2 text-sm text-zinc-300 pb-3 cursor-pointer">
          <input id="captions" type="checkbox" checked class="accent-lime-400 w-4 h-4"> Burn captions in
        </label>
      </div>
      <button id="submit-btn" type="submit"
        class="w-full bg-brand text-ink font-extrabold rounded-xl py-3.5 text-sm hover:bg-lime-300 active:scale-[.99] transition disabled:opacity-50 disabled:cursor-wait">
        ✂ Create clip
      </button>
      <p id="form-error" class="hidden text-sm text-red-400"></p>
    </form>

    <div class="grid grid-cols-3 gap-3 mt-6 text-center">
      <div class="bg-panel border border-edge rounded-2xl p-4"><p class="text-xl">⬇️</p><p class="text-xs font-bold mt-1">1 · Download</p><p class="text-[11px] text-zinc-500 mt-1">yt-dlp grabs only your section</p></div>
      <div class="bg-panel border border-edge rounded-2xl p-4"><p class="text-xl">💬</p><p class="text-xs font-bold mt-1">2 · Caption</p><p class="text-[11px] text-zinc-500 mt-1">Whisper word-level timestamps</p></div>
      <div class="bg-panel border border-edge rounded-2xl p-4"><p class="text-xl">📱</p><p class="text-xs font-bold mt-1">3 · Export 9:16</p><p class="text-[11px] text-zinc-500 mt-1">FFmpeg renders 1080×1920</p></div>
    </div>
  </section>

  <!-- Jobs / preview -->
  <section class="lg:col-span-2">
    <div class="flex items-center justify-between">
      <h2 class="font-extrabold tracking-tight">Your clips</h2>
      <button id="refresh" class="text-xs text-zinc-400 hover:text-brand font-semibold">↻ Refresh</button>
    </div>
    <div id="jobs" class="mt-4 space-y-4"></div>
    <p id="empty" class="mt-4 text-sm text-zinc-500 border border-dashed border-edge rounded-2xl p-6 text-center">No clips yet. Create your first one → then download it and upload it to your ContentReward campaign.</p>
  </section>
</main>

<footer class="border-t border-edge/70 mt-6">
  <p class="max-w-6xl mx-auto px-4 py-5 text-[11px] text-zinc-500">ContentReward Clipper · Only clip content you own or have permission to use, and follow each platform's &amp; campaign's rules.</p>
</footer>

<script>
const $ = (s) => document.querySelector(s);
const jobsEl = $('#jobs'), emptyEl = $('#empty');
let polling = null;

function esc(s){ const d=document.createElement('div'); d.textContent=s??''; return d.innerHTML; }
function badge(status){
  if(status==='completed') return '<span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-lime-400/15 text-lime-300">READY</span>';
  if(status==='failed') return '<span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-red-400/15 text-red-300">FAILED</span>';
  return '<span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-violet-400/15 text-violet-300">WORKING</span>';
}
function renderJobs(jobs){
  emptyEl.style.display = jobs.length ? 'none' : 'block';
  jobsEl.innerHTML = jobs.map(j => `
    <article class="bg-panel border border-edge rounded-3xl p-4">
      <div class="flex items-start justify-between gap-3">
        <p class="text-xs text-zinc-400 break-all line-clamp-2">${esc(j.url)}</p>${badge(j.status)}
      </div>
      <p class="text-xs mt-2 text-zinc-300">${esc(j.message || j.stage || '')}</p>
      ${j.status !== 'completed' && j.status !== 'failed' ? `
        <div class="h-2 bg-ink rounded-full mt-3 overflow-hidden">
          <div class="h-full bg-gradient-to-r from-brand to-violet2 rounded-full transition-all" style="width:${j.progress||0}%"></div>
        </div><p class="text-[11px] text-zinc-500 mt-1">${j.progress||0}% · ${esc(j.stage||'')}</p>` : ''}
      ${j.status === 'failed' ? `<p class="text-xs text-red-400 mt-2 break-words">${esc(j.error||'Something went wrong.')}</p>` : ''}
      ${j.status === 'completed' && j.video_url ? `
        <video class="w-full aspect-[9/16] max-h-[480px] rounded-2xl mt-3 object-contain" src="${j.video_url}" controls playsinline preload="metadata"></video>
        <div class="flex gap-2 mt-3">
          <a href="${j.download_url}" class="flex-1 text-center bg-brand text-ink text-xs font-extrabold rounded-xl py-2.5 hover:bg-lime-300">⬇ Download MP4</a>
          <button onclick="removeJob('${j.id}')" class="px-3 border border-edge rounded-xl text-xs text-zinc-400 hover:text-red-300">Delete</button>
        </div>` : ''}
    </article>`).join('');
  const active = jobs.some(j => j.status==='queued' || j.status==='processing');
  if(active && !polling) polling = setInterval(loadJobs, 2000);
  if(!active && polling){ clearInterval(polling); polling = null; }
}
async function loadJobs(){
  try { const r = await fetch('/api/jobs'); if(r.ok) renderJobs(await r.json()); } catch(e){}
}
async function removeJob(id){ await fetch('/api/jobs/'+id, {method:'DELETE'}); loadJobs(); }
$('#refresh').addEventListener('click', loadJobs);

$('#clip-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const err = $('#form-error'); err.classList.add('hidden');
  const btn = $('#submit-btn'); btn.disabled = true; btn.textContent = 'Queuing…';
  const payload = {
    url: $('#url').value.trim(),
    start: $('#start').value.trim() || '0',
    end: $('#end').value.trim() || null,
    duration: $('#duration').value ? String($('#duration').value) : null,
    caption_style: $('#style').value,
    fill_mode: $('#mode').value,
    whisper_model: $('#model').value,
    burn_captions: $('#captions').checked,
  };
  try {
    const r = await fetch('/api/clips', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const data = await r.json().catch(() => ({}));
    if(!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check the URL and times, then try again.');
    await loadJobs();
    if(!polling) polling = setInterval(loadJobs, 2000);
  } catch(ex) {
    err.textContent = ex.message; err.classList.remove('hidden');
  } finally {
    btn.disabled = false; btn.textContent = '✂ Create clip';
  }
});
loadJobs();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
