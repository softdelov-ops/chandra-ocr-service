"""
Chandra OCR 2 — Servicio REST local para Google Colab
=====================================================
Corre en Docker local, acepta PDFs via HTTP y guarda resultados en
una carpeta montada compartida con Google Drive (rclone).

Endpoints:
  POST /ocr/process   — Encola PDFs para procesar
  GET  /ocr/status    — Estado del job
  GET  /ocr/jobs      — Lista todos los jobs
  GET  /health        — Health check
"""

import os
import gc
import hashlib
import json
import logging
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import fitz  # PyMuPDF
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pdf2image import convert_from_path
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("chandra-ocr")

# ── Configuración desde ENV ────────────────────────────────────────────────────
CHANDRA_MODEL    = os.getenv("CHANDRA_MODEL",    "datalab-to/chandra-ocr-2")
OCR_DPI          = int(os.getenv("OCR_DPI",      "96"))
MAX_IMG_SIDE     = int(os.getenv("MAX_IMG_SIDE",  "1600"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE",    "2"))
PREFETCH_WORKERS = int(os.getenv("PREFETCH_WORKERS", "2"))
MIN_CHARS        = int(os.getenv("MIN_CHARS",     "50"))
LOAD_4BIT        = os.getenv("LOAD_4BIT", "true").lower() == "true"

# Directorios montados desde el host (via Docker volumes)
INPUT_DIR  = Path(os.getenv("INPUT_DIR",  "/data/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
CACHE_DIR  = Path(os.getenv("CACHE_DIR",  "/data/cache"))

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Estado global ─────────────────────────────────────────────────────────────
app       = FastAPI(title="Chandra OCR 2 Service", version="1.0.0")
JOBS: Dict[str, dict] = {}          # job_id → {status, files, progress, error}
MODEL_LOCK = threading.Lock()
MODEL_READY = threading.Event()

chandra_model = None
proc          = None


# ── Modelos Pydantic ───────────────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    files: Optional[List[str]] = None   # None → todos los PDFs en INPUT_DIR
    force_reprocess: bool = False

class JobStatus(BaseModel):
    job_id: str
    status: str          # queued | running | done | error
    total: int
    processed: int
    progress_pct: float
    files_done: List[str]
    error: Optional[str] = None


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _cache_key(pdf_path: str, page_num: int) -> str:
    mtime = os.path.getmtime(pdf_path)
    raw   = f"{CHANDRA_MODEL}::4bit::{pdf_path}::{page_num}::{mtime}"
    return hashlib.md5(raw.encode()).hexdigest()

def _cache_get(key: str) -> Optional[str]:
    p = CACHE_DIR / f"{key}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None

def _cache_set(key: str, text: str):
    (CACHE_DIR / f"{key}.txt").write_text(text, encoding="utf-8")


# ── Model loader (lazy, thread-safe) ──────────────────────────────────────────
def _load_model():
    global chandra_model, proc
    log.info("⏳ Cargando Chandra OCR 2...")

    try:
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        from chandra.model.hf     import generate_hf
        from chandra.model.schema import BatchInputItem
        from chandra.output       import parse_markdown

        # Guardar referencias globales a las funciones del módulo
        app.state.generate_hf   = generate_hf
        app.state.BatchInputItem = BatchInputItem
        app.state.parse_markdown = parse_markdown

        if LOAD_4BIT and torch.cuda.is_available():
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            chandra_model = AutoModelForImageTextToText.from_pretrained(
                CHANDRA_MODEL,
                quantization_config=bnb,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            chandra_model = AutoModelForImageTextToText.from_pretrained(
                CHANDRA_MODEL,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

        chandra_model.eval()
        proc = AutoProcessor.from_pretrained(CHANDRA_MODEL)
        proc.tokenizer.padding_side = "left"
        chandra_model.processor = proc

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            log.info(f"✅ Modelo listo | VRAM libre: {free/1024**3:.1f}/{total/1024**3:.1f} GB")
        else:
            log.info("✅ Modelo listo (CPU)")

    except Exception as e:
        log.error(f"❌ Error cargando modelo: {e}")
        raise

    MODEL_READY.set()


# ── Image helpers ──────────────────────────────────────────────────────────────
def _resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > MAX_IMG_SIDE:
        r = MAX_IMG_SIDE / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    return img

def _render_page(args: Tuple) -> Tuple[int, Optional[Image.Image]]:
    pdf_path, page_num = args
    try:
        imgs = convert_from_path(
            pdf_path, dpi=OCR_DPI,
            first_page=page_num + 1, last_page=page_num + 1,
        )
        return page_num, (_resize(imgs[0]) if imgs else None)
    except Exception:
        return page_num, None

def _limpiar_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ── OCR batch ─────────────────────────────────────────────────────────────────
def _ocr_batch(items: List[Tuple]) -> Dict[int, str]:
    """items: [(pdf_path, page_num, img), ...]"""
    generate_hf  = app.state.generate_hf
    BatchInput   = app.state.BatchInputItem
    parse_md     = app.state.parse_markdown

    batch_input = [BatchInput(image=img, prompt_type="ocr_layout") for _, _, img in items]
    try:
        with torch.inference_mode():
            results = generate_hf(batch_input, chandra_model)
        texts = [parse_md(r.raw).strip() for r in results]
    finally:
        del batch_input
        _limpiar_vram()

    out = {}
    for (pdf_path, page_num, img), text in zip(items, texts):
        key = _cache_key(pdf_path, page_num)
        _cache_set(key, text)
        out[page_num] = text
        del img
    gc.collect()
    return out


# ── Procesamiento de un PDF → guarda .md en OUTPUT_DIR ────────────────────────
def _process_pdf(pdf_path: Path, force: bool = False) -> dict:
    """
    Procesa un PDF y guarda el resultado en OUTPUT_DIR/<nombre>.md
    Mantiene la misma sub-ruta relativa a INPUT_DIR.
    Retorna stats.
    """
    rel = pdf_path.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / rel.with_suffix(".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Si ya existe el .md y no forzamos, saltear
    if out_path.exists() and not force:
        log.info(f"  ⏭  Skip (ya procesado): {rel}")
        return {"file": str(rel), "status": "skipped", "pages": 0}

    pages_text: Dict[int, str] = {}
    pages_to_ocr: List[int]    = []
    total_native = 0
    total_cached = 0

    try:
        pdf = fitz.open(str(pdf_path))
    except Exception as e:
        return {"file": str(rel), "status": "error", "error": str(e)}

    for i in range(len(pdf)):
        text = pdf[i].get_text("text").strip()
        if len(text) >= MIN_CHARS:
            pages_text[i] = text
            total_native += 1
        else:
            key    = _cache_key(str(pdf_path), i)
            cached = _cache_get(key)
            if cached is not None:
                pages_text[i] = cached
                total_cached += 1
            else:
                pages_to_ocr.append(i)
    pdf.close()

    total_ocr = 0
    if pages_to_ocr:
        # Pre-render en paralelo (CPU)
        images: Dict[int, Image.Image] = {}
        with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as ex:
            futs = {ex.submit(_render_page, (str(pdf_path), p)): p for p in pages_to_ocr}
            for fut in as_completed(futs):
                pnum, img = fut.result()
                if img is not None:
                    images[pnum] = img

        # GPU en batch
        items_gpu = [(str(pdf_path), p, images[p]) for p in pages_to_ocr if p in images]
        with MODEL_LOCK:
            for i in range(0, len(items_gpu), BATCH_SIZE):
                batch = items_gpu[i: i + BATCH_SIZE]
                try:
                    res = _ocr_batch(batch)
                    for pnum, text in res.items():
                        pages_text[pnum] = text
                        total_ocr += 1
                except Exception as e:
                    log.error(f"    ❌ Batch error en {rel}: {e}")
                    _limpiar_vram()

        for img in images.values():
            del img
        gc.collect()

    # Escribir resultado markdown
    lines = [f"# {pdf_path.stem}\n"]
    for i in sorted(pages_text.keys()):
        text = pages_text.get(i, "")
        if text:
            lines.append(f"\n---\n<!-- page:{i+1} -->\n{text}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  ✅ {rel} → {out_path.name}  [native:{total_native} cache:{total_cached} ocr:{total_ocr}]")

    return {
        "file"   : str(rel),
        "status" : "done",
        "pages"  : len(pages_text),
        "native" : total_native,
        "cached" : total_cached,
        "ocr"    : total_ocr,
        "output" : str(out_path),
    }


# ── Worker de job ──────────────────────────────────────────────────────────────
def _run_job(job_id: str, pdf_paths: List[Path], force: bool):
    JOBS[job_id]["status"]   = "running"
    JOBS[job_id]["total"]    = len(pdf_paths)
    JOBS[job_id]["processed"] = 0

    # Esperar a que el modelo esté listo (máx 10 min)
    if not MODEL_READY.wait(timeout=600):
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"]  = "Timeout esperando el modelo"
        return

    for pdf_path in pdf_paths:
        result = _process_pdf(pdf_path, force=force)
        JOBS[job_id]["files_done"].append(result)
        JOBS[job_id]["processed"] += 1

    JOBS[job_id]["status"]       = "done"
    JOBS[job_id]["progress_pct"] = 100.0
    log.info(f"✅ Job {job_id} finalizado ({len(pdf_paths)} archivos)")


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    """Carga el modelo en background al iniciar el servicio."""
    thread = threading.Thread(target=_load_model, daemon=True, name="model-loader")
    thread.start()
    log.info("🚀 Servicio iniciado — modelo cargando en background")


@app.get("/health")
def health():
    status = "ready" if MODEL_READY.is_set() else "loading"
    gpu_info = {}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu_info = {
            "device"   : torch.cuda.get_device_name(0),
            "vram_free": f"{free/1024**3:.1f} GB",
            "vram_total": f"{total/1024**3:.1f} GB",
        }
    return {"status": status, "model": CHANDRA_MODEL, "gpu": gpu_info}


@app.get("/ocr/jobs")
def list_jobs():
    return {"jobs": list(JOBS.values())}


@app.get("/ocr/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, f"Job {job_id} no encontrado")
    j = JOBS[job_id]
    total = j["total"] or 1
    return JobStatus(
        job_id       = job_id,
        status       = j["status"],
        total        = j["total"],
        processed    = j["processed"],
        progress_pct = round(j["processed"] / total * 100, 1),
        files_done   = [f["file"] for f in j["files_done"]],
        error        = j.get("error"),
    )


@app.post("/ocr/process")
def process_pdfs(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Encola PDFs para procesar.
    - Si req.files es None, procesa todos los PDFs en INPUT_DIR.
    - Si req.files tiene nombres, procesa solo esos.
    """
    if req.files is None:
        pdf_paths = sorted(INPUT_DIR.rglob("*.pdf"))
    else:
        pdf_paths = []
        for name in req.files:
            p = INPUT_DIR / name
            if not p.exists():
                raise HTTPException(400, f"Archivo no encontrado: {name}")
            pdf_paths.append(p)

    if not pdf_paths:
        return {"message": "No hay PDFs para procesar", "job_id": None}

    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id"    : job_id,
        "status"    : "queued",
        "total"     : len(pdf_paths),
        "processed" : 0,
        "files_done": [],
        "error"     : None,
    }

    background_tasks.add_task(_run_job, job_id, pdf_paths, req.force_reprocess)
    log.info(f"📋 Job {job_id} encolado: {len(pdf_paths)} archivos")

    return {
        "job_id"  : job_id,
        "queued"  : len(pdf_paths),
        "status"  : "queued",
        "poll_url": f"/ocr/status/{job_id}",
    }


@app.get("/ocr/files")
def list_output_files():
    """Lista los archivos .md ya procesados en OUTPUT_DIR."""
    files = [str(p.relative_to(OUTPUT_DIR)) for p in OUTPUT_DIR.rglob("*.md")]
    return {"output_dir": str(OUTPUT_DIR), "files": sorted(files), "count": len(files)}
