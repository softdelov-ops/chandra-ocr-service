"""
chandra-ocr-service · FastAPI  v4
Endpoints que consume chandra_colab_client.py desde Google Colab.

GET  /health                → estado del modelo y GPU
POST /process_all           → encola job de OCR para todos los PDFs en INPUT_DIR
GET  /jobs/{job_id}         → estado del job (progress_pct, processed, total)
GET  /status  (legacy)      → alias de /health para compatibilidad
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import os
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ocr-service")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── Config desde env ───────────────────────────────────────────────────────────
CHANDRA_MODEL   = os.getenv("CHANDRA_MODEL", "datalab-to/chandra-ocr-2")
USE_4BIT        = os.getenv("USE_4BIT", "1") == "1"
OCR_DPI         = int(os.getenv("OCR_DPI", "96"))
MAX_IMG_SIDE    = int(os.getenv("MAX_IMG_SIDE", "1600"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "2"))
MIN_CHARS       = int(os.getenv("MIN_CHARS", "50"))
# INPUT_DIR: carpeta con los PDFs originales (montada desde Drive via rclone/FUSE o bind mount)
INPUT_DIR       = Path(os.getenv("INPUT_DIR", "/input"))
# OUTPUT_DIR: donde se guardan los .md procesados (montada en Drive)
OUTPUT_DIR      = Path(os.getenv("OUTPUT_DIR", "/output"))
CACHE_DIR       = Path(os.getenv("CACHE_DIR", "/cache"))

for d in [INPUT_DIR, OUTPUT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Estado global ──────────────────────────────────────────────────────────────
_model_ready = False
_model       = None
_processor   = None
# Jobs en memoria: {job_id: {"status", "processed", "total", "errors": []}}
_jobs: Dict[str, Dict[str, Any]] = {}
# Semáforo para no ejecutar dos jobs simultáneos (1 GPU)
_job_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Carga del modelo (al startup)
# ─────────────────────────────────────────────────────────────────────────────
def _load_model():
    global _model, _processor, _model_ready
    log.info("Cargando Chandra OCR 2...")
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    bnb = None
    if USE_4BIT and torch.cuda.is_available():
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    _model = AutoModelForImageTextToText.from_pretrained(
        CHANDRA_MODEL,
        quantization_config=bnb,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True,
    )
    _model.eval()
    _processor = AutoProcessor.from_pretrained(CHANDRA_MODEL)
    _processor.tokenizer.padding_side = "left"
    _model.processor = _processor
    _model_ready = True
    log.info("✅ Modelo listo")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chandra OCR 2 Service",
    description="OCR local para PDFs escaneados — RAG Digesto",
    version="4.0.0",
)


@app.on_event("startup")
def startup():
    _load_model()


# ─────────────────────────────────────────────────────────────────────────────
# /health  (y alias /status para retrocompatibilidad)
# ─────────────────────────────────────────────────────────────────────────────
def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "free_gb" : round(free  / 1024**3, 2),
        "total_gb": round(total / 1024**3, 2),
        "device"  : torch.cuda.get_device_name(0),
    }


@app.get("/health")
@app.get("/status")          # alias legacy
def health():
    return {
        "status" : "ready" if _model_ready else "loading",
        "model"  : CHANDRA_MODEL,
        "gpu"    : _gpu_info(),
        "ready"  : _model_ready,          # campo extra para el cliente v3 legacy
    }


# ─────────────────────────────────────────────────────────────────────────────
# /process_all  — encola job
# ─────────────────────────────────────────────────────────────────────────────
class ProcessAllRequest(BaseModel):
    force: bool = False   # True = reprocesar aunque el .md ya exista


@app.post("/process_all")
async def process_all(req: ProcessAllRequest, background: BackgroundTasks):
    if not _model_ready:
        return JSONResponse(status_code=503, content={"error": "Modelo aún cargando."})

    pdfs = sorted(INPUT_DIR.rglob("*.pdf"))
    if not pdfs:
        return JSONResponse(status_code=404, content={
            "error": f"No se encontraron PDFs en {INPUT_DIR}",
            "tip"  : "Montá la carpeta de PDFs en INPUT_DIR (/input por defecto)",
        })

    # Filtrar los que ya tienen .md si force=False
    if req.force:
        pendientes = pdfs
    else:
        pendientes = []
        for p in pdfs:
            rel  = p.relative_to(INPUT_DIR)
            dest = OUTPUT_DIR / rel.with_suffix(".md")
            if not dest.exists():
                pendientes.append(p)

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status"      : "pending",
        "processed"   : 0,
        "total"       : len(pendientes),
        "skipped"     : len(pdfs) - len(pendientes),
        "errors"      : [],
        "progress_pct": 0.0,
    }

    log.info(f"Job {job_id}: {len(pendientes)} PDFs pendientes, {len(pdfs)-len(pendientes)} ya procesados")
    background.add_task(_run_job, job_id, pendientes)

    return {
        "job_id" : job_id,
        "queued" : len(pendientes),
        "skipped": len(pdfs) - len(pendientes),
        "total"  : len(pdfs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# /jobs/{job_id}  — estado del job
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        return JSONResponse(status_code=404, content={"error": f"Job {job_id} no encontrado."})
    j = _jobs[job_id]
    total = j["total"]
    pct   = (j["processed"] / total * 100) if total > 0 else 100.0
    return {**j, "progress_pct": round(pct, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Background task: corre el OCR
# ─────────────────────────────────────────────────────────────────────────────
async def _run_job(job_id: str, pdfs: List[Path]):
    async with _job_lock:
        _jobs[job_id]["status"] = "running"
        log.info(f"Job {job_id} iniciado — {len(pdfs)} PDFs")
        loop = asyncio.get_event_loop()
        for i, pdf in enumerate(pdfs):
            try:
                await loop.run_in_executor(None, _procesar_pdf, pdf)
                _jobs[job_id]["processed"] = i + 1
                pct = round((i + 1) / len(pdfs) * 100, 1)
                _jobs[job_id]["progress_pct"] = pct
                log.info(f"  [{i+1}/{len(pdfs)}] ✅ {pdf.name}  ({pct}%)")
            except Exception as e:
                _jobs[job_id]["errors"].append({"file": str(pdf), "error": str(e)})
                log.error(f"  [{i+1}/{len(pdfs)}] ❌ {pdf.name}: {e}")
        _jobs[job_id]["status"] = "done"
        log.info(f"Job {job_id} completado. Errores: {len(_jobs[job_id]['errors'])}")


# ─────────────────────────────────────────────────────────────────────────────
# Procesamiento de un PDF
# ─────────────────────────────────────────────────────────────────────────────
def _cache_key(pdf_path: Path, page_num: int) -> str:
    mtime = pdf_path.stat().st_mtime
    raw   = f"{CHANDRA_MODEL}::{pdf_path.name}::{page_num}::{mtime}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    p = CACHE_DIR / f"{key}.txt"
    return p.read_text("utf-8") if p.exists() else None


def _cache_set(key: str, text: str):
    (CACHE_DIR / f"{key}.txt").write_text(text, "utf-8")


def _resize(img):
    from PIL import Image
    w, h = img.size
    if max(w, h) > MAX_IMG_SIDE:
        r   = MAX_IMG_SIDE / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    return img


def _vram_clean():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def _procesar_pdf(pdf_path: Path):
    """
    Extrae texto de todas las páginas de un PDF.
    Páginas con texto nativo suficiente → directo.
    Páginas escaneadas → Chandra OCR 2.
    Resultado → OUTPUT_DIR/<rel_path>.md (Markdown con layout preservado).
    """
    import fitz
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown
    from pdf2image import convert_from_path

    rel     = pdf_path.relative_to(INPUT_DIR)
    out_md  = OUTPUT_DIR / rel.with_suffix(".md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    doc      = fitz.open(str(pdf_path))
    n_pages  = len(doc)
    pages_md: dict[int, str] = {}
    to_ocr:   list[int]      = []

    # Clasificar páginas
    for pn in range(n_pages):
        text = doc[pn].get_text("text").strip()
        if len(text) >= MIN_CHARS:
            pages_md[pn] = text
        else:
            key    = _cache_key(pdf_path, pn)
            cached = _cache_get(key)
            if cached is not None:
                pages_md[pn] = cached
            else:
                to_ocr.append(pn)
    doc.close()

    # OCR en batch
    if to_ocr:
        for i in range(0, len(to_ocr), BATCH_SIZE):
            batch_pns = to_ocr[i : i + BATCH_SIZE]
            imgs = []
            for pn in batch_pns:
                raw = convert_from_path(
                    str(pdf_path), dpi=OCR_DPI,
                    first_page=pn + 1, last_page=pn + 1,
                )
                imgs.append(_resize(raw[0]) if raw else None)

            valid = [(pn, img) for pn, img in zip(batch_pns, imgs) if img]
            if not valid:
                continue

            batch_in = [BatchInputItem(image=img, prompt_type="ocr_layout") for _, img in valid]
            try:
                with torch.inference_mode():
                    results = generate_hf(batch_in, _model)
                texts = [parse_markdown(r.raw).strip() for r in results]
            finally:
                del batch_in
                _vram_clean()

            for (pn, _), text in zip(valid, texts):
                key = _cache_key(pdf_path, pn)
                _cache_set(key, text)
                pages_md[pn] = text

            del valid, imgs
            gc.collect()

    # Escribir .md
    lines = [f"# {pdf_path.name}\n"]
    for pn in sorted(pages_md):
        lines.append(f"\n## Página {pn + 1}\n\n{pages_md[pn]}\n")
    out_md.write_text("".join(lines), "utf-8")
    log.debug(f"  → {out_md}  ({len(to_ocr)} OCR, {n_pages - len(to_ocr)} nativo/caché)")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
