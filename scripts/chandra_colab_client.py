"""
chandra_colab_client.py
=======================
Cliente liviano para llamar al servicio Chandra OCR 2
desde Google Colab, con soporte async no-bloqueante.

Uso en Colab:
    # Instalar una sola vez
    !pip install -q httpx nest_asyncio

    from chandra_colab_client import ChandraOCRClient

    client = ChandraOCRClient("http://TU_IP_LOCAL:8000")
    await client.wait_ready()           # espera que el modelo cargue
    job = await client.process_all()    # encola todos los PDFs
    # ... seguir ejecutando otras celdas ...
    await client.wait_done(job["job_id"])  # esperar si es necesario
"""

import asyncio
import time
from typing import Optional, List, Dict
import httpx


class ChandraOCRClient:
    """
    Cliente async para el servicio Chandra OCR 2.

    Parámetros
    ----------
    base_url : str
        URL del servicio, ej: "http://192.168.1.100:8000"
    timeout  : int
        Timeout HTTP en segundos para llamadas individuales.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._client  = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    # ── Health / espera ────────────────────────────────────────────────────────
    async def health(self) -> dict:
        r = await self._client.get("/health")
        r.raise_for_status()
        return r.json()

    async def wait_ready(self, max_wait: int = 600, poll: float = 5.0) -> bool:
        """
        Espera hasta que el modelo esté cargado.
        No bloquea el event loop de Colab; podés seguir ejecutando celdas
        con asyncio.create_task() o simplemente await esta coroutine.

        Retorna True si el modelo está listo, False si se agotó el tiempo.
        """
        deadline = time.time() + max_wait
        print(f"⏳ Esperando servicio OCR en {self.base_url}...")
        while time.time() < deadline:
            try:
                h = await self.health()
                if h.get("status") == "ready":
                    print(f"✅ Modelo listo  |  {h.get('gpu', {})}")
                    return True
                else:
                    print(f"   Cargando modelo... ({h.get('status')}) — reintentando en {poll}s")
            except Exception as e:
                print(f"   Sin conexión ({e}) — reintentando en {poll}s")
            await asyncio.sleep(poll)
        print("❌ Timeout esperando el servicio OCR")
        return False

    # ── Procesamiento ──────────────────────────────────────────────────────────
    async def process_all(self, force: bool = False) -> dict:
        """Encola todos los PDFs en INPUT_DIR del servicio."""
        r = await self._client.post(
            "/ocr/process",
            json={"files": None, "force_reprocess": force},
        )
        r.raise_for_status()
        data = r.json()
        print(f"📋 Job encolado: {data['job_id']}  ({data['queued']} PDFs)")
        return data

    async def process_files(self, files: List[str], force: bool = False) -> dict:
        """Encola PDFs específicos (nombres relativos a INPUT_DIR)."""
        r = await self._client.post(
            "/ocr/process",
            json={"files": files, "force_reprocess": force},
        )
        r.raise_for_status()
        return r.json()

    # ── Status ─────────────────────────────────────────────────────────────────
    async def status(self, job_id: str) -> dict:
        r = await self._client.get(f"/ocr/status/{job_id}")
        r.raise_for_status()
        return r.json()

    async def wait_done(
        self,
        job_id: str,
        poll: float = 5.0,
        verbose: bool = True,
    ) -> dict:
        """
        Polling hasta que el job termine.
        No bloquea: usa `await` o `asyncio.create_task()`.
        """
        while True:
            s = await self.status(job_id)
            if verbose:
                pct   = s["progress_pct"]
                done  = s["processed"]
                total = s["total"]
                bar   = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
                print(f"\r  [{bar}] {pct:5.1f}%  ({done}/{total})", end="", flush=True)
            if s["status"] in ("done", "error"):
                print()  # newline
                if s["status"] == "error":
                    print(f"❌ Job {job_id} error: {s.get('error')}")
                else:
                    print(f"✅ Job {job_id} completado — {s['total']} archivos procesados")
                return s
            await asyncio.sleep(poll)

    # ── Archivos de salida ─────────────────────────────────────────────────────
    async def list_output_files(self) -> List[str]:
        r = await self._client.get("/ocr/files")
        r.raise_for_status()
        return r.json()["files"]

    # ── Context manager ────────────────────────────────────────────────────────
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self._client.aclose()
