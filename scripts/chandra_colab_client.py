"""
chandra_colab_client.py  —  v4
Cliente async para que Google Colab se conecte al servicio Chandra OCR 2
corriendo en Docker local (PC/servidor), expuesto via ngrok o Cloudflare Tunnel.

Interfaz pública esperada por el notebook RAG_Digesto:
    async with ChandraOCRClient(url) as client:
        h   = await client.health()           → {"status", "gpu": {...}}
        job = await client.process_all(force) → {"job_id", "queued"}
        s   = await client.status(job_id)     → {"status", "progress_pct", "processed", "total"}
        s   = await client.wait_done(job_id)  → {"status": "done", ...}
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

VERSION = "4"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de instalación de httpx (Colab no lo trae por default)
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_httpx():
    try:
        import httpx  # noqa: F401
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "httpx"])


# ─────────────────────────────────────────────────────────────────────────────
# Detección automática de URL pública
# ─────────────────────────────────────────────────────────────────────────────
async def _detectar_url(url_manual: Optional[str], timeout: float = 4.0) -> Optional[str]:
    """
    Prueba candidatos en orden y devuelve el primero que responda en /health.
    Orden: url_manual → env OCR_SERVICE_URL → Colab Secret → ngrok API → Cloudflare file.
    """
    _ensure_httpx()
    import httpx
    from pathlib import Path

    candidatos: list[str] = []
    if url_manual and url_manual.strip():
        candidatos.append(url_manual.strip().rstrip("/"))
    env = os.environ.get("OCR_SERVICE_URL", "").strip()
    if env:
        candidatos.append(env.rstrip("/"))
    try:
        from google.colab import userdata
        s = userdata.get("OCR_SERVICE_URL")
        if s and s.strip():
            candidatos.append(s.strip().rstrip("/"))
    except Exception:
        pass
    # ngrok dashboard API
    try:
        async with httpx.AsyncClient(timeout=2) as hx:
            r = await hx.get("http://localhost:4040/api/tunnels")
            for t in r.json().get("tunnels", []):
                pu = t.get("public_url", "")
                if pu.startswith("https://"):
                    candidatos.append(pu.rstrip("/"))
    except Exception:
        pass
    # Cloudflare tunnel file
    cf = Path("/tmp/cf_tunnel_url.txt")
    if cf.exists():
        u = cf.read_text().strip()
        if u:
            candidatos.append(u.rstrip("/"))

    async with httpx.AsyncClient(timeout=timeout) as hx:
        for url in candidatos:
            try:
                r = await hx.get(f"{url}/health")
                if r.status_code == 200:
                    return url
            except Exception:
                continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Diagnóstico amigable cuando no conecta
# ─────────────────────────────────────────────────────────────────────────────
def _imprimir_ayuda_conexion(url: str):
    print(f"\n❌ No se pudo conectar al servicio OCR en {url}")
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  CAUSAS FRECUENTES")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    is_local = any(x in url for x in ["192.168", "10.", "172.", "localhost", "127.0.0.1"])
    if is_local:
        print()
        print("  ⚠️  Usás una IP/hostname local.")
        print("     Colab corre en servidores de Google — no puede")
        print("     alcanzar tu red local directamente.")
        print()
        print("  Solución A — ngrok (recomendado, gratis):")
        print("    En tu PC:")
        print("    docker compose --profile tunnel up -d")
        print("    # Copiá la URL del dashboard: http://localhost:4040")
        print("    # Pegala como OCR_SERVICE_URL en la celda 1.2-OCR")
        print()
        print("  Solución B — Cloudflare Tunnel (sin cuenta):")
        print("    docker run --rm cloudflare/cloudflared:latest \\")
        print("      tunnel --url http://host.docker.internal:8000")
        print("    # Copiá https://xxxx.trycloudflare.com")
        print()
        print("  Solución C — Secret de Colab:")
        print("    Guardá la URL ngrok/CF como Secret con nombre")
        print("    'OCR_SERVICE_URL' y dejá la variable vacía.")
    else:
        print()
        print("  1. ¿El Docker está corriendo?")
        print("     docker compose up -d && docker compose ps")
        print()
        print("  2. ¿El modelo terminó de cargar? (~2 min primera vez)")
        print("     En tu PC: curl http://localhost:8000/health")
        print()
        print("  3. ¿El tunnel sigue activo?")
        print("     Las URLs ngrok gratuitas expiran. Levantá el tunnel")
        print("     nuevamente y actualizá OCR_SERVICE_URL.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ─────────────────────────────────────────────────────────────────────────────
# ChandraOCRClient — clase principal
# ─────────────────────────────────────────────────────────────────────────────
class ChandraOCRClient:
    """
    Async context manager para comunicarse con el servicio Chandra OCR 2.

    Uso:
        async with ChandraOCRClient("https://xxxx.ngrok-free.app") as client:
            h   = await client.health()
            job = await client.process_all()
            s   = await client.status(job["job_id"])
            s   = await client.wait_done(job["job_id"])
    """

    def __init__(
        self,
        url: Optional[str] = None,
        timeout_connect: float = 6.0,
        timeout_request: float = 600.0,
    ):
        _ensure_httpx()
        self._url_manual       = url.rstrip("/") if url and url.strip() else None
        self._timeout_connect  = timeout_connect
        self._timeout_request  = timeout_request
        self._url: Optional[str] = None
        self._hx = None

    async def __aenter__(self) -> "ChandraOCRClient":
        import httpx
        # Detectar URL operativa
        self._url = await _detectar_url(self._url_manual, self._timeout_connect)
        self._hx  = httpx.AsyncClient(
            base_url=self._url or "http://localhost:8000",
            timeout=self._timeout_request,
        )
        return self

    async def __aexit__(self, *_):
        if self._hx:
            await self._hx.aclose()

    def _check_url(self):
        if not self._url:
            raise ConnectionError(
                "Servicio OCR no accesible. "
                "Verificá que el Docker esté corriendo y la URL sea pública (ngrok/Cloudflare)."
            )

    # ── /health ───────────────────────────────────────────────────────────────
    async def health(self) -> Dict[str, Any]:
        """
        Verifica conectividad y estado del modelo.
        Retorna: {"status": "ready"|"loading"|"error", "gpu": {"free_gb": X, "total_gb": Y}}
        Lanza ConnectionError si no conecta.
        """
        if not self._url:
            _imprimir_ayuda_conexion(self._url_manual or "ninguna URL configurada")
            raise ConnectionError("No se pudo detectar URL del servicio OCR.")
        try:
            r = await self._hx.get("/health")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            _imprimir_ayuda_conexion(self._url)
            raise ConnectionError(f"Error al conectar con {self._url}/health: {e}") from e

    # ── /process_all ──────────────────────────────────────────────────────────
    async def process_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Encola todos los PDFs de INPUT_DIR para OCR.
        force=True reprocesa aunque el .md ya exista.
        Retorna: {"job_id": "...", "queued": N, "skipped": M}
        No espera a que el job termine — retorna inmediatamente.
        """
        self._check_url()
        r = await self._hx.post("/process_all", json={"force": force})
        r.raise_for_status()
        return r.json()

    # ── /jobs/{job_id} ────────────────────────────────────────────────────────
    async def status(self, job_id: str) -> Dict[str, Any]:
        """
        Estado del job.
        Retorna: {"job_id", "status": "pending"|"running"|"done"|"error",
                  "processed", "total", "progress_pct", "errors": [...]}
        """
        self._check_url()
        r = await self._hx.get(f"/jobs/{job_id}")
        r.raise_for_status()
        return r.json()

    # ── wait_done ─────────────────────────────────────────────────────────────
    async def wait_done(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: float = 7200.0,
    ) -> Dict[str, Any]:
        """
        Hace polling hasta que el job llegue a estado 'done' o 'error'.
        Muestra barra de progreso en el output.
        Retorna el dict final de status().
        """
        self._check_url()
        t0 = asyncio.get_event_loop().time()
        last_pct = -1.0

        while True:
            elapsed = asyncio.get_event_loop().time() - t0
            if elapsed > timeout:
                raise TimeoutError(f"wait_done: timeout después de {timeout}s")

            s   = await self.status(job_id)
            pct = s.get("progress_pct", 0.0)
            st  = s.get("status", "?")

            if pct != last_pct:
                filled   = int(pct // 5)
                bar      = "█" * filled + "░" * (20 - filled)
                proc     = s.get("processed", "?")
                total    = s.get("total", "?")
                print(f"\r  [{bar}] {pct:.1f}%  ({proc}/{total})  {st}   ", end="", flush=True)
                last_pct = pct

            if st in ("done", "error"):
                print()  # nueva línea al terminar
                if st == "error":
                    errs = s.get("errors", [])
                    print(f"  ⚠️  Job terminó con errores: {errs}")
                return s

            await asyncio.sleep(poll_interval)
