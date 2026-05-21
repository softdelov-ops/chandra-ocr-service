# Chandra OCR 2 — Servicio Docker local para Google Colab

Servicio REST que corre **Chandra OCR 2** (`datalab-to/chandra-ocr-2`) en tu máquina local con GPU, expone una API HTTP, y sincroniza los resultados automáticamente a **Google Drive** con rclone.

Diseñado para usarse junto al notebook [RAG Digesto Comentado](../RAG_Digesto_Comentado_v18.ipynb): la celda `1.2-OCR` encola los PDFs aquí, libera la red de Colab, y la celda `1.2` lee directamente la carpeta de Drive ya procesada.

```
[Colab celda 1.2-OCR] ──HTTP──▶ [Docker :8000] ──GPU──▶ Chandra OCR 2
                                        │
                                 /data/output/*.md
                                        │
                                 rclone sync
                                        │
                              Google Drive / Digesto/.ocr_output
                                        │
                         [Colab celda 1.2 lee Drive] ──▶ documents
```

---

## Requisitos

| Item | Versión mínima |
|------|----------------|
| Docker + Docker Compose | v24 |
| NVIDIA GPU + drivers | CUDA 12.1+ |
| nvidia-container-toolkit | cualquiera |
| RAM sistema | 16 GB |
| VRAM | 8 GB (T4 suficiente con 4-bit) |

---

## Instalación rápida

```bash
# 1. Clonar el repo
git clone https://github.com/TU_USUARIO/chandra-ocr-service
cd chandra-ocr-service

# 2. Crear directorios de datos
mkdir -p data/{input,output,cache,models}

# 3. Copiar .env
cp .env.example .env
# Editar .env si querés cambiar BATCH_SIZE, OCR_DPI, etc.

# 4. Copiar tus PDFs a data/input/
#    (estructura de carpetas se preserva en el output)
cp /ruta/a/tus/pdfs/*.pdf data/input/

# 5. Iniciar el servicio
docker compose up -d

# Ver logs
docker compose logs -f chandra-ocr
```

---

## Sincronización automática a Google Drive

```bash
# 1. Configurar rclone (una sola vez)
bash scripts/setup_drive_sync.sh

# 2. Iniciar con sincronización
docker compose --profile with-drive up -d
```

El servicio `rclone-sync` corre un loop que cada 60 segundos sincroniza
`data/output/` → `gdrive:Digesto/.ocr_output` (configurable en `.env`).

---

## API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET`  | `/health` | Estado del servicio y modelo |
| `POST` | `/ocr/process` | Encolar PDFs para procesar |
| `GET`  | `/ocr/status/{job_id}` | Progreso de un job |
| `GET`  | `/ocr/jobs` | Listar todos los jobs |
| `GET`  | `/ocr/files` | Listar archivos `.md` generados |

### Ejemplo: encolar todos los PDFs

```bash
curl -X POST http://localhost:8000/ocr/process \
  -H "Content-Type: application/json" \
  -d '{"files": null, "force_reprocess": false}'
```

### Ejemplo: consultar progreso

```bash
curl http://localhost:8000/ocr/status/JOB_ID
```

---

## Uso desde Google Colab

La celda `1.2-OCR` del notebook hace esto automáticamente. Si querés usar el cliente manualmente:

```python
# Instalar cliente
!pip install -q httpx

import asyncio, nest_asyncio
nest_asyncio.apply()

# Descargar cliente
!wget -q https://raw.githubusercontent.com/TU_USUARIO/chandra-ocr-service/main/scripts/chandra_colab_client.py

from chandra_colab_client import ChandraOCRClient

async def main():
    async with ChandraOCRClient("http://TU_IP:8000") as c:
        await c.wait_ready()       # esperar que el modelo cargue
        job = await c.process_all()
        # No bloquea — podés ejecutar otras celdas mientras tanto
        # Cuando necesitás los resultados:
        await c.wait_done(job["job_id"])

asyncio.get_event_loop().run_until_complete(main())
```

### ¿Cómo acceder desde Colab a tu máquina local?

**Opción A — Red local (misma red WiFi/LAN):**
```
OCR_SERVICE_URL = "http://192.168.1.100:8000"  # IP de tu PC
```

**Opción B — ngrok (cualquier red):**
```bash
# En tu máquina local
ngrok http 8000
# Colab:
OCR_SERVICE_URL = "https://XXXX.ngrok.io"
```

**Opción C — Cloudflare Tunnel (gratuito, sin cuenta):**
```bash
cloudflared tunnel --url http://localhost:8000
```

---

## Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `CHANDRA_MODEL` | `datalab-to/chandra-ocr-2` | Modelo HuggingFace |
| `LOAD_4BIT` | `true` | Cuantización 4-bit NF4 |
| `OCR_DPI` | `96` | DPI para renderizar páginas |
| `BATCH_SIZE` | `2` | Páginas por llamada GPU |
| `MIN_CHARS` | `50` | Mínimo chars para considerar texto nativo |
| `DRIVE_REMOTE` | `gdrive` | Nombre del remote rclone |
| `DRIVE_PATH` | `Digesto/.ocr_output` | Ruta en Drive |
| `SYNC_INTERVAL` | `60` | Segundos entre syncs |

---

## Estructura del proyecto

```
chandra-ocr-service/
├── app/
│   └── main.py              ← FastAPI service
├── scripts/
│   ├── chandra_colab_client.py   ← Cliente async para Colab
│   └── setup_drive_sync.sh       ← Setup rclone
├── data/                    ← Creado en instalación (gitignore)
│   ├── input/               ← PDFs de entrada
│   ├── output/              ← Markdowns generados
│   ├── cache/               ← Caché de páginas OCR
│   └── models/              ← Cache modelos HuggingFace
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Caché

El servicio mantiene un caché en `data/cache/` usando el hash MD5 de
`(modelo + ruta_pdf + num_página + mtime)`. Al reprocesar un PDF sin cambios,
las páginas ya procesadas se recuperan del caché sin tocar la GPU.

Para forzar reprocesamiento completo:
```bash
curl -X POST http://localhost:8000/ocr/process \
  -d '{"force_reprocess": true}'
```

---

## Licencia

MIT
