# 🔍 Chandra OCR 2 · Servicio Docker Local

Servicio REST local para OCR de PDFs escaneados usando **Chandra OCR 2** (`datalab-to/chandra-ocr-2`).  
Diseñado para integrarse con el notebook **RAG Digesto Universitario** en Google Colab.

---

## ✨ Arquitectura

```
[Colab celda 1.2-OCR]
   │  HTTP (ngrok/Cloudflare Tunnel)
   ▼
[Docker local: puerto 8000]  ←  ChandraOCRClient (chandra_colab_client.py)
   │  GPU local
   ▼
Chandra OCR 2 (4B params, 4-bit NF4)
   │
   ▼
/output/*.md  ──  rclone/watch_and_sync.py  ──▶  Google Drive
                                                       │
                              [Colab celda 1.2 lee Drive] ──▶ documents
```

**Beneficios:**
- No usa GPU de Colab → GPU libre para embeddings y LLM
- No bloquea el notebook → celdas 1.3, 1.4, 1.5 ejecutan en paralelo
- Caché persistente → no reprocesa páginas ya hechas
- Resultados en Drive → disponibles en sesiones futuras

---

## 🚀 Inicio rápido

### 1. Clonar el repo

```bash
git clone https://github.com/softdelov-ops/chandra-ocr-service.git
cd chandra-ocr-service
```

### 2. Copiar los PDFs a /input

```bash
# Opción A: bind mount de tu carpeta de Digesto
mkdir -p input
# Copiá o symlinkeá tu carpeta de PDFs dentro de ./input/

# Opción B: editar docker-compose.yml y cambiar ./input por la ruta real
#   - ./input:/input
#   + /ruta/a/tus/pdfs:/input
```

### 3. Levantar el servicio

```bash
# Solo GPU local
docker compose up -d --build

# Con tunnel ngrok (para que Colab alcance tu máquina)
NGROK_AUTHTOKEN=tu_token docker compose --profile tunnel up -d --build
```

> **Primera vez:** descarga del modelo (~8 GB) + carga en GPU: ~2 min.  
> Verificar: `curl http://localhost:8000/health`

### 4. Sync automático a Drive (opcional)

```bash
# Instalar rclone y configurar remote "gdrive"
curl https://rclone.org/install.sh | sudo bash
rclone config   # crear remote → Google Drive → llamalo "gdrive"

# Iniciar watcher en background
python scripts/watch_and_sync.py \
  --output-dir ./output \
  --remote "gdrive:Digesto/.ocr_output" &
```

### 5. En el notebook (Colab)

```python
# Celda 1.2-OCR
OCR_SERVICE_URL = ""   # vacío = auto-detecta ngrok
                       # o pegá: "https://xxxx.ngrok-free.app"
```

---

## 🔌 API

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servicio y GPU |
| `POST` | `/process_all` | Encolar OCR de todos los PDFs en /input |
| `GET` | `/jobs/{job_id}` | Estado del job (progress_pct, processed, total) |
| `GET` | `/cache/stats` | Páginas en caché |

### Ejemplo manual

```bash
# Verificar
curl http://localhost:8000/health

# Encolar job
curl -X POST http://localhost:8000/process_all \
  -H "Content-Type: application/json" \
  -d '{"force": false}'
# → {"job_id": "a1b2c3d4", "queued": 12, "skipped": 3, "total": 15}

# Consultar progreso
curl http://localhost:8000/jobs/a1b2c3d4
# → {"status": "running", "processed": 5, "total": 12, "progress_pct": 41.7}
```

---

## ⚙️ Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `CHANDRA_MODEL` | `datalab-to/chandra-ocr-2` | HuggingFace model ID |
| `USE_4BIT` | `1` | Cuantización 4-bit NF4 |
| `OCR_DPI` | `96` | DPI para renderizar páginas |
| `MAX_IMG_SIDE` | `1600` | Máximo lado de imagen |
| `BATCH_SIZE` | `2` | Páginas por lote GPU |
| `MIN_CHARS` | `50` | Mín. caracteres para considerar página como nativa |
| `INPUT_DIR` | `/input` | Carpeta con PDFs fuente |
| `OUTPUT_DIR` | `/output` | Carpeta de resultados `.md` |
| `CACHE_DIR` | `/cache` | Caché de páginas procesadas |

---

## 🗂️ Estructura

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

## 🐛 Troubleshooting

**`❌ No se pudo conectar en http://192.168.x.x:8000`**  
→ Colab no puede alcanzar IPs locales. Usá ngrok o Cloudflare Tunnel.

```bash
# ngrok (automático — el cliente lo detecta solo):
NGROK_AUTHTOKEN=xxx docker compose --profile tunnel up -d
# Ver URL: http://localhost:4040

# Cloudflare (sin cuenta):
docker run --rm cloudflare/cloudflared:latest \
  tunnel --url http://host.docker.internal:8000
```

**`cuda out of memory`**  
→ Reducí `BATCH_SIZE=1` en `docker-compose.yml`.

**El modelo tarda en cargar**  
→ Normal la primera vez (~2 min). El healthcheck espera 120s.  
→ Los pesos se cachean en `~/.cache/huggingface` para la próxima vez.

**No aparecen PDFs en `/input`**  
→ Verificar que el volumen esté montado correctamente:  
```bash
docker exec chandra-ocr-service ls /input
```
