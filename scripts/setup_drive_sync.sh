#!/usr/bin/env bash
# ── scripts/setup_drive_sync.sh ───────────────────────────────────────────────
# Configura rclone para sincronizar OUTPUT_DIR → Google Drive
# Ejecutar una sola vez antes del primer docker compose up

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RCLONE_CONF="$PROJECT_DIR/rclone.conf"

echo "════════════════════════════════════════════════════════"
echo "  Chandra OCR 2 — Configuración de Google Drive Sync"
echo "════════════════════════════════════════════════════════"

# 1. Verificar que rclone esté instalado (o usar el contenedor)
if ! command -v rclone &>/dev/null; then
    echo "ℹ️  rclone no está instalado localmente."
    echo "   Usando el contenedor Docker para configurar..."
    USE_DOCKER=true
else
    USE_DOCKER=false
fi

# 2. Crear directorios de datos
echo ""
echo "📁 Creando estructura de directorios..."
mkdir -p "$PROJECT_DIR/data/input"
mkdir -p "$PROJECT_DIR/data/output"
mkdir -p "$PROJECT_DIR/data/cache"
mkdir -p "$PROJECT_DIR/data/models"
echo "   ✅ data/{input,output,cache,models}"

# 3. Configurar rclone
echo ""
echo "🔑 Configurando rclone para Google Drive..."
echo "   (Se abrirá un navegador para autorizar el acceso)"
echo ""

if [ "$USE_DOCKER" = true ]; then
    docker run --rm -it \
        -v "$PROJECT_DIR:/project" \
        rclone/rclone:latest \
        config --config /project/rclone.conf
else
    rclone config --config "$RCLONE_CONF"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Configuración completada."
echo ""
echo "Próximos pasos:"
echo "  1. Editar .env si necesitás cambiar DRIVE_PATH"
echo "  2. Copiar tus PDFs a: data/input/"
echo "  3. Iniciar el servicio:"
echo "       docker compose up -d"
echo "     O con sincronización a Drive:"
echo "       docker compose --profile with-drive up -d"
echo ""
echo "  Para ver logs:"
echo "       docker compose logs -f chandra-ocr"
echo "════════════════════════════════════════════════════════"
