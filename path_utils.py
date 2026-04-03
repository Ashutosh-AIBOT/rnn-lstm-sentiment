from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
CHARTS = ROOT / "charts"

# Folders should be pre-created in the Docker image to avoid runtime PermissionErrors
# No runtime mkdir here for production safety. Dockerfile handles it.
