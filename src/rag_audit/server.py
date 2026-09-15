from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from rag_audit.models import AuditDataset

app = FastAPI(title="RAG-Audit API")

# Permitir CORS para cuando desarrollas el frontend localmente con Vite (puerto 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRENT_DATASET: Optional[AuditDataset] = None


def set_dataset(dataset: AuditDataset) -> None:
    global CURRENT_DATASET
    CURRENT_DATASET = dataset


@app.get("/api/report", response_model=AuditDataset)
async def get_report():
    if CURRENT_DATASET is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    return CURRENT_DATASET


# Servir la interfaz SPA
dist_dir = Path(__file__).resolve().parent / "web" / "dist"

if dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=dist_dir / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        index_path = dist_dir / "index.html"
        return FileResponse(index_path)