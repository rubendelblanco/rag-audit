import json
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_audit.models import AuditDataset, RAGItem, EvaluationMetrics
from rag_audit.server import app as fastapi_app, set_dataset

console = Console()
app = typer.Typer(
    help="RAG-Audit: Inspecciona y depura alucinaciones de tus RAGs localmente."
)


def compute_metrics_placeholder(items: list[RAGItem]) -> AuditDataset:
    """Calcula o rellena métricas mockeando la lógica de evaluación."""
    processed = []
    faith_accum = 0.0
    rel_accum = 0.0

    for item in items:
        if not item.metrics:
            # Aquí irá la llamada al LLM evaluador (Ollama/OpenAI)
            item.metrics = EvaluationMetrics(
                faithfulness=0.45,
                answer_relevance=0.85,
                reasoning="El contexto menciona el puerto 80, pero el modelo respondió 8080.",
            )
        faith_accum += item.metrics.faithfulness
        rel_accum += item.metrics.answer_relevance
        processed.append(item)

    total = len(processed)
    return AuditDataset(
        total_queries=total,
        avg_faithfulness=round(faith_accum / total, 2) if total else 0.0,
        avg_relevance=round(rel_accum / total, 2) if total else 0.0,
        items=processed,
    )


@app.command()
def view(
    file_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Ruta al JSON con tus predicciones de RAG",
    ),
    port: int = typer.Option(
        8844, "--port", "-p", help="Puerto local para el dashboard"
    ),
    model: str = typer.Option(
        "ollama/llama3.2",
        "--model",
        "-m",
        help="Modelo evaluador (ej: ollama/llama3.2 o gpt-4o-mini)",
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="No abrir automáticamente el navegador"
    ),
):
    """Carga un dataset, evalúa las métricas y levanta el dashboard visual."""
    console.print(
        f"\n[bold green]RAG-Audit[/bold green] - Analizando [cyan]{file_path.name}[/cyan]...\n"
    )

    # 1. Leer archivo
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    items = [RAGItem(**item) for item in raw_data]

    # 2. Evaluación con feedback visual en consola
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(
            description=f"Evaluando con [bold]{model}[/bold]...", total=None
        )
        time.sleep(1)  # Simulación
        dataset = compute_metrics_placeholder(items)

    set_dataset(dataset)

    url = f"http://127.0.0.1:{port}"
    console.print(f"[bold green]✓[/bold green] Auditoría completada ({dataset.total_queries} queries procesadas).")
    console.print(f"[bold blue]→ Dashboard listo en:[/bold blue] [link={url}]{url}[/link]\n")

    # 3. Lanzar navegador en segundo plano
    if not no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    # 4. Servidor Uvicorn
    uvicorn.run(fastapi_app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    app()