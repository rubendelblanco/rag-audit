import json
import threading
import webbrowser
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_audit.evaluators import Evaluator, get_evaluator
from rag_audit.models import AuditDataset, RAGItem
from rag_audit.server import app as fastapi_app, set_dataset

console = Console()
app = typer.Typer(
    help="RAG-Audit: Inspect and debug hallucinations in your RAGs, locally."
)


def compute_metrics(items: list[RAGItem], evaluator: Evaluator) -> AuditDataset:
    """Evaluate each item with the given evaluator and aggregate dataset metrics."""
    processed = []
    faith_accum = 0.0
    rel_accum = 0.0

    for item in items:
        if not item.metrics:
            item.metrics = evaluator.evaluate(item)
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
        help="Path to the JSON file with your RAG predictions",
    ),
    port: int = typer.Option(
        8844, "--port", "-p", help="Local port for the dashboard"
    ),
    model: str = typer.Option(
        "ollama/llama3.2",
        "--model",
        "-m",
        help="Evaluator model (e.g. ollama/llama3.2 or gpt-4o-mini)",
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Don't open the browser automatically"
    ),
):
    """Load a dataset, evaluate its metrics, and launch the visual dashboard."""
    console.print(
        f"\n[bold green]RAG-Audit[/bold green] - Analyzing [cyan]{file_path.name}[/cyan]...\n"
    )

    # 1. Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    items = [RAGItem(**item) for item in raw_data]

    try:
        evaluator = get_evaluator(model)
    except ValueError as exc:
        console.print(f"[bold red]✗[/bold red] {exc}")
        raise typer.Exit(code=1)

    # 2. Evaluate with visual console feedback
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(
            description=f"Evaluating with [bold]{model}[/bold]...", total=None
        )
        dataset = compute_metrics(items, evaluator)

    set_dataset(dataset)

    url = f"http://127.0.0.1:{port}"
    console.print(f"[bold green]✓[/bold green] Audit complete ({dataset.total_queries} queries processed).")
    console.print(f"[bold blue]→ Dashboard ready at:[/bold blue] [link={url}]{url}[/link]\n")

    # 3. Launch the browser in the background
    if not no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    # 4. Uvicorn server
    uvicorn.run(fastapi_app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    app()