"""
Scientific Document Intelligence Pipeline — CLI Entry Point.

Usage:
    python -m src.main run --file path/to/doc.pdf
    python -m src.main run --url https://arxiv.org/...
    python -m src.main run --dir data/input/
    python -m src.main search "transformer architecture"
    python -m src.main stats
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.logger import get_logger

logger = get_logger(__name__)


@click.group()
@click.option("--config", "-c", default="config/config.yaml", help="Path to config file.")
@click.option("--mode", "-m",
              type=click.Choice(["full", "extract_only", "search_only", "graph_only"]),
              default=None, help="Override pipeline mode from config.")
@click.option("--schema", "-s",
              type=click.Choice(["default", "research_paper", "patent"]),
              default=None, help="Override schema type from config.")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              help="Set log level.")
@click.pass_context
def cli(ctx, config, mode, schema, log_level):
    """Scientific Document Intelligence Pipeline."""
    import logging
    logging.getLogger().setLevel(log_level)

    overrides = {}
    if mode:
        overrides.setdefault("pipeline", {})["mode"] = mode
    if schema:
        overrides.setdefault("pipeline", {})["schema_type"] = schema

    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["overrides"] = overrides


@cli.command()
@click.option("--file", "-f", "file_path", default=None, help="Path to a document file.")
@click.option("--url", "-u", default=None, help="URL to a web document.")
@click.option("--dir", "-d", "dir_path", default=None, help="Directory of documents to process.")
@click.option("--output", "-o", default=None, help="Override output directory.")
@click.pass_context
def run(ctx, file_path, url, dir_path, output):
    """Process one or more documents through the intelligence pipeline."""
    config_path = ctx.obj["config"]
    overrides = ctx.obj["overrides"]

    if output:
        overrides.setdefault("output", {})["dir"] = output

    orchestrator = PipelineOrchestrator(config_path=config_path, config_overrides=overrides)

    if file_path:
        result = orchestrator.process_file(file_path)
        _print_result(result)
    elif url:
        result = orchestrator.process_url(url)
        _print_result(result)
    elif dir_path:
        result = orchestrator.process_directory(dir_path)
        click.echo(f"\n{'='*60}")
        click.echo(f"Pipeline Complete | run_id={result['run_id']}")
        click.echo(f"Total: {result['total']} | Succeeded: {result['succeeded']} | Failed: {result['failed']}")
        click.echo(f"{'='*60}")
    else:
        # Default: process data/input/ directory
        default_input = "data/input"
        if Path(default_input).exists() and any(Path(default_input).iterdir()):
            click.echo(f"No source specified — processing default input directory: {default_input}")
            result = orchestrator.process_directory(default_input)
            click.echo(f"\nComplete: {result['succeeded']}/{result['total']} documents processed")
        else:
            click.echo("No source specified. Use --file, --url, or --dir.")
            click.echo("Example: python -m src.main run --file data/input/paper.pdf")
            sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=10, help="Number of results to return.")
@click.option("--filter", "-f", "filter_str", default=None,
              help='Metadata filter as JSON string. E.g. \'{"doc_type":"pdf"}\'')
@click.pass_context
def search(ctx, query, top_k, filter_str):
    """Semantic search over indexed documents."""
    config_path = ctx.obj["config"]
    overrides = ctx.obj["overrides"]
    orchestrator = PipelineOrchestrator(config_path=config_path, config_overrides=overrides)

    filters = json.loads(filter_str) if filter_str else None
    results = orchestrator.search(query, top_k=top_k, filters=filters)

    click.echo(f"\nSearch results for: {query!r} (top {top_k})\n{'─'*60}")
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        source = r.get("metadata", {}).get("source", "unknown")
        text_preview = r["text"][:200].replace("\n", " ")
        click.echo(f"\n[{i}] Score: {score:.4f} | Source: {source}")
        click.echo(f"    {text_preview}...")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show pipeline statistics (indexed docs, graph size, output files)."""
    config_path = ctx.obj["config"]
    overrides = ctx.obj["overrides"]
    orchestrator = PipelineOrchestrator(config_path=config_path, config_overrides=overrides)
    s = orchestrator.get_stats()
    click.echo(json.dumps(s, indent=2, default=str))


@cli.command()
@click.option("--collection", is_flag=True, help="Clear ChromaDB collection.")
@click.option("--graph", is_flag=True, help="Clear Neo4j graph.")
@click.option("--outputs", is_flag=True, help="Delete all output JSON files.")
@click.option("--all", "clear_all", is_flag=True, help="Clear everything.")
@click.pass_context
def clear(ctx, collection, graph, outputs, clear_all):
    """Clear stored data from pipeline stores."""
    config_path = ctx.obj["config"]
    overrides = ctx.obj["overrides"]
    orchestrator = PipelineOrchestrator(config_path=config_path, config_overrides=overrides)

    if collection or clear_all:
        orchestrator.chroma_store.delete_collection()
        click.echo("ChromaDB collection cleared.")

    if graph or clear_all:
        orchestrator.neo4j_store.clear_graph()
        click.echo("Neo4j graph cleared.")

    if outputs or clear_all:
        import shutil
        output_dir = Path(orchestrator.config.get("output", {}).get("dir", "data/output"))
        if output_dir.exists():
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        click.echo("Output files cleared.")


def _print_result(result: dict):
    status = result.get("status", "unknown")
    source = result.get("source", "")
    elapsed = result.get("elapsed_seconds", 0)

    if status == "success":
        tools_called = result.get("tools_called", [])
        click.echo(f"\n✓ Processed: {source}")
        click.echo(f"  Status: {status} | Time: {elapsed}s | Tools: {', '.join(tools_called)}")
        if result.get("agent_summary"):
            click.echo(f"\n{result['agent_summary']}")
    else:
        click.echo(f"\n✗ Failed: {source}")
        click.echo(f"  Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    cli(obj={})
