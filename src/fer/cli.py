"""Command-line interface: ``fer <command>``.

Subcommands mirror the pipeline stages — ``data``, ``train``, ``eval``, ``export``,
``bench``, ``webcam``, ``image`` — so the whole project is drivable without writing code.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Real-time facial emotion recognition: build FER+, train, export and run.",
)

DEFAULT_CONFIG = Path("configs/balanced.yaml")


def _setup_logging(verbose: bool = False) -> None:
    import sys

    from rich.logging import RichHandler

    # Windows consoles often default to cp1252, which cannot encode the symbols some
    # libraries print. Degrade those characters instead of crashing the command.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=verbose)],
    )
    # These are chatty at INFO and drown out the training log.
    for noisy in ("PIL", "matplotlib", "httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def data(
    root: Annotated[Path, typer.Option(help="Where to write the dataset cache.")] = Path("data"),
    force: Annotated[bool, typer.Option(help="Rebuild even if the cache exists.")] = False,
    verbose: bool = False,
) -> None:
    """Download and build the FER+ dataset (~35k images, a few hundred MB of cache)."""
    _setup_logging(verbose)
    from fer.data.ferplus import build_ferplus

    path = build_ferplus(root, force=force)
    typer.secho(f"Dataset ready at {path}", fg=typer.colors.GREEN)


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c", help="YAML config.")] = DEFAULT_CONFIG,
    out: Annotated[Path | None, typer.Option(help="Override the run output directory.")] = None,
    epochs: Annotated[int | None, typer.Option(help="Override the epoch count.")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Override the batch size.")] = None,
    device: Annotated[str, typer.Option(help="cuda, cpu, mps or auto.")] = "auto",
    no_compile: Annotated[bool, typer.Option(help="Disable torch.compile.")] = False,
    verbose: bool = False,
) -> None:
    """Fine-tune the backbone on FER+, saving the best checkpoint by macro-F1."""
    _setup_logging(verbose)
    from fer.config import Config
    from fer.train import train as run_train

    cfg = Config.from_yaml(config)
    if out is not None:
        cfg.out_dir = out
    if epochs is not None:
        cfg.train.epochs = epochs
    if batch_size is not None:
        cfg.data.batch_size = batch_size
    if no_compile:
        cfg.train.compile = False

    summary = run_train(cfg, device_str=device)
    typer.secho(
        f"Best macro-F1 {summary['best_macro_f1']:.4f} "
        f"(epoch {int(summary['best_epoch'])}, {summary['minutes']:.1f} min)",
        fg=typer.colors.GREEN,
    )


@app.command("eval")
def evaluate_cmd(
    checkpoint: Annotated[Path, typer.Argument(help="Path to a .pt checkpoint.")],
    split: Annotated[str, typer.Option(help="train, val or test.")] = "test",
    data_root: Annotated[Path | None, typer.Option(help="Override the dataset root.")] = None,
    out: Annotated[Path | None, typer.Option(help="Write metrics and a figure here.")] = None,
    device: str = "auto",
    tta: Annotated[bool, typer.Option(help="Average the horizontally flipped view.")] = False,
    raw: Annotated[
        bool, typer.Option(help="Ignore any fitted calibration and use plain argmax.")
    ] = False,
    verbose: bool = False,
) -> None:
    """Score a checkpoint on a held-out split and print a per-class report."""
    _setup_logging(verbose)
    from fer.evaluate import evaluate

    evaluate(
        checkpoint,
        split=split,
        data_root=data_root,
        device_str=device,
        out_dir=out,
        tta=tta,
        use_calibration=not raw,
    )


@app.command()
def calibrate(
    checkpoint: Annotated[Path, typer.Argument(help="Path to a .pt checkpoint.")],
    split: Annotated[str, typer.Option(help="Split to fit on; never 'test'.")] = "val",
    tta: Annotated[bool, typer.Option(help="Fit on flip-averaged logits.")] = False,
    force: Annotated[
        bool, typer.Option(help="Store the bias even if it fails cross-validation.")
    ] = False,
    device: str = "auto",
    verbose: bool = False,
) -> None:
    """Fit a macro-F1-optimal per-class bias on val and store it in the checkpoint.

    The fit is only kept if it still helps on held-out folds; a bias that only looks good
    on the split it was fitted to is noise and is discarded.
    """
    _setup_logging(verbose)
    from fer.calibrate import calibrate as run_calibrate

    result = run_calibrate(checkpoint, split=split, tta=tta, device_str=device, force=force)
    if result["generalises"] or force:
        typer.secho(
            f"Stored. In-sample {result['macro_f1_before']:.4f} -> "
            f"{result['macro_f1_after']:.4f}, cross-validated {result['cv_mean_gain']:+.4f}",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            f"Rejected: cross-validated gain {result['cv_mean_gain']:+.4f} is not positive, "
            "so the fit is noise. The model is left using plain argmax.",
            fg=typer.colors.YELLOW,
        )


@app.command()
def export(
    checkpoint: Annotated[Path, typer.Argument(help="Path to a .pt checkpoint.")],
    out: Annotated[Path | None, typer.Option(help="Destination .onnx path.")] = None,
    opset: int = 18,
    verbose: bool = False,
) -> None:
    """Export a checkpoint to ONNX and verify it matches the PyTorch outputs."""
    _setup_logging(verbose)
    from fer.export import export_onnx

    path = export_onnx(checkpoint, out, opset=opset)
    typer.secho(f"Exported to {path}", fg=typer.colors.GREEN)


@app.command()
def bench(
    model: Annotated[Path, typer.Argument(help="Path to an exported .onnx model.")],
    batch: Annotated[int, typer.Option(help="Faces per forward pass.")] = 1,
    runs: int = 60,
    verbose: bool = False,
) -> None:
    """Measure CPU inference latency for an exported model."""
    _setup_logging(verbose)
    from fer.export import benchmark

    stats = benchmark(model, batch=batch, runs=runs)
    typer.echo(
        f"batch {int(stats['batch'])}: {stats['ms_per_batch']:.2f} ms/batch, "
        f"{stats['ms_per_image']:.2f} ms/image, {stats['images_per_second']:.0f} img/s"
    )


@app.command()
def webcam(
    model: Annotated[Path, typer.Argument(help="A .onnx (fast) or .pt checkpoint.")],
    camera: Annotated[int, typer.Option(help="Camera index.")] = 0,
    device: Annotated[str, typer.Option(help="cpu or cuda.")] = "cpu",
    detect_every: Annotated[int, typer.Option(help="Detect every N frames.")] = 1,
    smoothing: Annotated[float, typer.Option(help="EMA factor; lower is smoother.")] = 0.35,
    min_score: Annotated[float, typer.Option(help="Face detector confidence floor.")] = 0.7,
    top_k: Annotated[int, typer.Option(help="How many classes to show bars for.")] = 8,
    no_bars: Annotated[bool, typer.Option(help="Hide the probability bars.")] = False,
    no_mirror: Annotated[bool, typer.Option(help="Do not mirror the preview.")] = False,
    record: Annotated[Path | None, typer.Option(help="Also write an .mp4 here.")] = None,
    max_frames: Annotated[
        int | None, typer.Option(help="Stop after N frames instead of waiting for 'q'.")
    ] = None,
    headless: Annotated[
        bool, typer.Option(help="Skip the preview window (pair with --max-frames/--record).")
    ] = False,
    source: Annotated[
        Path | None, typer.Option(help="Read a video file instead of the camera.")
    ] = None,
    verbose: bool = False,
) -> None:
    """Run live emotion recognition on a webcam. Press q to quit."""
    _setup_logging(verbose)
    from fer.webcam import run_webcam

    summary = run_webcam(
        model,
        camera=camera,
        device=device,
        detect_every=detect_every,
        smoothing=smoothing,
        min_score=min_score,
        top_k=top_k,
        show_bars=not no_bars,
        mirror=not no_mirror,
        record=record,
        max_frames=max_frames,
        headless=headless,
        source=source,
    )
    typer.secho(
        f"{int(summary['frames'])} frames, {int(summary['faces_seen'])} face detections, "
        f"{summary['mean_fps']:.1f} FPS ({summary['mean_ms_per_frame']:.1f} ms/frame)",
        fg=typer.colors.GREEN,
    )


@app.command()
def image(
    model: Annotated[Path, typer.Argument(help="A .onnx or .pt model.")],
    path: Annotated[Path, typer.Argument(help="Image to analyse.")],
    out: Annotated[Path | None, typer.Option(help="Write an annotated copy here.")] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> None:
    """Detect and classify every face in a still image."""
    _setup_logging(verbose)
    from fer.webcam import annotate_image

    results = annotate_image(model, path, out_path=out, device=device)
    if not results:
        typer.secho("No faces found.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    for i, (box, ranked) in enumerate(results, start=1):
        top = ", ".join(f"{name} {prob * 100:.1f}%" for name, prob in ranked)
        typer.echo(f"face {i} at {box}: {top}")


if __name__ == "__main__":
    app()
