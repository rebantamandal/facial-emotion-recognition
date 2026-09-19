"""Export a checkpoint to ONNX for fast CPU inference.

ONNX Runtime is substantially quicker than eager PyTorch on CPU for a model this size,
which is what makes the webcam app comfortably real time without a GPU. The exported
graph takes a dynamic batch dimension so every face in a frame is scored in one call.

The preprocessing spec and class names are written to a sidecar ``.json`` so the ``.onnx``
is self-contained in practice: nothing downstream has to guess the normalisation.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from pathlib import Path

import numpy as np
import torch

from fer.models import load_checkpoint

logger = logging.getLogger(__name__)


def export_onnx(
    checkpoint: Path | str,
    out_path: Path | str | None = None,
    opset: int = 18,
    verify: bool = True,
) -> Path:
    """Convert ``checkpoint`` to ONNX and verify the outputs still match."""
    checkpoint = Path(checkpoint)
    out_path = Path(out_path) if out_path else checkpoint.with_suffix(".onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, spec, class_names, payload = load_checkpoint(checkpoint, device="cpu")
    model.eval()

    example = torch.randn(1, 3, spec.image_size, spec.image_size)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    try:
        # torch>=2.5 routes through the dynamo exporter, which produces a cleaner graph.
        # Its progress output contains emoji, which raises UnicodeEncodeError on a
        # cp1252 Windows console, so it is captured rather than printed.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            torch.onnx.export(
                model,
                (example,),
                str(out_path),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                dynamo=True,
                # Keep the weights inside the .onnx. The dynamo exporter otherwise
                # writes them to a sibling .onnx.data, and copying the .onnx alone
                # then yields a model that fails to load. These models are tens of
                # MB, far under the 2 GB protobuf ceiling.
                external_data=False,
            )
        logger.debug("dynamo exporter output:\n%s", captured.getvalue())
    except (TypeError, ImportError, ModuleNotFoundError, UnicodeEncodeError) as exc:
        # Older torch has no `dynamo` kwarg; newer torch has one but routes it through
        # onnxscript, which may not be installed. Either way the legacy exporter works.
        logger.info("Dynamo ONNX export unavailable (%s); using the legacy exporter.", exc)
        torch.onnx.export(
            model,
            example,
            str(out_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "preprocess": spec.to_dict(),
                "class_names": class_names,
                "opset": opset,
                "source_checkpoint": str(checkpoint),
                # The graph stays uncalibrated; the bias rides alongside so the same
                # .onnx can be re-calibrated without re-exporting.
                "logit_bias": payload.get("logit_bias"),
                "calibration": payload.get("calibration"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _ensure_self_contained(out_path)

    size_mb = out_path.stat().st_size / 1e6
    logger.info("Exported %s (%.1f MB) with metadata at %s", out_path, size_mb, meta_path)

    if verify:
        max_diff = verify_onnx(model, out_path, spec.image_size)
        logger.info("Max |torch - onnx| logit difference: %.2e", max_diff)
        if max_diff > 1e-3:
            raise RuntimeError(
                f"ONNX output diverges from PyTorch by {max_diff:.2e}, which is too large "
                "to be numerical noise. The exported model would not match training."
            )
    return out_path


def _ensure_self_contained(out_path: Path) -> None:
    """Fold any external weight file back into the ``.onnx``.

    Some exporter paths write tensors to a sibling ``.onnx.data`` regardless of what was
    requested. That makes the ``.onnx`` useless on its own, which is a nasty surprise
    when someone copies just that file, so the weights are consolidated here and the
    stray sidecar removed.
    """
    import onnx

    graph = onnx.load(str(out_path), load_external_data=False).graph
    if not any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.initializer):
        return

    logger.info("Consolidating external weights into %s", out_path.name)
    model = onnx.load(str(out_path), load_external_data=True)
    onnx.save_model(model, str(out_path), save_as_external_data=False)
    sidecar = out_path.with_suffix(out_path.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()


def verify_onnx(model: torch.nn.Module, onnx_path: Path, image_size: int) -> float:
    """Compare torch and ONNX logits on random input, including a batched case."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    worst = 0.0
    for batch in (1, 4):  # exercise the dynamic axis, not just the traced shape
        sample = torch.randn(batch, 3, image_size, image_size)
        with torch.inference_mode():
            expected = model(sample).numpy()
        actual = session.run(None, {input_name: sample.numpy()})[0]
        worst = max(worst, float(np.abs(expected - actual).max()))
    return worst


def benchmark(onnx_path: Path | str, batch: int = 1, runs: int = 60) -> dict[str, float]:
    """Time the exported model on CPU, reporting per-batch latency and throughput."""
    import time

    import onnxruntime as ort

    meta = json.loads(Path(onnx_path).with_suffix(".json").read_text(encoding="utf-8"))
    size = int(meta["preprocess"]["image_size"])

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    sample = np.random.randn(batch, 3, size, size).astype(np.float32)

    for _ in range(10):  # warm up so the first-call graph setup is not timed
        session.run(None, {name: sample})

    started = time.perf_counter()
    for _ in range(runs):
        session.run(None, {name: sample})
    elapsed = time.perf_counter() - started

    per_batch_ms = elapsed / runs * 1000
    return {
        "batch": float(batch),
        "ms_per_batch": per_batch_ms,
        "ms_per_image": per_batch_ms / batch,
        "images_per_second": batch * runs / elapsed,
    }
