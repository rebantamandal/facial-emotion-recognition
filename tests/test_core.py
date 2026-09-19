"""Unit tests that run without the dataset, a GPU or a camera.

The emphasis is on the pieces where a silent bug would be invisible in training curves:
the FER2013 to FER+ class mapping, the soft-target loss, preprocessing normalisation, and
landmark alignment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fer.align import ARCFACE_TEMPLATE_112, align_face, canonical_template, crop_face
from fer.config import Config, DataConfig
from fer.data.datamodule import (
    FERPlusDataset,
    ToThreeChannels,
    _balanced_sampler,
    build_eval_transform,
    build_train_transform,
)
from fer.data.ferplus import EXPECTED_COUNTS, FER2013_TO_FERPLUS, SPLIT_ALIASES
from fer.engine import build_scheduler, mixup_cutmix, soft_cross_entropy
from fer.infer import CentroidTracker, ProbabilitySmoother, preprocess_batch, softmax
from fer.labels import CLASS_NAMES, NUM_CLASSES, VOTE_COLUMNS, class_index
from fer.metrics import compute_metrics
from fer.models import PreprocessSpec

# --------------------------------------------------------------------------- labels


def test_class_list_is_consistent():
    assert NUM_CLASSES == 8
    assert len(set(CLASS_NAMES)) == NUM_CLASSES
    # The vote columns must start with the trainable classes so a raw row can be sliced.
    assert VOTE_COLUMNS[:NUM_CLASSES] == CLASS_NAMES
    assert VOTE_COLUMNS[NUM_CLASSES:] == ("unknown", "NF")


def test_class_index_round_trips():
    for i, name in enumerate(CLASS_NAMES):
        assert class_index(name) == i
    with pytest.raises(KeyError):
        class_index("elation")


def test_fer2013_to_ferplus_mapping_is_correct():
    """FER2013's label order must map onto the FER+ names it actually denotes."""
    fer2013_order = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    expected = [CLASS_NAMES.index(name) for name in fer2013_order]
    assert FER2013_TO_FERPLUS.tolist() == expected


def test_split_bookkeeping_matches_published_counts():
    assert set(SPLIT_ALIASES) == set(EXPECTED_COUNTS)
    assert sum(EXPECTED_COUNTS.values()) == 35887
    assert sorted(SPLIT_ALIASES.values()) == ["test", "train", "val"]


# --------------------------------------------------------------------------- config


def test_config_defaults_and_yaml_round_trip(tmp_path):
    cfg = Config()
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    loaded = Config.from_yaml(path)
    assert loaded.model.backbone == cfg.model.backbone
    assert loaded.data.image_size == cfg.data.image_size
    assert loaded.train.epochs == cfg.train.epochs


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown key"):
        Config.from_dict({"train": {"learning_rate": 1e-3}})


def test_shipped_configs_load():
    for name in ("balanced", "fast"):
        cfg = Config.from_yaml(f"configs/{name}.yaml")
        assert cfg.data.source == "ferplus"
        assert cfg.data.image_size > 0
        assert 0.0 <= cfg.train.mixup_prob <= 1.0


# --------------------------------------------------------------------------- loss


def test_soft_cross_entropy_matches_hard_cross_entropy():
    """With one-hot targets and no smoothing it must equal the standard loss."""
    torch.manual_seed(0)
    logits = torch.randn(16, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES, (16,))
    one_hot = F.one_hot(labels, NUM_CLASSES).float()
    assert torch.allclose(
        soft_cross_entropy(logits, one_hot), F.cross_entropy(logits, labels), atol=1e-6
    )


def test_label_smoothing_matches_torch_reference():
    torch.manual_seed(1)
    logits = torch.randn(8, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES, (8,))
    one_hot = F.one_hot(labels, NUM_CLASSES).float()
    ours = soft_cross_entropy(logits, one_hot, label_smoothing=0.1)
    reference = F.cross_entropy(logits, labels, label_smoothing=0.1)
    assert torch.allclose(ours, reference, atol=1e-6)


def test_soft_cross_entropy_is_minimised_by_the_true_distribution():
    target = torch.tensor([[0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    perfect = torch.log(target + 1e-9)
    wrong = torch.log(torch.tensor([[0.1, 0.1, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0]]) + 1e-9)
    assert soft_cross_entropy(perfect, target) < soft_cross_entropy(wrong, target)


# --------------------------------------------------------------------------- mixup


def test_mixup_preserves_shapes_and_target_mass():
    torch.manual_seed(0)
    images = torch.rand(8, 3, 32, 32)
    targets = F.one_hot(torch.randint(0, NUM_CLASSES, (8,)), NUM_CLASSES).float()
    mixed_images, mixed_targets = mixup_cutmix(images, targets, 0.2, 1.0, prob=1.0)
    assert mixed_images.shape == images.shape
    assert mixed_targets.shape == targets.shape
    # Mixing two probability vectors must still yield a probability vector.
    assert torch.allclose(mixed_targets.sum(dim=1), torch.ones(8), atol=1e-5)


def test_mixup_is_a_noop_when_probability_is_zero():
    images = torch.rand(4, 3, 16, 16)
    targets = torch.rand(4, NUM_CLASSES)
    out_images, out_targets = mixup_cutmix(images, targets, 0.2, 1.0, prob=0.0)
    assert torch.equal(out_images, images)
    assert torch.equal(out_targets, targets)


# --------------------------------------------------------------------------- schedule


def test_scheduler_warms_up_then_decays():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = build_scheduler(
        optimizer, epochs=10, steps_per_epoch=10, warmup_epochs=2, min_lr=1e-6, base_lrs=[0.1]
    )
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert lrs[0] < lrs[10] < lrs[20]          # warming up
    assert lrs[20] == pytest.approx(0.1, rel=1e-3)  # peak at end of warmup
    assert lrs[-1] < lrs[20]                    # decayed afterwards
    assert min(lrs) >= 0.0


# --------------------------------------------------------------------------- dataset


def _toy_votes() -> tuple[np.ndarray, np.ndarray]:
    images = np.random.randint(0, 255, (6, 48, 48), dtype=np.uint8)
    votes = np.array(
        [
            [10, 0, 0, 0, 0, 0, 0, 0],
            [0, 8, 2, 0, 0, 0, 0, 0],
            [0, 0, 10, 0, 0, 0, 0, 0],
            [5, 5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 10],
            [1, 1, 1, 1, 1, 1, 1, 3],
        ],
        dtype=np.uint8,
    )
    return images, votes


def test_dataset_soft_targets_are_normalised_vote_shares():
    images, votes = _toy_votes()
    transform = build_eval_transform(32, (0.5,) * 3, (0.5,) * 3, grayscale=True)
    dataset = FERPlusDataset(images, votes, transform, label_mode="soft")

    image, target = dataset[1]
    assert image.shape == (3, 32, 32)
    assert target.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert target[1].item() == pytest.approx(0.8, abs=1e-5)
    assert target[2].item() == pytest.approx(0.2, abs=1e-5)


def test_dataset_majority_mode_is_one_hot():
    images, votes = _toy_votes()
    transform = build_eval_transform(32, (0.5,) * 3, (0.5,) * 3, grayscale=True)
    dataset = FERPlusDataset(images, votes, transform, label_mode="majority")
    _, target = dataset[1]
    assert target.sum().item() == pytest.approx(1.0)
    assert int(target.argmax()) == 1
    assert set(target.tolist()) == {0.0, 1.0}


def test_dataset_rejects_mismatched_inputs():
    images, votes = _toy_votes()
    transform = build_eval_transform(32, (0.5,) * 3, (0.5,) * 3)
    with pytest.raises(ValueError):
        FERPlusDataset(images[:3], votes, transform)
    with pytest.raises(ValueError, match="label_mode"):
        FERPlusDataset(images, votes, transform, label_mode="hard")


def test_balanced_sampler_upweights_rare_classes():
    images, votes = _toy_votes()
    transform = build_eval_transform(32, (0.5,) * 3, (0.5,) * 3)
    dataset = FERPlusDataset(images, votes, transform)
    weights = _balanced_sampler(dataset).weights.numpy()

    counts = dataset.class_counts()
    majority = dataset.majority
    # Class 0 appears twice (rows 0 and 3), class 2 once -> class 2 must weigh more.
    assert counts[0] == 2 and counts[2] == 1
    assert weights[majority == 2].max() > weights[majority == 0].max()


# --------------------------------------------------------------------------- metrics


def test_metrics_on_a_perfect_predictor():
    targets = np.eye(NUM_CLASSES)[np.arange(NUM_CLASSES)]
    result = compute_metrics(targets.copy(), targets)
    assert result.accuracy == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.balanced_accuracy == pytest.approx(1.0)


def test_macro_f1_punishes_majority_class_collapse():
    """A model that always predicts the dominant class should look bad on macro-F1."""
    n = 100
    y_true = np.zeros(n, dtype=int)
    y_true[:10] = 1  # 10% minority
    targets = np.eye(NUM_CLASSES)[y_true]
    always_zero = np.tile(np.eye(NUM_CLASSES)[0], (n, 1))

    result = compute_metrics(always_zero, targets)
    assert result.accuracy == pytest.approx(0.9)
    assert result.macro_f1 < 0.15
    assert result.per_class["happiness"]["recall"] == pytest.approx(0.0)


def test_confusion_matrix_shape_and_totals():
    rng = np.random.default_rng(0)
    probs = rng.random((50, NUM_CLASSES))
    targets = np.eye(NUM_CLASSES)[rng.integers(0, NUM_CLASSES, 50)]
    result = compute_metrics(probs, targets)
    assert result.confusion.shape == (NUM_CLASSES, NUM_CLASSES)
    assert result.confusion.sum() == 50


# --------------------------------------------------------------------------- inference


def test_preprocess_matches_the_training_normalisation():
    """A mid-grey crop must map to exactly (0.5 - mean) / std in every channel."""
    spec = PreprocessSpec(
        image_size=32, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), grayscale=True
    )
    crop = np.full((32, 32, 3), 128, dtype=np.uint8)
    batch = preprocess_batch([crop], spec)

    assert batch.shape == (1, 3, 32, 32)
    assert batch.dtype == np.float32
    for channel, (mean, std) in enumerate(zip(spec.mean, spec.std, strict=True)):
        expected = (128 / 255.0 - mean) / std
        assert batch[0, channel].mean() == pytest.approx(expected, abs=1e-4)


def test_preprocess_resizes_mismatched_crops():
    spec = PreprocessSpec(image_size=64, mean=(0.5,) * 3, std=(0.5,) * 3, grayscale=True)
    crops = [np.zeros((20, 30, 3), np.uint8), np.zeros((90, 90, 3), np.uint8)]
    assert preprocess_batch(crops, spec).shape == (2, 3, 64, 64)


def test_softmax_is_stable_on_large_logits():
    probs = softmax(np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32))
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)


def test_smoother_moves_toward_new_observations():
    smoother = ProbabilitySmoother(alpha=0.5)
    first = np.array([1.0, 0.0, 0.0])
    assert np.allclose(smoother.update(0, first), first)  # first observation passes through

    second = smoother.update(0, np.array([0.0, 1.0, 0.0]))
    assert second[0] == pytest.approx(0.5)
    assert second[1] == pytest.approx(0.5)


def test_tracker_keeps_identity_across_small_movements():
    tracker = CentroidTracker(max_distance=60)
    first = tracker.update([(100, 100, 50, 50), (400, 100, 50, 50)])
    second = tracker.update([(105, 102, 50, 50), (403, 98, 50, 50)])
    assert first == second

    far = tracker.update([(100, 100, 50, 50), (1000, 900, 50, 50)])
    assert far[0] == first[0]
    assert far[1] != first[1]  # a jump that large is a new face


# --------------------------------------------------------------------------- alignment


def test_canonical_template_scales_with_size():
    assert np.allclose(canonical_template(112), ARCFACE_TEMPLATE_112)
    assert np.allclose(canonical_template(224), ARCFACE_TEMPLATE_112 * 2.0)


def test_alignment_maps_landmarks_onto_the_template():
    """An identity-position face should land on the template within a pixel."""
    frame = np.zeros((300, 300, 3), np.uint8)
    landmarks = ARCFACE_TEMPLATE_112 + 50.0  # translated copy of the template
    size = 112
    aligned = align_face(frame, landmarks, size, expand=1.0)
    assert aligned is not None and aligned.shape == (size, size, 3)


def test_alignment_corrects_in_plane_rotation():
    """Rotating the landmarks must not change where they land after alignment."""
    import cv2

    base = ARCFACE_TEMPLATE_112 + 100.0
    rotation = cv2.getRotationMatrix2D((156.0, 156.0), 25.0, 1.0)
    rotated = (rotation[:, :2] @ base.T).T + rotation[:, 2]

    target = canonical_template(112, expand=1.0)
    for landmarks in (base, rotated.astype(np.float32)):
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, target, method=cv2.LMEDS)
        mapped = (matrix[:, :2] @ landmarks.T).T + matrix[:, 2]
        assert np.abs(mapped - target).max() < 1.0


def test_crop_face_falls_back_to_the_box_without_landmarks():
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    crop = crop_face(frame, None, (50, 50, 60, 60), size=96)
    assert crop.shape == (96, 96, 3)


def test_crop_face_handles_boxes_that_run_off_frame():
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    crop = crop_face(frame, None, (90, 90, 40, 40), size=64)
    assert crop.shape == (64, 64, 3)


# --------------------------------------------------------------------------- transforms


def test_eval_transform_produces_three_normalised_channels():
    transform = build_eval_transform(64, (0.5,) * 3, (0.5,) * 3, grayscale=True)
    out = transform(torch.randint(0, 255, (1, 48, 48), dtype=torch.uint8))
    assert out.shape == (3, 64, 64)
    assert out.dtype == torch.float32
    # (x/255 - 0.5) / 0.5 lands in [-1, 1].
    assert out.min() >= -1.001 and out.max() <= 1.001


def test_data_config_defaults_are_sane():
    cfg = DataConfig()
    assert cfg.label_mode in {"soft", "majority"}
    assert cfg.balanced_sampler is True
    assert cfg.image_size % 16 == 0


def test_three_channel_transform_expands_and_passes_through():
    transform = ToThreeChannels()
    single = torch.arange(4 * 4, dtype=torch.uint8).reshape(1, 4, 4)
    expanded = transform(single)
    assert expanded.shape == (3, 4, 4)
    # Every channel must be the same image, not a colour conversion.
    assert torch.equal(expanded[0], expanded[1]) and torch.equal(expanded[1], expanded[2])

    already_three = torch.zeros(3, 4, 4, dtype=torch.uint8)
    assert transform(already_three).shape == (3, 4, 4)

    with pytest.raises(ValueError):
        transform(torch.zeros(2, 4, 4, dtype=torch.uint8))


def test_transforms_are_picklable_for_windows_dataloader_workers():
    """Windows spawns worker processes, so every transform must survive pickling."""
    import pickle

    for transform in (
        build_eval_transform(64, (0.5,) * 3, (0.5,) * 3),
        build_train_transform(64, (0.5,) * 3, (0.5,) * 3),
    ):
        restored = pickle.loads(pickle.dumps(transform))
        out = restored(torch.randint(0, 255, (1, 48, 48), dtype=torch.uint8))
        assert out.shape == (3, 64, 64)


def test_head_init_scaling_starts_near_uniform_prediction():
    """A scaled head must start training at roughly ln(num_classes), not far above it."""
    import math

    from torch import nn

    from fer.models import scale_head_init

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(64, NUM_CLASSES)
            # Deliberately oversized, as a pretrained trunk's fresh head often is.
            with torch.no_grad():
                self.fc.weight.normal_(0, 1.0)
                self.fc.bias.normal_(0, 1.0)

        def get_classifier(self) -> nn.Module:
            return self.fc

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    torch.manual_seed(0)
    features = torch.randn(256, 64)
    model = Tiny()
    uniform = torch.full((256, NUM_CLASSES), 1.0 / NUM_CLASSES)

    before = soft_cross_entropy(model(features), uniform).item()
    scale_head_init(model, 0.05)
    after = soft_cross_entropy(model(features), uniform).item()

    assert before > 3.0                                   # oversized head, inflated loss
    # Residual logit spread keeps this a shade above ln(C); the point is the gap to
    # `before` has closed, not that the loss is exactly uniform.
    assert after == pytest.approx(math.log(NUM_CLASSES), abs=0.15)
    assert after < before
    assert torch.equal(model.fc.bias, torch.zeros(NUM_CLASSES))


def test_head_init_scaling_is_a_noop_at_scale_one():
    from torch import nn

    from fer.models import scale_head_init

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, NUM_CLASSES)

        def get_classifier(self) -> nn.Module:
            return self.fc

    model = Tiny()
    original = model.fc.weight.clone()
    scale_head_init(model, 1.0)
    assert torch.equal(model.fc.weight, original)


# --------------------------------------------------------------------------- webcam loop


def _synthetic_face_video(path, frames: int = 12, size=(480, 360)):
    """Render a short clip with one drifting upscaled FER+ face, or skip if unavailable."""
    import cv2

    from fer.data.ferplus import load_ferplus

    try:
        images, votes = load_ferplus("data", split="test")
    except FileNotFoundError:
        pytest.skip("FER+ cache not built; run `fer data` to exercise this test")

    best = int(np.argmax((votes.argmax(1) == 1) * votes.max(1)))  # confident happiness
    face = cv2.cvtColor(
        cv2.resize(images[best], (200, 200), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR
    )
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 15.0, (width, height))
    for t in range(frames):
        frame = np.full((height, width, 3), 40, np.uint8)
        x = 60 + t * 4
        frame[70 : 70 + 200, x : x + 200] = face
        writer.write(frame)
    writer.release()
    return path


def test_webcam_loop_runs_bounded_and_headless_over_a_video(tmp_path):
    """The real loop must run to completion without a display, camera or keypress."""
    model = Path("runs/balanced/best.onnx")
    if not model.exists():
        pytest.skip("no exported model; run `fer train` then `fer export`")

    from fer.webcam import run_webcam

    video = _synthetic_face_video(tmp_path / "clip.mp4")
    out = tmp_path / "annotated.mp4"
    summary = run_webcam(
        model, source=video, headless=True, mirror=False, record=out, max_frames=12
    )

    assert summary["frames"] == 12
    # One face is present in every frame, so detection should not be dropping any.
    assert summary["faces_seen"] >= 10
    assert summary["mean_fps"] > 0
    assert out.exists() and out.stat().st_size > 0


def test_webcam_max_frames_caps_a_longer_source(tmp_path):
    model = Path("runs/balanced/best.onnx")
    if not model.exists():
        pytest.skip("no exported model; run `fer train` then `fer export`")

    from fer.webcam import run_webcam

    video = _synthetic_face_video(tmp_path / "clip.mp4", frames=20)
    summary = run_webcam(model, source=video, headless=True, mirror=False, max_frames=5)
    assert summary["frames"] == 5


def test_webcam_reports_a_clear_error_for_a_missing_video(tmp_path):
    model = Path("runs/balanced/best.onnx")
    if not model.exists():
        pytest.skip("no exported model; run `fer train` then `fer export`")

    from fer.webcam import run_webcam

    with pytest.raises(RuntimeError, match="Could not open video file"):
        run_webcam(model, source=tmp_path / "nope.mp4", headless=True, max_frames=1)


# --------------------------------------------------------------------------- calibration


def test_bias_fit_improves_the_data_it_was_fitted_on():
    """Coordinate ascent must at minimum not make its own objective worse."""
    from fer.calibrate import fit_logit_bias, macro_f1

    rng = np.random.default_rng(0)
    n = 400
    # Heavily skewed labels, like FER+: class 0 dominates, class 2 is rare.
    labels = rng.choice(3, size=n, p=[0.8, 0.15, 0.05])
    logits = rng.normal(0, 1.0, size=(n, 3))
    logits[np.arange(n), labels] += 1.2  # weakly informative
    logits[:, 0] += 1.0  # a prior pull toward the majority class

    before = macro_f1(logits, labels)
    bias, after = fit_logit_bias(logits, labels, rounds=3)
    assert after >= before
    assert bias.shape == (3,)


def test_cross_validation_detects_a_bias_fitted_to_pure_noise():
    """With no signal to find, the held-out gain must not come out positive."""
    from fer.calibrate import cross_validated_gain, fit_logit_bias, macro_f1

    rng = np.random.default_rng(1)
    n = 300
    labels = rng.choice(4, size=n)
    logits = rng.normal(0, 1.0, size=(n, 4))  # independent of the labels

    _, in_sample = fit_logit_bias(logits, labels, rounds=3)
    in_sample_gain = in_sample - macro_f1(logits, labels)
    mean_gain, folds = cross_validated_gain(logits, labels, folds=4)

    # Fitting noise always looks like an improvement on the data it was fitted to...
    assert in_sample_gain > 0
    # ...but must not survive being held out, which is the whole point of the guard.
    assert mean_gain < in_sample_gain
    assert len(folds) == 4


def test_calibrate_refuses_to_fit_on_the_test_split(tmp_path):
    from fer.calibrate import calibrate

    with pytest.raises(ValueError, match="Refusing to calibrate on the test split"):
        calibrate(tmp_path / "missing.pt", split="test")


def test_classifier_applies_a_stored_logit_bias():
    """A bias must shift the prediction, and its absence must be a plain no-op."""
    from fer.infer import EmotionClassifier, softmax

    names = list(CLASS_NAMES)
    plain = EmotionClassifier(
        PreprocessSpec(32, (0.5,) * 3, (0.5,) * 3, True), names, backend="test"
    )
    assert np.allclose(plain.logit_bias, 0.0)

    bias = np.zeros(len(names), dtype=np.float32)
    bias[7] = 5.0  # push hard toward contempt
    biased = EmotionClassifier(
        PreprocessSpec(32, (0.5,) * 3, (0.5,) * 3, True), names, backend="test", logit_bias=bias
    )
    logits = np.zeros((1, len(names)), dtype=np.float32)
    logits[0, 0] = 1.0  # neutral would otherwise win

    assert int(softmax(logits + plain.logit_bias).argmax()) == 0
    assert int(softmax(logits + biased.logit_bias).argmax()) == 7

