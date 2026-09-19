# Facial Emotion Recognition

Real-time facial emotion recognition: a FER+ training pipeline built on PyTorch and
`timm`, ONNX export, and a webcam app that detects, aligns and classifies every face in
the frame.

Eight classes — neutral, happiness, surprise, sadness, anger, disgust, fear, contempt.

```bash
git clone https://github.com/rebantamandal/facial-emotion-recognition.git
cd facial-emotion-recognition
uv sync                                   # install
fer data                                  # build FER+ (~35k images)
fer train -c configs/balanced.yaml        # fine-tune, ~18 min on a mid-range GPU
fer export runs/balanced/best.pt          # -> ONNX
fer webcam runs/balanced/best.onnx        # live demo
```

To skip training, grab `best.onnx` and `best.json` from the
[latest release](https://github.com/rebantamandal/facial-emotion-recognition/releases/latest)
and drop them in `runs/balanced/`. Keep the two files together — the `.json` carries the
preprocessing, and the model gives wrong answers without it.

## Results

<!--RESULTS-->

**MobileNetV4-conv-medium** at 224px, on the **test** split (FER2013's PrivateTest,
which model selection never touches):

| Metric | Score |
|---|---|
| Accuracy | **0.833** |
| Macro-F1 | **0.696** |
| Balanced accuracy | 0.678 |

3.1 ms per face on CPU via ONNX Runtime, 34 MB exported, 46 FPS in the live loop.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neutral | 0.850 | 0.848 | 0.849 | 1254 |
| happiness | 0.920 | 0.928 | 0.924 | 925 |
| surprise | 0.821 | 0.888 | 0.853 | 438 |
| sadness | 0.690 | 0.683 | 0.686 | 439 |
| anger | 0.803 | 0.803 | 0.803 | 325 |
| disgust | 0.409 | 0.409 | 0.409 | 22 |
| fear | 0.657 | 0.478 | 0.553 | 92 |
| contempt | 0.667 | 0.385 | 0.488 | 26 |

Accuracy is the weaker number to quote: the two largest classes are over half the
dataset, so macro-F1 is what tracks whether the rare emotions are learned at all.
`contempt` and `disgust` have 26 and 22 test images, so their per-class scores swing
a lot between runs — which is also why calibrating against them fails (see below).

Reproduce with `fer data && fer train -c configs/balanced.yaml && fer eval runs/balanced/best.pt --split test --out runs/balanced`.

<!--/RESULTS-->

## How it works

```
webcam frame
   └─ YuNet detector ──────────► box + 5 landmarks per face
        └─ similarity align ───► canonical 112-point template crop
             └─ batched CNN ───► 8-way softmax
                  └─ EMA smoothing per tracked face ──► label
```

**Detection.** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
a 230 KB CNN from the OpenCV model zoo, downloaded and cached on first run. It is far
more robust than a Haar cascade at non-frontal angles, and it returns five facial
landmarks.

**Alignment.** Those landmarks drive a similarity transform onto the standard ArcFace
five-point template ([`align.py`](src/fer/align.py)). Warping out in-plane head rotation
means the classifier does not have to spend capacity learning to be rotation-invariant.
Rotation, scale and translation only — a full affine would shear the face and distort the
expression itself. Without landmarks it falls back to a padded box crop, since a tight
box clips the brow and jaw where much of the expression signal lives.

**Classification.** Any `timm` backbone, set in the config. The default is
MobileNetV4-conv-medium, fine-tuned end to end. Preprocessing statistics are read off the
checkpoint via `timm.data.resolve_model_data_config` rather than hardcoded, so the input
distribution always matches what the backbone was pretrained on.

**Smoothing.** Raw per-frame predictions flicker at expression boundaries. A centroid
tracker keeps identity across frames and probabilities are EMA-smoothed per face, which
stabilises the label without perceptible lag.

## The data

FER+ ([Barsoum et al., 2016](https://arxiv.org/abs/1608.01041)) re-annotates every
FER2013 image with ten crowd votes. It is a strict improvement over FER2013's original
single labels, which are noisy enough to cap achievable accuracy.

FER+ distributes **labels only**. The pixels come from a FER2013 mirror, and the two are
aligned purely by row order — nothing checksums that assumption. So
[`ferplus.py`](src/fer/data/ferplus.py) verifies it before writing anything: FER+ majority
labels are compared against FER2013's own labels, which must agree far above chance
(~1/7). Below a 50% floor the build aborts rather than silently training on scrambled
labels.

Two details that matter more than they look:

- **Targets are vote distributions, not single labels.** The FER+ paper shows training on
  the full distribution beats collapsing it to an argmax, and plenty of faces are
  genuinely ambiguous. One-hot is just the degenerate case, so label smoothing and mixup
  flow through the same soft-target loss. Set `label_mode: majority` to compare.
- **The dataset is severely imbalanced.** Happiness and neutral dominate; contempt is
  well under 1%. The training loader uses an inverse-frequency sampler, and model
  selection keys off **macro-F1**, not accuracy — a model that never predicts contempt
  still scores well on accuracy.

Splits follow FER2013's own: `train` (Training, 28,709), `val` (PublicTest, 3,589),
`test` (PrivateTest, 3,589). Training selects on `val`; `test` is only ever scored once.

## Training recipe

Both configs use AdamW with cosine decay after linear warmup, bf16 autocast, and:

| | |
|---|---|
| Discriminative LR | backbone at 0.1x the head's — the head is randomly initialised, the trunk is not |
| No weight decay on norms/biases | decaying 1-D parameters fights the normalisation statistics |
| Mixup + CutMix | `p=0.5` per batch, targets mixed by the same factor |
| Label smoothing | 0.1, folded into the target vector |
| EMA weights | `decay=0.9998` with warmup; the averaged weights are what gets scored and exported |
| Augmentation | flip, affine jitter, brightness/contrast, blur, random erasing — geometry and intensity only, since the source images are grayscale |
| Early stopping | on val macro-F1 |
| Head init scaling | a fresh head on this backbone emits logits with std ~5, starting training at loss ~7 instead of ln(8)=2.08 and firing large gradients into the pretrained trunk; scaling it down starts from an honest uniform prior |

`torch.compile` is enabled by default and falls back to eager with a warning where the
backend is unavailable — notably on Windows, where Triton is not shipped. The fallback
runs a throwaway forward pass at startup, because `torch.compile` is lazy and would
otherwise fail several minutes into training.

Two configs ship: `balanced.yaml` is the default (224px, MobileNetV4-conv-medium) and
`fast.yaml` halves the cost per face again (112px, MobileNetV4-conv-small: 1.6 ms versus
3.1 ms). FER+ images are natively 48×48, so 112px upscales far less aggressively than
224 does.

`sampler_power` controls how hard the class rebalancing pushes; the shipped checkpoint
was trained at 1.0 (full inverse frequency).

## CLI

| Command | Purpose |
|---|---|
| `fer data` | Download, merge, verify and cache FER+ |
| `fer train -c configs/balanced.yaml` | Fine-tune; writes `best.pt`, curves, confusion matrix |
| `fer eval runs/balanced/best.pt --split test` | Per-class precision/recall/F1 report (`--tta`, `--raw`) |
| `fer calibrate runs/balanced/best.pt` | Fit a macro-F1-optimal decision rule, kept only if it cross-validates |
| `fer export runs/balanced/best.pt` | Single self-contained ONNX + preprocessing sidecar, verified against PyTorch |
| `fer bench runs/balanced/best.onnx` | CPU latency and throughput |
| `fer webcam runs/balanced/best.onnx` | Live demo — `q` quit, `b` bars, `m` mirror |
| `fer webcam model.onnx --headless --max-frames 120 --record out.mp4` | Bounded run with no window, for testing or over SSH |
| `fer image model.onnx photo.jpg --out out.jpg` | Annotate a still image |

Every command takes `--help`. Useful flags: `--device cuda`, `--detect-every N` to run
detection on a stride, `--smoothing` to tune the EMA, `--record out.mp4`, and
`--source clip.mp4` to run the pipeline over a video file instead of the camera.

### Measured on the webcam loop

At 1280x720 on CPU with the balanced model: **32 FPS** with no face in frame, **42 FPS**
on a 960x540 clip with two faces (both scored in one batched forward pass). On a
150-frame clip holding two known faces, detection fired on 300/300 and the tracker held
exactly two identities throughout, labelling both correctly on every frame.

## As a library

```python
from fer.align import crop_face
from fer.detect import FaceDetector
from fer.infer import EmotionClassifier

detector = FaceDetector()
classifier = EmotionClassifier.load("runs/balanced/best.onnx")

faces = detector.detect(frame)                      # BGR numpy array
crops = [crop_face(frame, f.landmarks, f.box, classifier.spec.image_size) for f in faces]
probs = classifier.predict_batch(crops)             # (N, 8), one batched forward pass
```

`EmotionClassifier.load` dispatches on extension — `.onnx` goes through ONNX Runtime,
`.pt` through PyTorch — and reproduces the training preprocessing in both cases.

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) tours the same API: dataset inspection,
augmentation previews, evaluation and single-image prediction. Run the webcam from a
terminal rather than a notebook cell; `cv2.imshow` needs a real window loop.

## Layout

```
configs/            balanced.yaml, fast.yaml
src/fer/
  labels.py         class definitions
  config.py         typed config, YAML-backed
  data/ferplus.py   dataset build + alignment verification
  data/datamodule.py  transforms, sampler, dataloaders
  models.py         timm factory, param groups, checkpoint I/O
  engine.py         train/eval loops, mixup, EMA, schedule
  metrics.py        macro-F1, confusion matrix, plots
  detect.py         YuNet wrapper
  align.py          landmark alignment
  infer.py          torch/ONNX predictor, tracker, smoother
  webcam.py         real-time app
  cli.py            fer <command>
tests/              unit tests, no dataset or GPU needed
```

## Requirements

Python 3.12–3.13. `uv sync` pins CUDA 13.0 wheels for PyTorch; for CPU-only or a
different CUDA version, change the `tool.uv.index` URL in `pyproject.toml`. Training
needs a GPU to be pleasant; inference is real-time on CPU via ONNX.

```bash
uv run pytest        # test suite
uv run ruff check .  # lint
```

## A negative result worth keeping

`argmax` is optimal for accuracy but not for macro-F1: on a set where contempt is 0.6% of
the data, it can pay to predict a rare class on weaker evidence, because the recall gained
on a tiny class moves the macro average more than the precision lost on a large one. So
`fer calibrate` fits a per-class logit bias by coordinate ascent directly on macro-F1.

On FER+ **it does not work**, and the interesting part is how that is established. Fitted
on validation it looks like a clear win — macro-F1 0.741 → 0.755. Cross-validated within
validation it is **−0.012**, and on the test split it does indeed lose: 0.696 → 0.692.
Validation holds 24 contempt and 34 disgust images, few enough that a per-class bias just
memorises them.

So `fer calibrate` refuses to store a fit that does not survive a 5-fold check, and clears
any bias a previous run left behind. The command is kept because the technique is sound
and would pay off on a larger or less skewed dataset; the guard is what stops it being a
silent regression here. `--force` overrides it, and `fer eval --raw` ignores any stored
bias. Flip-averaged TTA (`--tta`) is real but tiny: +0.002 macro-F1 for double the compute.

## Why it says *neutral* so much

Two separate reasons, one fixed and one inherent.

**The crop was too wide.** Alignment defaulted to `expand=1.25`, on the reasoning that
brow and jaw carry expression so more context should help. Measured on held-out sad and
angry faces, that was wrong: 68.3% correct at 1.25 against 74.1% at 1.0, with a quarter
of them falling into *neutral* instead. FER2013's own framing is tight — the face fills
almost the whole image — and matching it beats giving the model more to look at. The
default is now 1.0.

**FER2013 expressions are theatrical.** The training faces are posed and exaggerated:
bared teeth for anger, a wide-open mouth for surprise. A real person looking mildly
unhappy at a webcam sits, in this model's feature space, much closer to FER2013's
*neutral* than to its *sadness*. So the webcam demo needs the expression played up to
register — that is a property of the dataset, not a bug, and it is the main reason a
model with 83% test accuracy can feel unresponsive in the mirror.

All eight classes are drawn as bars, so a genuinely uncertain reading looks uncertain
rather than like a confident wrong answer. `--top-k 3` shows fewer.

## Limitations

FER+ is small, posed, mostly Western, and 48×48 — the upper bound on what any model
trained purely on it can do is modest, and accuracy on real webcam footage is lower than
the test-set numbers suggest. Categorical emotion labels are themselves contested: facial
expressions do not map cleanly onto discrete internal states, especially across cultures.
Treat the output as "this face resembles the training set's *happiness* category," not as
a read of what someone actually feels. Not suitable for any consequential decision about
a person.

## License

MIT — see [LICENSE](LICENSE).
