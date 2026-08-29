# Oral Cancer Classifier — Backend

FastAPI inference service for binary oral cancer classification (Cancer /
Non-Cancer) from intraoral photographs, using a trained EfficientNet-B0.

> **Research prototype — not a diagnostic device.** Not validated for clinical
> use and not approved by any regulatory body. Output must not be used for
> clinical decisions or to replace evaluation by a qualified clinician.

---

## Contents

- [Quick start](#quick-start)
- [The checkpoint](#the-checkpoint)
- [Configuration](#configuration)
- [Preprocessing pipeline](#preprocessing-pipeline)
- [API reference](#api-reference)
- [Verifying your model](#verifying-your-model)
- [Tests](#tests)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

---

## Quick start

Requires Python 3.10+ (3.12 recommended — see [runtime.txt](runtime.txt)).

```bash
cd backend

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env              # defaults already match the EXP-4A checkpoint
```

Place your `.pkl` in `models/`, then:

```bash
uvicorn app.main:app --reload --port 8000
```

Startup is healthy when you see this line — **not** merely "Application startup
complete":

```
Model ready: state_dict from 'state_dict' via pickle.load | 1 output unit(s) |
activation=sigmoid | threshold=0.50 | positive index 1 = 'Cancer' | warmup 330.5 ms
```

Then:

```bash
curl localhost:8000/api/health
# {"status":"ok","model_loaded":true,"device":"cpu"}
```

Interactive API docs: <http://localhost:8000/docs>

### On CPU-only torch

`requirements.txt` pins `torch==2.13.0+cpu` via an extra index. This is
deliberate: installing plain `torch` from PyPI on Linux pulls the ~2.5 GB CUDA
build, which blows past the build limits on Render and Railway free tiers. The
CPU wheels are ~200 MB.

On **macOS or Windows**, drop the `+cpu` suffixes and the `--extra-index-url`
line — those platforms ship CPU-only wheels on PyPI already. On a **CUDA host**,
substitute the matching CUDA channel from
[pytorch.org](https://pytorch.org/get-started/locally/) and drop the suffixes.

---

## The checkpoint

This service was built against `EfficientNetB0_Dataset02_OralCancer_EXP4A.pkl`,
which records its own provenance:

| Field | Value |
| --- | --- |
| Architecture | `torchvision.models.efficientnet_b0` |
| Head | `classifier.1` → `Linear(1280, 1)` |
| Output | **single raw logit** (`output_type: single_logit`) |
| Activation | sigmoid, applied **by this service** — the checkpoint does not apply it |
| Classes | `{0: NON CANCER, 1: CANCER}` |
| Input | 224 × 224 RGB, ImageNet mean/std |
| Threshold | 0.5 |
| Training set | `Dataset02_resized` (external: `Dataset01_resized`) |
| Best epoch / val F1 | 11 / 0.9417 |

Two properties of this file are worth knowing because they are **not** what the
extension suggests:

1. **It is a plain `pickle.dump` stream, not `torch.save` format.** `torch.load`
   rejects it outright with `RuntimeError: Invalid magic number; corrupt file?`.
   The loader falls back to a bare `pickle.load`, and logs which path it used.
2. **The recorded `activation: sigmoid` means sigmoid must be applied here.**
   Training used `BCEWithLogitsLoss`, which applies sigmoid internally, so the
   saved model emits raw logits. A dummy forward pass spans `[-1406, +242]`,
   confirming this. Setting `MODEL_OUTPUTS_PROBABILITY=true` would squash every
   prediction toward 0.5 — the service raises an error rather than doing so
   silently.

### Loading a different checkpoint

The loader is deliberately defensive and handles, in order:

1. A full pickled `nn.Module` → used directly.
2. A `state_dict`, flat or nested under `state_dict` / `model_state_dict` /
   `model` / `net` / `weights` → EfficientNet-B0 is rebuilt, the head resized to
   match, and weights loaded with `strict=False` (missing/unexpected keys logged).
3. Keys prefixed `module.` (DataParallel), `model.`, or `_orig_mod.` → stripped.
4. A pickle referencing a class that no longer exists → retried with a tolerant
   unpickler that substitutes stubs, then recovers the tensors.

Head shape drives post-processing automatically: **1 unit → sigmoid**,
**2 units → softmax**. Anything else is rejected as non-binary.

Metadata inside the checkpoint (`class_names`, `mean`, `std`, `image_size`)
takes **precedence over `.env`**, with a warning logged on any disagreement, so
serving config cannot silently drift from how the model was trained.

### Inspecting an unknown checkpoint

Before trusting any checkpoint, run the standalone diagnostic:

```bash
python scripts/inspect_model.py models/your_model.pkl
python scripts/inspect_model.py models/your_model.pkl --json report.json
```

It reports which loader succeeded, the object type, state_dict structure and key
prefixes, the classifier shape (→ number of output units), any embedded
metadata, the result of a `strict=False` reconstruction, and a multi-probe
forward pass that determines whether outputs are raw logits or already
activated. It ends with the exact `.env` values to use.

The probe uses **five inputs of increasing magnitude**, not one. A single zeros
input can land inside `[0, 1]` by chance and falsely read as "already
activated"; the script requires corroborating evidence (a trailing
`Sigmoid`/`Softmax` module, or 2-unit outputs summing to 1 on every probe) and
reports `AMBIGUOUS` rather than guessing.

---

## Configuration

All settings come from `.env` (see [.env.example](.env.example)). Every value is
optional; defaults match the EXP-4A checkpoint.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `models/EfficientNetB0_…EXP4A.pkl` | Checkpoint location. Relative paths resolve against `backend/`, **not** the working directory. |
| `MODEL_URL` | *(empty)* | Direct-download link, fetched at startup when `MODEL_PATH` is absent. For deployment, where weights are gitignored. |
| `MODEL_SHA256` | *(empty)* | Optional integrity check on that download. |
| `DEVICE` | `auto` | `auto` \| `cpu` \| `cuda`. `auto` uses CUDA when available. |
| `THRESHOLD` | `0.5` | P(Cancer) at or above this is reported as Cancer. |
| `POSITIVE_CLASS_INDEX` | `1` | Which index means Cancer. Verified against checkpoint metadata. |
| `MODEL_OUTPUTS_PROBABILITY` | `false` | `true` skips the activation. Leave `false` for EXP-4A. |
| `INPUT_SIZE` | `224` | Model input edge length. |
| `RESIZE_MODE` | `resize` | `resize` → `Resize((224,224))`; `resize_crop` → `Resize(256) + CenterCrop(224)`. |
| `CENTER_CROP_RATIO` | `0.875` | 224/256, used only by `resize_crop`. |
| `MAX_UPLOAD_MB` | `10` | Per-file size cap → `413`. |
| `MAX_BATCH_FILES` | `10` | Batch endpoint limit → `413`. |
| `PORT` | `8000` | Local only; hosts inject `$PORT`. |
| `ALLOWED_ORIGINS` | localhost `5173`/`5174`/`5175` | Comma-separated CORS origins. |
| `ALLOWED_ORIGIN_REGEX` | *(empty)* | For origins that can't be enumerated (Vercel previews). Must match the whole origin. |
| `LOG_LEVEL` | `INFO` | Standard Python levels. |

> Vite's fallback ports (5174, 5175) are allowed by default because Vite
> silently moves to them when 5173 is taken, which otherwise breaks every API
> call with a CORS error that *looks* like the backend being down.

---

## Preprocessing pipeline

Built **once** at import as a module-level `transforms.Compose`, never
per request.

```
  upload bytes
       │
       ▼
  PIL.Image.open              decode; validates by decoding, not by
       │                      trusting the client's Content-Type
       ▼
  ImageOps.exif_transpose     apply camera orientation BEFORE resize,
       │                      or the crop would be taken from a rotated frame
       ▼
  .convert("RGB")             drop alpha, expand greyscale → 3 channels
       │
       ▼
  Resize((224, 224))          ← RESIZE_MODE switches this line
       │
       ▼
  ToTensor()                  → float32 CHW, scaled to [0, 1]
       │
       ▼
  Normalize(mean, std)        ImageNet: mean [0.485, 0.456, 0.406]
       │                                std  [0.229, 0.224, 0.225]
       ▼
  .unsqueeze(0)               → (1, 3, 224, 224)
```

**Assumption to check against your training notebook:** the checkpoint records
its *training* augmentation but not its *eval* transform. `RESIZE_MODE=resize`
is used because EXP-4A trained on an already-resized 224 px dataset. If your
validation transform was `Resize(256) + CenterCrop(224)`, set
`RESIZE_MODE=resize_crop` — no code change. A mismatch here quietly degrades
accuracy rather than raising an error.

---

## API reference

Base URL `http://localhost:8000`. Every failure returns the same envelope:

```json
{ "error": { "code": "INVALID_IMAGE", "message": "…" } }
```

### `GET /api/health`

Always returns **200** so a client can distinguish "backend down" (network
error) from "backend up, model broken".

```json
{ "status": "ok", "model_loaded": true, "device": "cpu" }
```

`status` is `degraded` and `model_loaded` is `false` when the checkpoint failed
to load. **A 200 alone does not mean the service is usable — check
`model_loaded`.**

### `GET /api/model-info`

Everything the server discovered about the loaded checkpoint: architecture,
device, input size, output units, activation, threshold, class names,
normalisation, accepted MIME types, how the checkpoint was read, missing and
unexpected key counts, and the full embedded metadata. Use it to audit a
deployment without shell access. The filesystem path is not exposed.

### `POST /api/predict`

`multipart/form-data`, field name **`file`**.

```bash
curl -X POST localhost:8000/api/predict -F "file=@lesion.jpg"
```

```json
{
  "prediction": "Cancer",
  "probability": 0.8734,
  "confidence": 0.8734,
  "threshold": 0.5,
  "raw_output": 1.9231,
  "inference_time_ms": 42.7,
  "image": {
    "original_size": [3024, 4032],
    "processed_size": [224, 224],
    "exif_corrected": true,
    "format": "JPEG",
    "mode": "RGB"
  }
}
```

- `probability` — **P(Cancer)**, regardless of which label won.
- `confidence` — probability of the *predicted* class (`p` if Cancer, else
  `1 − p`). At the default threshold this is never below 0.5. Note it reads
  oddly at a non-default threshold: with `THRESHOLD=0.05` and `p = 0.12` you get
  "Cancer, 12% confidence", which is correct but worth knowing if you tune the
  threshold for recall.
- `raw_output` — the pre-activation logit.

### `POST /api/predict/batch`

Field name **`files`**, up to `MAX_BATCH_FILES`. One bad file yields an error
entry for that file only; the rest still return results.

```json
{
  "total": 3, "succeeded": 2, "failed": 1,
  "results": [
    { "filename": "a.jpg", "success": true,  "result": { "…": "…" }, "error": null },
    { "filename": "b.jpg", "success": false, "result": null,
      "error": { "code": "INVALID_IMAGE", "message": "…" } }
  ]
}
```

### Status codes

| Code | Meaning |
| --- | --- |
| `200` | Success |
| `413` | `FILE_TOO_LARGE`, `TOO_MANY_FILES` |
| `415` | `UNSUPPORTED_MEDIA_TYPE` — decoded fine but not an accepted format |
| `422` | `INVALID_IMAGE`, `EMPTY_UPLOAD`, `VALIDATION_ERROR` |
| `500` | `INFERENCE_FAILED`, `INTERNAL_ERROR` |
| `503` | `MODEL_NOT_LOADED`, `MODEL_LOAD_FAILED` |

Accepted formats: JPEG, PNG, WebP, BMP, TIFF — determined by **decoding the
bytes with PIL**, never by the declared `Content-Type`.

### Privacy

Uploads are held in memory for the request and **never written to disk, logged,
or cached**. These may be patient photographs. Only image dimensions and the
prediction are logged. Each response carries an `X-Request-ID` correlating it to
the log line.

---

## Verifying your model

The one thing checkpoint inspection *cannot* prove is whether the class mapping
is inverted. An inverted mapping produces confidently wrong predictions with no
error anywhere — the single most dangerous failure mode in this service.

For EXP-4A the checkpoint settles it (`class_names: {0: NON CANCER, 1: CANCER}`),
but **verify empirically before trusting any checkpoint**:

1. **Gather labelled images.** Take 5–10 known-Cancer and 5–10 known-Non-Cancer
   images from your dataset. Prefer the **validation or test split** — images
   the model trained on will score well even if the mapping is wrong.

2. **Run them through the API:**

   ```bash
   for f in known_cancer/*.jpg; do
     printf '%-30s ' "$(basename "$f")"
     curl -s -X POST localhost:8000/api/predict -F "file=@$f" \
       | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['prediction'], round(d['probability'],3))"
   done
   ```

3. **Read the result:**

   | Observation | Meaning |
   | --- | --- |
   | Cancer images → high `probability`, Non-Cancer → low | ✅ Mapping correct |
   | Cancer images → **low** `probability`, Non-Cancer → **high** | ❌ **Inverted.** Set `POSITIVE_CLASS_INDEX=0` and re-verify |
   | Everything near 0.5 | Preprocessing mismatch — try `RESIZE_MODE=resize_crop`, and confirm normalisation matches training |
   | Everything the same class | Threshold or a genuinely skewed model — check `raw_output` varies between images |

   A systematic inversion is unmistakable: accuracy sits near **1 − expected**,
   not near chance. If your labelled set scores ~6% accuracy, the mapping is
   flipped, not the model broken.

4. **Cross-check against your training notebook.** If it used
   `ImageFolder`, the mapping is alphabetical: `cancer` → 0, `non_cancer` → 1 —
   the *opposite* of this checkpoint's recorded order. Confirm with
   `print(train_dataset.class_to_idx)`.

5. **Confirm the threshold** with `curl localhost:8000/api/model-info`, which
   echoes the checkpoint's own `decision_threshold`.

---

## Tests

```bash
pytest                # 140 tests, ~3s
pytest -v             # per-test names
pytest tests/test_inference.py -v
```

The suite runs **without the checkpoint present** — startup loading is patched
out in fixtures and a stub model is injected, so it is fast and deterministic.
The one test that exercises the real `.pkl` skips automatically when it is
absent.

Coverage of note:

| File | Focus |
| --- | --- |
| `test_preprocessing.py` | Tensor shape/dtype, normalisation constants, EXIF rotation reaching the model input (with an unrotated control), format acceptance and rejection |
| `test_inference.py` | sigmoid vs softmax branching, threshold boundary, confidence, **inverted class names**, misconfigured `MODEL_OUTPUTS_PROBABILITY` raising rather than corrupting |
| `test_model_loader.py` | Both serialisation formats, prefix stripping, head inference, class-mapping normalisation, download-on-boot, CORS config |
| `test_api.py` | Happy path, every error code and status, batch isolation, CORS, content-type spoofing |

Note that `softmax([a,b])[1] ≡ sigmoid(b−a)`, so branch-discrimination tests use
logits where the two genuinely differ.

---

## Deployment

Weights are gitignored, so a fresh deploy has no `.pkl`. Two options:

**A. Fetch at boot (recommended).** Upload the file somewhere with a *direct*
download link (GitHub release asset, S3, R2) and set:

```
MODEL_URL=https://…/EfficientNetB0_Dataset02_OralCancer_EXP4A.pkl
MODEL_SHA256=1bba670b960ab3fd53160ce59d524d875093f2d19a3f79a37d3717d5b0e19ab4
```

It downloads to a `.part` file and renames only on success, so an interrupted
transfer can never leave a truncated checkpoint. A Google Drive or Dropbox
*sharing page* returns HTML rather than the file — this is detected and rejected
with a clear message.

**B. Commit the file.** 16 MB is well within GitHub limits, if you're comfortable
having weights in git.

### Render

Blueprint at [`../render.yaml`](../render.yaml). Root directory `backend`,
health check `/api/health`. Set `MODEL_URL`, `MODEL_SHA256`, and
`ALLOWED_ORIGINS` (your Vercel URL) in the dashboard.

> The free tier's 512 MB RAM is too tight — torch alone approaches it before the
> model loads. `render.yaml` specifies `starter`.

### Railway

[`../railway.json`](../railway.json) builds from this `Dockerfile`. Railway's
usage-based pricing handles the memory footprint more comfortably.

### Docker

```bash
docker build -t oral-cancer-api ./backend
docker run -p 8000:8000 -v "$(pwd)/backend/models:/app/models" oral-cancer-api
```

Multi-stage, runs as a non-root user (this service accepts network file
uploads), with a healthcheck that reports unhealthy while the model is missing.
Weights are **not** baked into the image — mount them or use `MODEL_URL`.

### CORS in production

Add your Vercel production URL to `ALLOWED_ORIGINS`. Preview deployments get a
new hostname per commit, so match them with a regex:

```
ALLOWED_ORIGIN_REGEX=https://your-app-.*-yourteam\.vercel\.app
```

---

## Troubleshooting

**`/api/health` returns 200 but the app says the model isn't ready.**
Check `model_loaded` in the JSON, not the status code. The startup log says why.
A common cause was a relative `MODEL_PATH` resolving against the working
directory — now fixed, paths resolve against `backend/` regardless of where you
launch uvicorn from.

**Frontend says "cannot reach the server" but `curl` works.**
Almost always CORS. The browser refuses to hand a cross-origin response to
JavaScript when the origin isn't approved, so the client sees *no response* —
indistinguishable from a dead server. Check the browser Console for
`blocked by CORS policy`, then confirm the page's port appears in
`ALLOWED_ORIGINS`. Vite silently falls back to 5174/5175 when 5173 is taken.

**`RuntimeError: Invalid magic number; corrupt file?`**
The file is a plain pickle, not `torch.save` format. Handled automatically —
if you see this raised, run `scripts/inspect_model.py` on the file.

**`ModuleNotFoundError` / `AttributeError` on load.**
The pickle references a class from your training notebook that isn't importable
here. The tolerant unpickler tries to recover the weights; if it can't, re-export
from your training environment:

```python
torch.save(model.state_dict(), "weights.pth")
```

**All predictions cluster near 0.5.** Preprocessing mismatch. Try
`RESIZE_MODE=resize_crop`, and confirm normalisation matches training via
`/api/model-info`.

**Build pulls gigabytes of `nvidia-*` packages.** `--extra-index-url` was
dropped from `requirements.txt` or the `+cpu` pins were removed.

---

## Project layout

```
backend/
├── app/
│   ├── config.py          pydantic-settings; paths anchored to backend/
│   ├── exceptions.py      error types + handlers (one JSON envelope)
│   ├── inference.py       forward pass + adaptive post-processing
│   ├── main.py            FastAPI app, lifespan, CORS, routes, logging
│   ├── model_loader.py    defensive checkpoint loading, singleton, warmup
│   ├── preprocessing.py   the transform pipeline (built once)
│   └── schemas.py         Pydantic request/response models
├── scripts/
│   └── inspect_model.py   standalone checkpoint diagnostic
├── models/                .pkl goes here (gitignored)
├── tests/                 140 tests
├── Dockerfile             multi-stage, non-root
├── Procfile               web: uvicorn … --port $PORT
├── requirements.txt       runtime deps (CPU torch)
└── requirements-dev.txt   pytest, httpx
```

The model is loaded **once** in the FastAPI lifespan and held as a module-level
singleton — never per request. A load failure does not crash the process; the
app starts degraded so `/api/health` can report the problem instead of the
container restart-looping with the traceback buried in the logs. The blocking
forward pass runs in a threadpool so the event loop stays responsive.
