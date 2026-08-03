# AI-Enabled-OCR-Document-Processing-System
Computer Vision
# Optical Character Recognition (OCR)

Extract text from images, scanned documents and PDFs with a modular,
config-driven Python pipeline:

```
input (image / PDF) -> preprocessing -> text detection -> text recognition -> validation
```

Verified on Windows (Python 3.12, OpenCV 5). Test suite: **20 passing**.

## Features

- **PDF support** — digital PDFs are read from their embedded text layer
  (fast and exact); scanned PDFs are rendered page-by-page and OCR'd.
  One `.txt` + `.json` per page.
- **3 recognition backends** — Tesseract (classic), EasyOCR (CNN),
  TrOCR (Transformer-based). Optional backends lazy-load so the base
  install stays lightweight.
- **2 detection strategies** — EAST (OpenCV DNN) and EasyOCR (CRAFT)
  bounding-box detection, or no detection for full-image OCR.
- **Preprocessing chain** (`src/preprocess.py`) — grayscale, denoising
  (gaussian/median/bilateral/NLM), binarization (Otsu/adaptive/global),
  and contour-based skew correction.
- **Line-level region merging** — fixes EAST/CRAFT word-fragmentation on
  dense documents; small crops are auto-upscaled before recognition.
- **AVIF/HEIC support** — Pillow fallback when OpenCV can't decode a format.
- **Windows-first ergonomics** — auto-detects the Tesseract binary from
  common install paths, auto-picks files, runnable directly from an IDE
  debugger (no package/relative-import headaches).
- **Batch mode** — OCRs every image/PDF in a folder, one `.txt` + `.json` +
  `_viz.jpg` per file/page; a failing file doesn't stop the batch.
- **Validation** (`src/validation.py`) — Levenshtein distance, CER, WER,
  character accuracy, exact match.
- **Visualization** — detected boxes and per-region confidence drawn onto
  the output image.
- **CLI** (`main.py`) and a structured, JSON-serializable pipeline API
  (`src/pipeline.py`).

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| Image handling | [OpenCV](https://pypi.org/project/opencv-python/) ≥4.8 | I/O, preprocessing, EAST DNN inference, box drawing |
| Arrays | [NumPy](https://pypi.org/project/numpy/) ≥1.24 | Image arrays, geometry, confidence aggregation |
| Image fallback | [Pillow](https://pypi.org/project/pillow/) ≥10.0 | Decodes formats OpenCV can't (AVIF, HEIC, ...) |
| OCR engine | [Tesseract](https://github.com/tesseract-ocr/tesseract) + [pytesseract](https://pypi.org/project/pytesseract/) | Default recognition backend |
| PDF | [PyMuPDF](https://pypi.org/project/PyMuPDF/) | Text-layer extraction, page rendering |
| Deep-learning OCR | [EasyOCR](https://pypi.org/project/easyocr/) *(optional)* | CRAFT-based detection + recognition |
| Transformer OCR | [HuggingFace Transformers](https://pypi.org/project/transformers/) + PyTorch *(optional)* | TrOCR recognizer |
| Config | [PyYAML](https://pypi.org/project/pyyaml/) | `config.yaml` pipeline settings |
| Testing | [pytest](https://pypi.org/project/pytest/) | Unit + end-to-end tests |

**Model:** EAST text detector (`frozen_east_text_detection.pb`, ~90 MB) —
fetched by `scripts/download_east.py`.

## Architecture

```
                       ┌─────────────────────────────────────────────┐
   input image ──────▶ │ 1. PREPROCESS   src/preprocess.py           │
                       │    grayscale, denoise, threshold, deskew    │
                       ├─────────────────────────────────────────────┤
                       │ 2. DETECTION    src/detection.py            │
                       │    EAST (OpenCV DNN) / EasyOCR (CRAFT)      │
                       │    line-box merging, crop upscaling         │
                       ├─────────────────────────────────────────────┤
                       │ 3. RECOGNITION  src/recognition.py          │
                       │    Tesseract / EasyOCR / TrOCR              │
                       ├─────────────────────────────────────────────┤
                       │ 4. VALIDATION   src/validation.py           │
                       │    CER, WER, Levenshtein, char accuracy     │
                       └─────────────────────────────────────────────┘
                                          │
                     output\*.txt  output\*.json  output\*_viz.jpg
```

- **`main.py`** — CLI with single-image and batch modes.
- **`src/pipeline.py`** — `OCRPipeline` orchestrator, `OCRResult` data
  model, batch runner.
- **`config.yaml`** — all knobs (denoise method, threshold, PSM,
  confidence thresholds, backends).

## Installation

Requires Python 3.9+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Tesseract binary** (required for the default backend):

- Windows: `winget install UB-Mannheim.TesseractOCR`
- Ubuntu: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

Optional heavy backends — install only if you need them:

```powershell
pip install -r requirements-easyocr.txt   # deep-learning OCR (PyTorch)
pip install -r requirements-trocr.txt     # TrOCR (PyTorch + Transformers)
```

## Quick start

```powershell
# Generate a synthetic sample image, then OCR it
python scripts/make_sample.py
python main.py samples/sample_printed.png
```

```
Recognized text:
-----------------
Optical Character Recognition
Digitizing documents is fun.
Invoice number: 482913
Total amount: $1,234.56

Confidence: 0.9xx
```

## Batch OCR — one file per image/PDF

Drop any number of images and PDFs into `samples\`, then run with no
arguments. Every file is OCR'd and saved **separately** to `output\`:

```powershell
python main.py                       # OCRs everything in samples\
python main.py --dir C:\my\folder    # OCRs everything in another folder
```

Per-file files written to `output\`:

```
sample_printed.txt      # plain text
sample_printed.json     # boxes + confidence + text
sample_printed_viz.jpg  # boxes drawn on the image (--save-viz)
sample_text_p1.txt      # PDF page 1 text
sample_text_p1.json     # PDF page 1 structured result
```

## PDF extraction

```powershell
python scripts/make_sample_pdf.py      # generate test PDFs (optional)
python main.py samples\sample_text.pdf     # digital PDF -> text layer (exact)
python main.py samples\sample_scanned.pdf  # scanned PDF  -> OCR per page
```

- **Digital PDFs** (embedded text) are read directly — fast and exact.
- **Scanned PDFs** (no text layer) are rendered at 200 DPI and OCR'd,
  one `output\<name>_p<N>.txt` + `.json` per page.
- Mixed PDFs (some pages text, some scanned) are handled automatically
  on a per-page basis.

## Using text detection (EAST)

Download the frozen EAST model (~90 MB), then enable detection:

```powershell
python scripts/download_east.py
python main.py samples/sample_printed.png --detect east --backend tesseract
```

## Validation with ground truth

Create a text file with the expected content, then run:

```powershell
python main.py samples/sample_printed.png --ground-truth ground_truth.txt --save-viz --json-output output/result.json
```

A validation report (CER, WER, char accuracy, exact match) is printed.
Tune the `min_confidence` / `nms_threshold` values in `config.yaml` to
improve detection precision, and `psm` to adapt Tesseract to your page
layout.

## CLI reference

```
positional:
  image                  Path to the input image or PDF (optional).

options:
  --config FILE          Alternative config.yaml.
  --backend {tesseract,easyocr,trocr}
  --detect {none,east,easyocr}
  --lang LANG            e.g. eng / en
  --ground-truth FILE    Validate against expected text.
  --save-viz             Save box visualization to output/.
  --json-output FILE     Write structured results as JSON.
  --no-preprocess        Skip image preprocessing.
  --dir FOLDER           Batch: OCR every image/PDF in a folder.
  --out-dir FOLDER       Where batch results go (default: output/).
```

## Project layout

```
.
├── main.py                 # CLI entry point
├── config.yaml             # pipeline configuration
├── requirements*.txt       # dependency sets (core / easyocr / trocr)
├── src/
│   ├── preprocess.py       # denoise, threshold, deskew
│   ├── detection.py        # EAST / EasyOCR text detection
│   ├── recognition.py      # Tesseract / EasyOCR / TrOCR
│   ├── validation.py       # CER, WER, accuracy metrics
│   └── pipeline.py         # end-to-end orchestrator + PDF runner
├── scripts/
│   ├── make_sample.py      # generate a synthetic test image
│   ├── make_sample_pdf.py  # generate sample text/scanned PDFs
│   └── download_east.py    # fetch EAST model -> models/
├── tests/                  # pytest suite
├── samples/                # sample images & PDFs
├── models/                 # downloaded detection models
└── output/                 # per-file results (txt/json/viz)
```

## Testing

```powershell
pytest -q
```

## Extensions

- **Real-time OCR** — run the pipeline on video frames; on embedded
  devices use Tesseract or a quantized EasyOCR/TrOCR model.
- **Multilingual support** — Tesseract: download additional tessdata
  (`pip install` + set `lang`, e.g. `ara`, `chi_sim`, `hin`). EasyOCR:
  change the reader `lang`. TrOCR: swap in a multilingual model such as
  `microsoft/trocr-large-printed` or a fine-tuned variant.
- **Document layouts** — replace the naive row sort in detection with
  reading-order layout analysis (e.g. contour clustering or a LayoutLM
  model).
