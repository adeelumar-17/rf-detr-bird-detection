# 🦅 AeroWatch — Airport Bird Detection

A complete bird detection system built for airport runway safety. The project covers everything from dataset preparation to model training to a live detection web app.

Birds near airport runways are a real danger — [bird strikes cause over $1.2 billion in damage annually](https://www.faa.gov/airports/airport_safety/wildlife) to the aviation industry. This system detects birds in images and video feeds so airport operators can assess risk in real time.

---

## What This Project Does

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Collect &   │     │  Fine-tune   │     │  Export to   │     │  Streamlit   │
│  Prepare     │────▶│  RF-DETR     │────▶│  ONNX        │────▶│  Web App     │
│  Dataset     │     │  Model       │     │  (CPU-ready) │     │  (AeroWatch) │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
   Notebook 1           Notebook 2           Notebook 2            app.py
```

1. **Dataset Preparation** — Merges bird images from COCO 2017 and Open Images v7 into one clean dataset
2. **Model Training** — Fine-tunes RF-DETR (a state-of-the-art object detector) on that dataset
3. **ONNX Export** — Converts the trained model to ONNX so it runs on any CPU without needing PyTorch
4. **Web App** — A Streamlit app that lets you upload images or videos and get instant bird detections with risk assessment

---

## Project Structure

```
bird/
├── airport_bird_dataset_prep.ipynb   # Step 1: Build the dataset (run on Kaggle)
├── rfdetr_finetune_bird_detection.ipynb  # Step 2: Train + evaluate + export (run on Kaggle)
├── app.py                            # Step 3: Streamlit detection app
├── requirements.txt                  # Python dependencies for the app
├── .env                              # HuggingFace token (local dev only, not committed)
└── inference_model.onnx              # Trained ONNX model (~110 MB)
```

---

## How Each Part Works

### Step 1 — Dataset Preparation (`airport_bird_dataset_prep.ipynb`)

This notebook builds a bird-only detection dataset from two public sources:

| Source | Train Images | Val Images | What It Provides |
|--------|-------------|------------|------------------|
| [COCO 2017](https://cocodataset.org/) | ~4,000 | ~500 | High-quality annotated bird images (category 16) |
| [Open Images v7](https://storage.googleapis.com/openimages/web/index.html) | ~5,000 | ~1,000 | More bird images including subspecies (via [FiftyOne](https://docs.voxel51.com/)) |

**What the notebook does:**
1. Downloads only bird-containing images (not the full 18 GB COCO dataset)
2. Exports Open Images annotations to COCO format and normalises all category IDs to `1` (bird)
3. Letterbox-resizes every image to 1024×1024 px while preserving aspect ratio (grey padding)
4. Adjusts all bounding box coordinates to match the resized images
5. Filters out tiny boxes (< 4 px) that would just be noise
6. Merges everything into train/val splits with unique image and annotation IDs
7. Validates the final dataset and generates preview images

**Output format:** Standard [COCO JSON](https://cocodataset.org/#format-data) with one class (`bird`, id=1).

> **Run this on Kaggle** with Internet enabled. No GPU needed. Takes ~60–90 minutes (mostly downloading). After it finishes, publish the output as a Kaggle Dataset named `airport-bird-detection`.

---

### Step 2 — Model Training (`rfdetr_finetune_bird_detection.ipynb`)

This notebook fine-tunes [RF-DETR](https://github.com/roboflow/rf-detr) on the dataset from Step 1.

**Why RF-DETR?**
- It's a transformer-based object detector that gets high accuracy without needing anchor boxes or NMS post-processing
- The Base model (~29M parameters) hits a good accuracy/speed sweet spot
- It supports ONNX export out of the box, which is what we need for CPU deployment

**Training configuration:**

| Setting | Value | Why |
|---------|-------|-----|
| Model | RF-DETR-Base | Best accuracy/speed tradeoff for this dataset size |
| Resolution | 560 px | Fits in T4 GPU memory, fast enough for edge deployment |
| Epochs | 100 | With early stopping (patience=20) |
| Batch size | 4 per GPU | Effective batch = 16 with 2 GPUs × 2 grad accumulation |
| Learning rate | 1e-4 (head), 1e-5 (backbone) | Lower LR for pretrained backbone to avoid catastrophic forgetting |

**What the notebook does:**
1. Installs RF-DETR and dependencies
2. Verifies the dataset paths and annotation counts
3. Sets up training callbacks (logging, best checkpoint saving, early stopping, live loss curves)
4. Fine-tunes RF-DETR-Base starting from COCO pretrained weights
5. Runs COCO-style evaluation on the val set (mAP, mAP50, AR by object size)
6. Draws predicted vs ground-truth boxes on sample images for visual sanity checking
7. Exports the best checkpoint to ONNX format and simplifies it with [onnxsim](https://github.com/daquexian/onnx-simplifier)
8. Benchmarks ONNX inference speed on CPU

**Key outputs:**
- `best_model.pth` — Best PyTorch checkpoint
- `bird_detector.onnx` — ONNX model for deployment
- `metrics.json` — All training curves and final evaluation results
- `inference_demo.jpg` — Visual predictions on validation images

> **Run this on Kaggle** with T4 GPU × 2 and Internet enabled. Takes ~3–5 hours.

---

### Step 3 — Web App (`app.py`)

A Streamlit app called **AeroWatch** that runs the trained ONNX model for real-time bird detection.

**Features:**
- **Image detection** — Upload a photo, see detected birds with bounding boxes and confidence scores
- **Video detection** — Upload a video, process it frame-by-frame with live risk updates
- **Risk assessment** — Classifies each frame as SAFE (0 birds), LOW RISK (1–5), or HIGH RISK (5+)
- **Adjustable settings** — Confidence threshold, input resolution, frame skip rate
- **Download results** — Save annotated images

**How the inference pipeline works:**
1. Image is letterbox-resized to 560×560 and normalised to [0, 1]
2. ONNX model outputs 300 candidate detections (boxes + class probabilities)
3. Low-confidence detections are filtered out
4. Box coordinates are mapped back to original image dimensions
5. Detections are drawn on the image with confidence labels

The model is loaded from HuggingFace Hub at startup, so you don't need to bundle the 110 MB ONNX file in your deployment.

---

## Getting Started

### Run the App Locally

```bash
# Clone the repo
git clone https://github.com/adeelumar17/airport-bird-detector.git
cd airport-bird-detector

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set your HuggingFace token (needed to download the model)
echo "HF_TOKEN=hf_your_token_here" > .env

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set `app.py` as the main file
4. In **Settings → Secrets**, add:
   ```
   HF_TOKEN = "hf_your_token_here"
   ```
5. Deploy

### Retrain the Model

If you want to retrain from scratch or tweak the training:

1. Open `airport_bird_dataset_prep.ipynb` on [Kaggle](https://www.kaggle.com/)
2. Run all cells → publish the output as a dataset
3. Open `rfdetr_finetune_bird_detection.ipynb` on Kaggle
4. Add your dataset as input, enable T4 × 2 GPU
5. Run all cells → download the ONNX model from the output
6. Upload the ONNX file to your HuggingFace repo

---

## Requirements

**For the app:**
```
streamlit >= 1.35.0
onnxruntime >= 1.18.0
opencv-python-headless >= 4.9.0
numpy >= 1.26.0
Pillow >= 10.0.0
huggingface_hub >= 1.5.0
```

**For training (handled by the notebooks):**
- PyTorch with CUDA
- [rfdetr](https://github.com/roboflow/rf-detr)
- pycocotools
- [FiftyOne](https://docs.voxel51.com/) (dataset download)

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | [RF-DETR-Base](https://github.com/roboflow/rf-detr) |
| Parameters | ~29M |
| Input | 560 × 560 px, float32, normalised to [0, 1] |
| Output | 300 candidate boxes (cx, cy, w, h) + class probabilities |
| Format | ONNX (opset 17) |
| File size | ~110 MB |
| Classes | 1 (bird) |
| Hosted on | [HuggingFace](https://huggingface.co/adeelumar17/airport-bird-detector) |

---

## References

- **RF-DETR** — [GitHub](https://github.com/roboflow/rf-detr) · [Paper](https://arxiv.org/abs/2501.00238) — The object detection model this project fine-tunes
- **COCO Dataset** — [Website](https://cocodataset.org/) · [Paper](https://arxiv.org/abs/1405.0312) — Source of bird training images
- **Open Images v7** — [Website](https://storage.googleapis.com/openimages/web/index.html) · [Paper](https://arxiv.org/abs/1811.00982) — Additional bird images
- **FiftyOne** — [Docs](https://docs.voxel51.com/) — Tool used to download and manage Open Images data
- **ONNX Runtime** — [GitHub](https://github.com/microsoft/onnxruntime) — Runs the exported model on CPU
- **Streamlit** — [Docs](https://docs.streamlit.io/) — Framework for the web app
- **COCO Evaluation** — [pycocotools](https://github.com/cocodataset/cocoapi) — Used for mAP/AR metrics
- **FAA Wildlife Strike Data** — [FAA](https://www.faa.gov/airports/airport_safety/wildlife) — Context on why bird detection at airports matters

---

## License

The training data comes from COCO 2017 and Open Images v7, both under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The code in this repository is provided as-is for educational purposes.
