# Fine crack measurement: running the pipeline on another computer

This guide gets the crack-extraction pipeline running on a second machine and shows the commands for each mode.
`run_pipeline.py` does one image; `run_folder.py` does a whole folder, or every Nth frame of it.

## What the pipeline does

For each image (same steps as `crack.py`, but with no windows and no gridline removal):

1. Upscale the image (x2 by default; RealESRGAN through IOPaint with `--iopaint`, plain resizing without it).
2. Run UNet (`models/crack_segmentation`) and DeepCrack (`models/DeepCrack`).
3. Merge the two predictions and threshold them (value 23) into a black-and-white crack mask.
4. Optionally keep only the cracks inside the dataset's reference mask (`references/<dataset>_ref.png`).

## 1. Set up the other computer

Needs Python 3.10 or 3.11 and git. A GPU is optional but makes it much faster (about 10 minutes per image on CPU at x2).

```bash
git clone https://github.com/aanishwaseem/fine-cracks-measurement.git
cd fine-cracks-measurement
git checkout claude/document-review-retrieval-ulb4y4

python -m venv venv
# Windows:  venv\Scripts\activate        Linux/macOS:  source venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch first, then the rest:

```bash
# GPU (NVIDIA, CUDA 12.8), as in "steps to enable gpu":
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# or CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install iopaint gdown        # iopaint is only needed for --iopaint
```

### Model files (not in the git repo)

The two model folders in the repo are empty placeholders. Download the folder from Drive and copy its contents in:

https://drive.google.com/drive/folders/1OpXdL55au31fcUAiGHPOax8ChaUBdegI

You need exactly these two files (keep the folder structure; the rest of the model code ships in the same folders):

| File | Goes to |
|---|---|
| `DeepCrack/codes/checkpoints/DeepCrack_CT260_FT1.pth` | `models/DeepCrack/codes/checkpoints/` |
| `crack_segmentation/models/model_unet_vgg16_best.pt` | `models/crack_segmentation/models/` |

Copy the whole `DeepCrack` and `crack_segmentation` folders over `models/DeepCrack` and `models/crack_segmentation`, so the code files come with the weights.
You do not need the `Unified-Line-Segment-Detection` folder; it is only used for gridline removal, which this pipeline skips.

The first UNet run also downloads the ImageNet VGG16 weights (about 550 MB) into the torch cache. This needs internet once.

### Check that it works

```bash
python run_pipeline.py --image "references/sample_B_scaled2.jpg" --no-reference --out outputs/test
```

You should get `outputs/test/sample_B_scaled2_binary_no_reference.png`.

## 2. IOPaint server (only for `--iopaint`)

`--iopaint` uses RealESRGAN inside IOPaint for the upscale, which gives sharper cracks than plain resizing. Start the server in its own terminal and leave it running:

```bash
iopaint install-plugins-packages     # first time only
# GPU:
iopaint start --model lama --device cuda --port 8000 --realesrgan-model RealESRGAN_x4plus --realesrgan-device cuda --enable-realesrgan
# CPU:
iopaint start --model lama --device cpu  --port 8000 --realesrgan-model RealESRGAN_x4plus --realesrgan-device cpu  --enable-realesrgan
```

Wait until it prints `Uvicorn running on http://127.0.0.1:8000` before running the pipeline.

## 3. Commands

All commands run from the repo folder with the venv active. Outputs go to `outputs/` unless you pass `--out`.

### One image

```bash
# Full pipeline, with the reference mask for that dataset (found automatically from the folder name):
python run_pipeline.py --iopaint --image "<path to image>"

# Same, but choose the reference yourself:
python run_pipeline.py --iopaint --reference references/B-868-7_14_Cropped_ref.png --image "<path to image>"

# No reference intersection (keeps gridlines and everything else the models find):
python run_pipeline.py --iopaint --no-reference --image "<path to image>"

# Upscale x4 instead of x2 (much slower):
python run_pipeline.py --iopaint --upscale 4 --image "<path to image>"

# One model only:
python run_pipeline.py --deepcrack-only --image "<path to image>"
python run_pipeline.py --unet-only --image "<path to image>"
python run_pipeline.py --unet-only --upscale 4 --iopaint --image "<path to image>"

# Smaller DeepCrack tiles (finer cracks, more noise); default is 400:
python run_pipeline.py --iopaint --deepcrack-tile 200 --image "<path to image>"

# Resize image and reference to 448x224 first:
python run_pipeline.py --iopaint --size 448x224 --image "<path to image>"
```

Output files, named after the image (`_iopaint` / `_x4` etc. are added to match the options):

| File | Meaning |
|---|---|
| `*_binary.png` | final crack mask (white = crack), with the reference applied |
| `*_binary_no_reference.png` | same, before the reference step (the only mask when `--no-reference` is used) |
| `*_overlay.png` | cracks drawn in red on the upscaled image |
| `*_unet_raw.png`, `*_deepcrack_raw.png` | model outputs before thresholding |

### A whole folder, or every Nth frame

```bash
# Every 6th frame (frames 0, 6, 12, ...), no reference:
python run_folder.py "datasets/B-1330-5.56_Cropped" --every 6 --iopaint --no-reference --out outputs/B1330

# Every frame, with a reference:
python run_folder.py "datasets/C-1700-4.60_Cropped" --iopaint --reference references/C-1700-4_60_Cropped_ref.png --out outputs/C
```

- Frames are sorted by the number in the file name, e.g. `...X(12).jpg`. `--every N` takes every Nth frame; `--offset K` starts later.
- Frames that already have an output are skipped, so you can stop and restart the same command.
- Run one copy at a time per checkout; the models share tile folders under `models/`.
- Extra options such as `--deepcrack-tile 200` are passed through to `run_pipeline.py`.

## 4. Datasets and references

- Cropped frames, one folder per dataset: https://drive.google.com/drive/folders/1QhgXs6dHmk0DsKheCFufxRph28Rub5Lx
  (the Drive page lists only the first 50 files per folder, so use the Drive "Download" button for whole folders).
- A reference is a black-and-white mask of the real cracks, 2x the frame size, stored as `references/<folder name>_ref.png`
  (for example `B-868-7_14_Cropped_ref.png` for `B-868-7.14_Cropped`; characters other than letters, digits, `-` and `_` become `_`).
- References that fit their dataset: B-868-7.14, C-1700-4.60 and D-1700-4.60 (`D-1700-4_60_Cropped_ref.png`).
  Check these before relying on them:
  `D-1700-4-60_Cropped_ref.png` shows the C-dataset cracks, not D-1700, and `D-868-7_14_Cropped_ref.png` is noisy (it includes gridline pieces and a marker scribble).
- A dataset with no reference (for example B-1330) can only be run with `--no-reference`.
- The reference is resized automatically to the upscaled image, so use `--no-reference` if its aspect ratio does not match the frames.

## 5. Notes and troubleshooting

- **Speed:** about 10 minutes per frame at x2 on a CPU, 25-30 minutes at x4. A GPU cuts this a lot.
- **`IOPaint` connection refused:** the server from step 2 is not running; start it, or drop `--iopaint`.
- **Out of memory at x4:** use x2, or process fewer frames at a time.
- **Windows paths with spaces or brackets:** keep the quotes around file and folder names, e.g. `"cropped_C-1700-4.600.2mmX (81).jpg"`.
- **Do not run two pipelines at once** in the same checkout; they overwrite each other's tiles.
- The original GUI (`crack.py`) still works as before: see `Running the code`.
- How the reference images are labelled with the five crack classes: see `LABELING.md`.
