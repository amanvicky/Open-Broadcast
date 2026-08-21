# OpenBroadcast — Eye Gaze Correction for Low-End PCs

**Real-time eye gaze correction that works without a GPU.**

When you're on a video call and reading notes or looking at a second screen, OpenBroadcast corrects your eye gaze in real-time so it appears you're looking straight at the camera.

## Features

- **Real-time eye gaze correction** — makes eyes appear to look straight at camera
- **Works on low-end PCs** — no GPU required, runs on CPU at 30fps
- **Two correction modes** — Geometric (fastest) or Neural (best)
- **Iris position overlay** — red/green dots + shift arrow show exactly what's happening
- **Face landmark overlay** — visualize detected eye positions
- **Interactive calibration wizard** — 8-point guided calibration with moving dot
- **Auto-calibration** — silently calibrates during first 3 seconds of use
- **Keyboard shortcuts** — Space, C, L, T, S, 1-5, Esc for power users
- **Preset system** — save/load 5 different configurations for different use cases
- **Online learning** — model continuously improves while app runs
- **Virtual camera output** — use with Zoom, Teams, OBS, etc.
- **Static image testing** — test correction on photos without a webcam
- **Large-scale data pipeline** — download 300K+ face images for training
- **Dark theme UI** — professional broadcast-style interface

## Quick Start

### Install

```bash
# Clone or download this project
cd openbroadcast

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Basic Usage

1. **Camera starts automatically** — you'll see your webcam feed
2. **Correction is enabled by default** — eyes should appear centered
3. **Auto-calibration** — waits 3 seconds, then calibrates silently
4. **Enable Compare mode** — checkbox shows original vs corrected side-by-side
5. **Adjust strength** — use the slider (0-100%)
6. **Adjust amplification** — multiplier for correction shift (1.0x-5.0x)

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Space** | Toggle correction ON/OFF |
| **C** | Toggle compare mode |
| **L** | Toggle face landmarks |
| **T** | Start/stop training |
| **S** | Save current settings as preset 1 |
| **1-5** | Load preset 1-5 |
| **Esc** | Quit app |

### Controls

| Control | Description |
|---------|-------------|
| Correction Strength | 0-100% — how much to correct gaze |
| Correction Amplification | 1.0x-5.0x — multiplier for correction shift |
| Compare: Original vs Corrected | Side-by-side split view |
| Show Face Landmarks | Display detected eye landmarks overlay |
| Performance Mode | Geometric (Fastest) or Neural (Best) |
| Calibrate Gaze | Calibrate to your eye position |
| Interactive Calibration | 8-point guided wizard with moving dot |
| Virtual Camera | Output corrected video to OBS/Zoom/Teams |
| Train Model | Collect data + train neural model in-app |

### Presets

| Preset | Strength | Amplification | Use Case |
|--------|----------|---------------|----------|
| 1. Zoom Call | 85% | 3.0x | Video meetings |
| 2. Streaming | 100% | 5.0x | Live streaming |
| 3. Recording | 90% | 4.0x | Video recording |
| 4. Subtle | 50% | 2.0x | Minimal correction |
| 5. Maximum | 100% | 5.0x | Maximum correction |

Press **1-5** to load, **S** to save current settings to slot 1.

### Test on Static Image

```bash
# Run correction on a photo (no webcam needed)
python main.py --test-image photo.jpg --output result.jpg

# Adjust strength and amplification
python main.py --test-image photo.jpg --strength 0.9 --amplification 5.0
```

## Training the Neural Model

### Quick Training (from webcam)

```bash
# Step 1: Collect training data (2 minutes)
python train.py --collect

# Step 2: Train the model (~5 minutes on CPU)
python train.py --train --epochs 50

# Step 3: Run the app — select "Neural (Best)" in Performance Mode
python main.py
```

### Guided Training (better data quality)

```bash
# Collects data in 8 directions for balanced training
python train.py --guided

# Train with more epochs for better results
python train.py --train --epochs 100
```

### Large-Scale Training (best quality)

```bash
# Step 1: Download 300K+ face images (~5GB)
python -m data.download_datasets --all

# Step 2: Generate 100K+ training pairs (~10GB)
python -m data.large_scale_generator --input data/raw --target-count 100000

# Step 3: Train the model (~30 minutes on CPU)
python train.py --train --data data/large_pairs.npz --epochs 100

# Step 4: Run the app with neural correction
python main.py
```

### In-App Training

1. Run `python main.py`
2. Click **"Train Model"** in Neural Training group
3. Follow the guided directions (~20 seconds)
4. Training runs automatically (~30 seconds)
5. "Neural (Best)" appears in Performance Mode dropdown

## Online Learning

The app continuously improves while running:

- Collects 1 training pair per second in background
- Auto fine-tunes model every 500 pairs (5 epochs)
- Keeps max 2000 pairs buffer (most recent)
- Model improves continuously while app runs
- Reloads neural model automatically after fine-tune

## Project Structure

```
openbroadcast/
├── main.py                  # Entry point (--test-image support)
├── config.py                # Settings persistence
├── train.py                 # Neural model training pipeline
├── requirements.txt         # Dependencies
├── core/
│   ├── camera.py            # Webcam capture (QThread)
│   ├── face_detector.py     # MediaPipe face mesh + iris (478 landmarks)
│   ├── gaze_estimator.py    # Geometric gaze estimation
│   ├── eye_corrector.py     # Iris transplant correction
│   ├── gaze_model.py        # Tiny U-Net for neural correction
│   └── neural_corrector.py  # Neural model inference wrapper
├── ui/
│   ├── main_window.py       # Main window + pipeline orchestration
│   ├── preview_widget.py    # Camera preview + overlays
│   ├── control_panel.py     # Settings sidebar
│   └── styles.py            # Dark theme
├── utils/
│   └── performance.py       # FPS counter
├── data/
│   ├── download_datasets.py # Download academic face datasets
│   └── large_scale_generator.py # Generate training pairs
└── models/
    └── gaze_correction.pth  # Trained neural model (after training)
```

## How It Works

### Pipeline

```
Camera → FaceDetector → GazeEstimator → EyeCorrector → Display
  ↓         ↓              ↓              ↓             ↓
cv2      MediaPipe      Geometric      Iris transplant  PyQt6
         478 landmarks  (<1ms)         or Neural U-Net
```

### Gaze Estimation

1. MediaPipe detects 468 face landmarks + 10 iris landmarks (5 per eye)
2. Iris center position is measured relative to eye corners
3. Offset from eye center = gaze direction
4. Temporal smoothing reduces landmark jitter across frames

### Eye Correction (Geometric)

1. Calculate pixel displacement to move iris to eye center
2. Extract iris region with feathered circular mask
3. Paste iris at new position with alpha blending
4. EMA smoothing prevents jitter across frames

### Eye Correction (Neural)

1. Extract 64×64 eye patch centered on eye socket
2. Feed patch + offset vector through U-Net
3. Model predicts corrected eye patch
4. Blend result back into frame with feathered mask

### Iris Position Overlay

- **Red dot**: current iris position
- **Green dot**: target position (where iris will be moved)
- **Yellow arrow**: shift vector with pixel count (e.g., "30px")

Shows on the preview when correction is active, making the correction visible even at small offsets.

## Requirements

### Minimum (Geometric Mode)
- Python 3.10+
- OpenCV
- MediaPipe
- PyQt6
- NumPy

### Full (Neural Mode)
All of the above plus:
- PyTorch (training + inference)

### Optional
- pyvirtualcam (virtual camera output)

## Performance

| Metric | Geometric | Neural |
|--------|-----------|--------|
| FPS (i5 CPU) | 30 fps | 25 fps |
| Latency | <2ms | ~5ms |
| Model size | 0 MB | 1.2 MB |
| RAM usage | ~50 MB | ~200 MB |

## Troubleshooting

### "Looking away" when looking at camera
- Run **Interactive Calibration** to calibrate to your eye position
- Increase **Correction Amplification** to 4.0x-5.0x

### Correction not visible
- Enable **Compare: Original vs Corrected** to see side-by-side
- Check the **iris overlay** (red/green dots) to see shift amount
- Increase **Correction Strength** to 100%

### Low FPS
- Close other applications using camera
- Use **Geometric (Fastest)** mode
- Reduce camera resolution in settings

### Neural model not available
- Click **"Train Model"** in the app, or
- Run `python train.py --collect` then `python train.py --train`

## License

This project is open source. Use responsibly.
