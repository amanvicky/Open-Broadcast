# OpenBroadcast — Eye Gaze Correction for Low-End PCs

**Real-time eye gaze correction that works without a GPU.**

When you're on a video call and reading notes or looking at a second screen, OpenBroadcast corrects your eye gaze in real-time so it appears you're looking straight at the camera.

## Features

- **Real-time eye gaze correction** — makes eyes appear to look straight at camera
- **Works on low-end PCs** — no GPU required, runs on CPU at 30fps
- **Demo mode** — simulate 20° gaze offset to prove correction works
- **Iris segmentation** — learned mask for precise iris boundaries (55K params)
- **Iris position overlay** — red/green dots + shift arrow with pixel count
- **Correction quality score** — real-time "Shift: 12px" in status bar
- **Two correction modes** — Geometric (fastest) or Neural (best)
- **Interactive calibration wizard** — 8-point guided calibration with moving dot
- **Auto-calibration** — silently calibrates during first 3 seconds
- **Temporal consistency** — 70/30 frame blending to eliminate flicker
- **Virtual camera output** — use with Zoom, Teams, OBS, etc.
- **Recording** — save corrected video to file
- **Batch video processing** — apply correction to existing video files
- **Keyboard shortcuts** — Space, C, L, D, R, T, S, 1-5, Ctrl+A, Ctrl+B, Esc
- **Preset system** — save/load 5 configurations for different use cases
- **Online learning** — model continuously improves while app runs
- **Dark theme UI** — professional broadcast-style interface

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Run the automated setup script
setup.bat
```

This will:
1. Download Python embeddable if not installed
2. Install all dependencies
3. Download face detection model
4. Create required directories

### Option 2: Manual Setup

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
5. **Press D for Demo mode** — see dramatic correction without moving
6. **Check the status bar** — shows "Shift: XXpx" for correction quality

## Controls

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Space** | Toggle correction ON/OFF |
| **C** | Toggle compare mode |
| **L** | Toggle face landmarks |
| **D** | Toggle demo mode (simulate 20° offset) |
| **R** | Toggle recording |
| **T** | Start/stop training |
| **S** | Save current settings as preset 1 |
| **1-5** | Load preset 1-5 |
| **Ctrl+A** | About dialog |
| **Ctrl+B** | Batch process video file |
| **Esc** | Quit app |

### UI Controls

| Control | Description |
|---------|-------------|
| Correction Strength | 0-100% — how much to correct gaze |
| Correction Amplification | 1.0x-5.0x — multiplier for correction shift |
| Compare: Original vs Corrected | Side-by-side split view |
| Show Face Landmarks | Display detected eye landmarks overlay |
| Demo Mode | Simulate 20° gaze offset to prove correction |
| Process Video File | Apply correction to existing video (Ctrl+B) |
| Performance Mode | Geometric (Fastest) or Neural (Best) |
| Calibrate Gaze | Calibrate to your eye position |
| Interactive Calibration | 8-point guided wizard with moving dot |
| Virtual Camera | Output corrected video to OBS/Zoom/Teams |
| Record Corrected Video | Save corrected output to file (R) |
| Train Model | Collect data + train neural model in-app |
| About OpenBroadcast | Version info and credits (Ctrl+A) |

### Presets

| Preset | Strength | Amplification | Use Case |
|--------|----------|---------------|----------|
| 1. Zoom Call | 85% | 3.0x | Video meetings |
| 2. Streaming | 100% | 5.0x | Live streaming |
| 3. Recording | 90% | 4.0x | Video recording |
| 4. Subtle | 50% | 2.0x | Minimal correction |
| 5. Maximum | 100% | 5.0x | Maximum correction |

Press **1-5** to load, **S** to save current settings to slot 1.

## Demo Mode

The fastest way to see if correction works:

1. Run `python main.py`
2. Look at the camera (correction is subtle at small angles)
3. Press **D** to enable Demo Mode
4. The app simulates a 20° gaze offset
5. You'll see dramatic correction in the preview
6. Status bar shows "DEMO" and "Shift: XXpx"
7. Press **D** again to disable

This proves the correction works without requiring you to physically look away.

## Training the Iris Segmenter

The iris segmenter learns exact iris boundaries for better mask quality:

```bash
# Collect training data from webcam (2 minutes)
python train_segmenter.py --collect

# Train the model (~5 minutes on CPU)
python train_segmenter.py --train

# Or do both at once
python train_segmenter.py --all
```

The trained model (55K params, 248KB) automatically loads when the app starts.

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

### In-App Training

1. Run `python main.py`
2. Click **"Train Model"** in Neural Training group
3. Follow the guided directions (~20 seconds)
4. Training runs automatically (~30 seconds)
5. "Neural (Best)" appears in Performance Mode dropdown

### Large-Scale Training (best quality)

```bash
# Step 1: Download 300K+ face images (~5GB)
python -m data.download_datasets --all

# Step 2: Generate 100K+ training pairs (~10GB)
python -m data.large_scale_generator --input data/raw --target-count 100000

# Step 3: Train the model (~30 minutes on CPU)
python train.py --train --data data/large_pairs.npz --epochs 100
```

## Batch Video Processing

Apply correction to existing video files:

1. Press **Ctrl+B** or click "Process Video File"
2. Select input video (MP4, AVI, MKV, MOV)
3. Choose output location
4. Processing runs in background with progress display
5. Output is saved as AVI with correction applied

## Virtual Camera Output

Use corrected video in Zoom, Teams, OBS, or any webcam app:

1. Click "Virtual Camera: OFF" to enable
2. Open your video app (Zoom, Teams, etc.)
3. Select "OBS Virtual Camera" as your webcam source
4. Corrected video appears in the app

Requires: `pip install pyvirtualcam`

## Project Structure

```
openbroadcast/
├── main.py                  # Entry point
├── config.py                # Settings persistence
├── train.py                 # Neural model training
├── train_segmenter.py       # Iris segmenter training
├── setup.bat                # Automated Windows setup
├── run.bat                  # App launcher
├── requirements.txt         # Dependencies
├── core/
│   ├── camera.py            # Webcam capture (QThread)
│   ├── face_detector.py     # MediaPipe face mesh + iris
│   ├── gaze_estimator.py    # Geometric gaze estimation
│   ├── eye_corrector.py     # Iris transplant correction
│   ├── gaze_model.py        # Tiny U-Net for neural correction
│   ├── neural_corrector.py  # Neural model inference
│   └── iris_segmenter.py    # Iris segmentation model
├── ui/
│   ├── main_window.py       # Main window + pipeline
│   ├── preview_widget.py    # Camera preview + overlays
│   ├── control_panel.py     # Settings sidebar
│   └── styles.py            # Dark theme
├── utils/
│   └── performance.py       # FPS counter
├── data/
│   ├── download_datasets.py # Download face datasets
│   └── large_scale_generator.py # Generate training pairs
├── installer/
│   └── OpenBroadcast.iss    # Inno Setup installer script
└── models/
    ├── iris_segmenter.pth   # Trained iris segmenter
    └── gaze_correction.pth  # Trained neural model
```

## How It Works

### Pipeline

```
Camera → FaceDetector → GazeEstimator → EyeCorrector → Temporal Blend → Display
  ↓         ↓              ↓              ↓                ↓            ↓
cv2      MediaPipe      Geometric      Iris transplant   cv2        PyQt6
         478 landmarks  (<1ms)         + Segmentation   addWeighted
```

### Gaze Estimation

1. MediaPipe detects 468 face landmarks + 10 iris landmarks
2. Iris center position measured relative to eye corners
3. Offset from eye center = gaze direction
4. Calibration offset applied to iris pixel position
5. Temporal smoothing reduces landmark jitter

### Eye Correction

1. Calculate pixel displacement to move iris to eye center
2. Extract iris region (segmented mask or circular fallback)
3. Paste iris at new position with feathered blend
4. Clamp position to stay within eye socket (max 40% from center)
5. EMA smoothing prevents jitter across frames

### Iris Position Overlay

- **Red dot**: current iris position
- **Green dot**: target position (where iris will be moved)
- **Yellow arrow**: shift vector with pixel count (e.g., "30px")

### Temporal Consistency

Lightweight 70/30 blend with previous frame using `cv2.addWeighted`. Eliminates frame-to-frame flicker without noticeable latency.

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
| Iris segmenter | 55K params, 2ms | — |

## Troubleshooting

### Camera not working
- Close other apps using the camera (browser, Zoom, etc.)
- Select a different camera from the dropdown
- Click "Restart Camera" to retry

### Correction not visible
- Enable **Compare: Original vs Corrected** to see side-by-side
- Press **D** for Demo mode to see dramatic correction
- Check the **Shift: XXpx** in status bar
- Increase **Correction Strength** to 100%
- Run **Interactive Calibration** to calibrate to your eyes

### Low FPS
- Close other applications using camera
- Use **Geometric (Fastest)** mode
- Reduce camera resolution in settings

### Neural model not available
- Click **"Train Model"** in the app, or
- Run `python train.py --collect` then `python train.py --train`

### Virtual camera not working
- Install pyvirtualcam: `pip install pyvirtualcam`
- Install OBS Studio (provides the virtual camera backend)

## Version

Current version: **1.1.0**

Press **Ctrl+A** or click "About OpenBroadcast" for version info.

## License

This project is open source. Use responsibly.
