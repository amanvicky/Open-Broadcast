# OpenBroadcast — Eye Gaze Correction for Low-End PCs

**A complete eye gaze correction application that works without a GPU.**

When you're on a video call and reading notes or looking at a second screen, OpenBroadcast corrects your eye gaze in real-time so it appears you're looking straight at the camera.

## Features

- **Real-time eye gaze correction** — makes eyes appear to look straight at camera
- **Works on low-end PCs** — no GPU required, runs on CPU only
- **Auto hardware detection** — detects your PC specs and configures optimal settings
- **Geometric gaze estimation** — runs in <1ms, no model needed
- **Optional neural model** — GazeNet-Lite for improved accuracy
- **Virtual camera output** — use with Zoom, Teams, OBS, etc.
- **Dark theme UI** — professional broadcast-style interface
- **Calibration** — calibrate to your specific eye position

## Performance Tiers

| Tier | Hardware | Mode | Resolution | FPS |
|------|----------|------|-----------|-----|
| ULTRA_LOW | i3 + 4GB, no GPU | Geometric only | 480p | 15+ |
| LOW | i5 + 6GB, no GPU | Geometric + smoothing | 480p | 20+ |
| MEDIUM | i5 + 8GB | Hybrid (geo + neural) | 720p | 25+ |
| HIGH | i7 + 16GB | Full quality | 1080p | 30+ |

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

### First Launch

On first launch, the app will:
1. Detect your hardware (CPU, RAM, GPU, camera)
2. Classify your PC into a performance tier
3. Show a wizard with recommended settings
4. Auto-configure optimal settings

### Basic Usage

1. **Camera starts automatically** — you'll see your webcam feed
2. **Correction is enabled by default** — eyes should appear centered
3. **Adjust strength** — use the slider in the control panel
4. **Toggle correction** — click ENABLED/DISABLED button
5. **Calibrate** — click "Calibrate Gaze" and look at camera for 2 seconds

### Controls

| Control | Description |
|---------|-------------|
| Correction Strength | 0-100% — how much to correct gaze |
| Show Face Landmarks | Display detected face landmarks |
| Performance Mode | Geometric (fastest) → Hybrid (best) |
| Calibrate Gaze | Calibrate to your eye position |
| Virtual Camera | Output to OBS/Zoom/Teams |

## Training the Neural Model (Optional)

The geometric mode works without any model. For improved accuracy:

```bash
# 1. Download MPIIFaceGaze dataset
#    Visit: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/
#    Register and download, then extract to data/raw/mpiigaze/

# 2. Setup dataset
python -m data.setup_dataset --output_dir data/processed

# 3. Train model (~5 hours on CPU)
python -m models.train --data_dir data/processed --epochs 100

# 4. Export to ONNX for fast inference
python -m models.export_onnx --weights models/weights/gaze_net_best.pth

# 5. Run with neural model
python main.py
```

## Building Windows .exe

```bash
# Install PyInstaller
pip install pyinstaller

# Build
pyinstaller OpenBroadcast.spec

# Result: dist/OpenBroadcast.exe
```

## Project Structure

```
openbroadcast/
├── main.py                     # Entry point
├── config.py                   # Settings
├── core/
│   ├── camera.py               # Camera capture
│   ├── face_detector.py        # MediaPipe face mesh + iris
│   ├── gaze_estimator.py       # Geometric gaze estimation
│   ├── eye_corrector.py        # Eye warp correction engine
│   ├── system_detector.py      # Hardware detection
│   └── virtual_camera.py       # Virtual camera output
├── models/
│   ├── gaze_net.py             # GazeNet-Lite CNN
│   ├── eye_preprocessor.py     # Eye ROI extraction
│   ├── train.py                # Training pipeline
│   └── export_onnx.py          # ONNX export + quantization
├── data/
│   ├── dataset.py              # Dataset loader
│   ├── augmentations.py        # Training augmentations
│   └── setup_dataset.py        # Dataset download/setup
├── ui/
│   ├── main_window.py          # Main window
│   ├── preview_widget.py       # Camera preview
│   ├── control_panel.py        # Settings sidebar
│   ├── first_run_wizard.py     # First launch wizard
│   └── styles.py               # Dark theme
├── utils/
│   ├── geometry.py             # Eye geometry math
│   ├── image_utils.py          # Blending utilities
│   ├── performance.py          # FPS tracking
│   └── calibration.py          # Gaze calibration
└── tests/
    └── test_core.py            # Component tests
```

## How It Works

### Pipeline

```
Camera → Face Detection → Eye Landmarks → Gaze Estimation → Eye Warp → Display
  ↓         ↓                ↓                ↓               ↓          ↓
cv2      MediaPipe         468+10          Geometric        cv2.remap  PyQt6
         Face Mesh         landmarks       (<1ms)          + blending
```

### Gaze Estimation

1. MediaPipe detects 468 face landmarks + 10 iris landmarks (5 per eye)
2. Iris center position is measured relative to eye corners
3. Offset from eye center = gaze direction
4. No neural network needed for basic mode

### Eye Correction

1. Calculate pixel displacement needed to center the iris
2. Create distance-weighted displacement map (only eye moves, skin stays)
3. Apply OpenCV remap (uses SIMD on CPU — very fast)
4. Blend corrected region back with feathered mask
5. Optional color correction in LAB color space

## Requirements

### Minimum (Geometric Mode)
- Python 3.10+
- OpenCV
- MediaPipe
- PyQt6
- NumPy
- psutil

### Full (Neural Model)
All of the above plus:
- PyTorch (training only)
- ONNX Runtime (inference)
- py-cpuinfo

### Optional
- pyvirtualcam (virtual camera output)
- WMI (Windows hardware detection)
- screeninfo (display detection)

## License

This project is open source. Use responsibly.

