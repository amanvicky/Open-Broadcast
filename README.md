# OpenBroadcast — Eye Gaze Correction for Low-End PCs

**Real-time eye gaze correction that works without a GPU.**

When you're on a video call and reading notes or looking at a second screen, OpenBroadcast corrects your eye gaze in real-time so it appears you're looking straight at the camera.

## Features

- **Real-time eye gaze correction** — makes eyes appear to look straight at camera
- **Works on low-end PCs** — no GPU required, runs on CPU at 30fps
- **Two correction modes** — Geometric (fastest) or Neural (best)
- **Face landmark overlay** — visualize detected eye positions
- **Gaze calibration** — calibrate to your specific eye position
- **Virtual camera output** — use with Zoom, Teams, OBS, etc.
- **Static image testing** — test correction on photos without a webcam
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
3. **Enable Compare mode** — checkbox shows original vs corrected side-by-side
4. **Adjust strength** — use the slider (0-100%)
5. **Adjust amplification** — multiplier for correction shift (1.0x-5.0x)
6. **Calibrate** — click "Calibrate Gaze", look at camera for 2 seconds, click Stop

### Controls

| Control | Description |
|---------|-------------|
| Correction Strength | 0-100% — how much to correct gaze |
| Correction Amplification | 1.0x-5.0x — multiplier for correction shift |
| Show Face Landmarks | Display detected eye landmarks overlay |
| Performance Mode | Geometric (Fastest) or Neural (Best) |
| Calibrate Gaze | Calibrate to your eye position |
| Virtual Camera | Output corrected video to OBS/Zoom/Teams |

### Test on Static Image

```bash
# Run correction on a photo (no webcam needed)
python main.py --test-image photo.jpg --output result.jpg

# Adjust strength and amplification
python main.py --test-image photo.jpg --strength 0.9 --amplification 5.0
```

## Correction Modes

### Geometric (Fastest)

Default mode. Uses iris transplant — extracts the iris and pastes it at the eye center with feathered blending. No model needed, runs at <2ms per frame.

### Neural (Best)

Optional mode. Uses a trained U-Net (~300K parameters) that generates corrected eye patches. Produces more natural results at small gaze offsets.

#### Training the Neural Model

```bash
# Step 1: Collect training data from webcam
# Move your head around for 2 minutes to capture variety
python train.py --collect

# Step 2: Train the model (~5 minutes on CPU)
python train.py --train --epochs 50

# Step 3: Run the app — select "Neural (Best)" in Performance Mode
python main.py
```

Or collect from a video file:

```bash
python train.py --collect --video path/to/video.mp4
python train.py --train --epochs 50
```

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
4. Head pose is NOT subtracted (unreliable with current landmarks)

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

## License

This project is open source. Use responsibly.
