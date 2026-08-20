@echo off
echo ========================================
echo   OpenBroadcast - Neural Model Training
echo ========================================
echo.
echo This script will:
echo   1. Check for MPIIGaze dataset
echo   2. Preprocess eye crops
echo   3. Train GazeNet-Lite (~2 hours on CPU)
echo   4. Export to ONNX for inference
echo.

:: Find Python
set PYTHON_CMD=
where python >nul 2>nul && set PYTHON_CMD=python
if "%PYTHON_CMD%"=="" where py >nul 2>nul && set PYTHON_CMD=py
if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Using: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: Check if dataset exists
if not exist "data\raw\mpiigaze\data" (
    echo ============================================
    echo   MPIIGaze Dataset Not Found
    echo ============================================
    echo.
    echo Please download the MPIIFaceGaze dataset:
    echo.
    echo   1. Go to: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/the-mpiigaze-dataset-15000-images-under-real-world-lighting-conditions/
    echo   2. Register for free (takes 1 minute)
    echo   3. Download the dataset zip file
    echo   4. Extract to: data\raw\mpiigaze\
    echo      (should contain 'data\' and 'annotation\ folders)
    echo.
    echo Press any key after extracting, or close to cancel.
    pause
)

:: Check again after user action
if not exist "data\raw\mpiigaze\data" (
    echo [ERROR] Dataset still not found at data\raw\mpiigaze\
    echo Please extract the dataset and try again.
    pause
    exit /b 1
)

echo.
echo [Step 1/4] Preprocessing dataset...
%PYTHON_CMD% -m data.setup_dataset --output_dir data\processed --mpiigaze_dir data\raw\mpiigaze
if %errorlevel% neq 0 (
    echo [ERROR] Preprocessing failed!
    pause
    exit /b 1
)

echo.
echo [Step 2/4] Training GazeNet-Lite...
echo This will take ~2 hours on CPU. Training progress will be shown.
%PYTHON_CMD% -m models.train --data_dir data\processed --output_dir models\weights --epochs 100 --batch_size 256
if %errorlevel% neq 0 (
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

echo.
echo [Step 3/4] Exporting to ONNX...
%PYTHON_CMD% -c "from models.gaze_net import GazeNetLite, export_to_onnx; import torch; m = GazeNetLite(); m.load_state_dict(torch.load('models/weights/gaze_net_best.pth', map_location='cpu')); export_to_onnx(m, 'models/weights/gaze_net.onnx', quantize=True)"
if %errorlevel% neq 0 (
    echo [ERROR] ONNX export failed!
    pause
    exit /b 1
)

echo.
echo [Step 4/4] Verifying model...
%PYTHON_CMD% -c "import onnxruntime as ort; s = ort.InferenceSession('models/weights/gaze_net.onnx'); print('ONNX model loaded successfully!'); print(f'Input: {s.get_inputs()[0].shape}'); print(f'Output: {s.get_outputs()[0].shape}')"
if %errorlevel% neq 0 (
    echo [WARNING] ONNX verification failed, but model may still work.
)

echo.
echo ========================================
echo   Training Complete!
echo ========================================
echo.
echo   Model saved to: models\weights\gaze_net.onnx
echo   Quantized: models\weights\gaze_net_quantized.onnx
echo.
echo   The app will now use hybrid mode (geometric + neural).
echo   Restart OpenBroadcast to use the trained model.
echo.
pause
