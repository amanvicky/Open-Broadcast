@echo off
title OpenBroadcast - Automated Setup
cd /d "%~dp0"

echo ========================================
echo   OpenBroadcast - Automated Setup
echo ========================================
echo.

:: Step 1: Check for Python
echo [1/4] Checking for Python...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   Found: 
    python --version
    set PYTHON=python
    goto :install_deps
)

:: Try py launcher
py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   Found via py launcher:
    py --version
    set PYTHON=py
    goto :install_deps
)

:: Try embedded Python
if exist "python\python.exe" (
    echo   Found embedded Python
    set PYTHON=python\python.exe
    goto :install_deps
)

:: Download Python embeddable
echo   Python not found. Downloading embedded Python...
echo.

:: Create python directory
if not exist "python" mkdir python

:: Download Python 3.12 embeddable
echo   Downloading Python 3.12...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.4/python-3.12.4-embed-amd64.zip' -OutFile 'python_embed.zip'}"

if not exist "python_embed.zip" (
    echo   ERROR: Failed to download Python.
    echo   Please install Python 3.10+ manually from https://python.org
    pause
    exit /b 1
)

echo   Extracting...
powershell -Command "Expand-Archive -Path 'python_embed.zip' -DestinationPath 'python' -Force"
del python_embed.zip

:: Enable pip in embeddable Python
echo   Enabling pip...
powershell -Command "(Get-Content 'python\python312._pth') -replace '#import site', 'import site' | Set-Content 'python\python312._pth'"

:: Download get-pip.py
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'python\get-pip.py'}"

:: Install pip
python\python.exe python\get-pip.py --quiet
del python\get-pip.py

set PYTHON=python\python.exe

:install_deps
echo.
echo [2/4] Installing dependencies...
%PYTHON% -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo   WARNING: Some dependencies failed to install.
    echo   The app may still work with reduced features.
)

:: Step 3: Download MediaPipe model
echo.
echo [3/4] Downloading face detection model...
if not exist "models\weights" mkdir models\weights
if not exist "models\weights\face_landmarker.task" (
    echo   Downloading face_landmarker.task...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task' -OutFile 'models\weights\face_landmarker.task'}"
) else (
    echo   Model already exists.
)

:: Step 4: Create directories
echo.
echo [4/4] Creating directories...
if not exist "models" mkdir models
if not exist "recordings" mkdir recordings
if not exist "data\raw" mkdir data\raw
if not exist "data\pairs" mkdir data\pairs

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo   Run the app: run.bat
echo   Or: %PYTHON% main.py
echo.
pause
