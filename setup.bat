@echo off
title OpenBroadcast Setup
color 0B
setlocal enabledelayedexpansion

echo.
echo  ========================================
echo   OpenBroadcast - Automated Setup
echo  ========================================
echo.

:: ============================================================
:: STEP 1: Check for Python
:: ============================================================
echo [1/6] Checking for Python installation...

set PYTHON_FOUND=0
set PIP_CMD=

:: Method 1: Check common install locations first
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Python312\python.exe"
    "C:\Python313\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    if exist %%P (
        set PYTHON_FOUND=1
        set PIP_CMD=%%P -m pip
        for /f "tokens=*" %%i in ('%%P -c "import sys; print(sys.version)"') do set PY_VERSION=%%i
        echo   Found Python at: %%P
        echo   Version: !PY_VERSION!
        goto :check_pip
    )
)

:: Method 2: py launcher
py -c "import sys; print('ok')" >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_FOUND=1
    set PIP_CMD=py -m pip
    for /f "tokens=*" %%i in ('py -c "import sys; print(sys.version)"') do set PY_VERSION=%%i
    echo   Found Python via py launcher: !PY_VERSION!
    goto :check_pip
)

:: Method 3: python command
python -c "import sys; print('ok')" >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_FOUND=1
    set PIP_CMD=python -m pip
    for /f "tokens=*" %%i in ('python -c "import sys; print(sys.version)"') do set PY_VERSION=%%i
    echo   Found Python: !PY_VERSION!
    goto :check_pip
)

:: Python NOT found - need to install it
echo.
echo   [!] Python is NOT installed on this system.
echo.
echo   Will now download and install Python 3.12 automatically.
echo   This requires an internet connection.
echo.
echo   [!] If a UAC prompt appears, please click "Yes"
echo.
goto :install_python


:: ============================================================
:: STEP 2: Check pip
:: ============================================================
:check_pip
echo.
echo [2/6] Checking for pip...

%PIP_CMD% --version >nul 2>nul
if %errorlevel% neq 0 (
    echo   pip not found, installing...
    %PIP_CMD% ensurepip --upgrade >nul 2>nul
    if %errorlevel% neq 0 (
        echo   [ERROR] Could not install pip.
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%i in ('%PIP_CMD% --version 2^>nul') do set PIP_VER=%%i
echo   !PIP_VER!


:: ============================================================
:: STEP 3: Upgrade pip
:: ============================================================
echo.
echo [3/6] Upgrading pip...
%PIP_CMD% install --upgrade pip --quiet 2>nul
if %errorlevel% equ 0 (
    echo   pip upgraded.
) else (
    echo   [WARNING] pip upgrade skipped, continuing...
)


:: ============================================================
:: STEP 4: Install dependencies
:: ============================================================
echo.
echo [4/6] Installing dependencies...
echo   This may take 2-5 minutes on first install...
echo.

%PIP_CMD% install opencv-python mediapipe numpy PyQt6 psutil py-cpuinfo pyvirtualcam onnxruntime --quiet
if %errorlevel% neq 0 (
    echo.
    echo   Some packages failed together. Installing one by one...
    echo.
    for %%P in (opencv-python mediapipe numpy PyQt6 psutil py-cpuinfo pyvirtualcam onnxruntime) do (
        echo   Installing %%P...
        %PIP_CMD% install %%P --quiet 2>nul
        if !errorlevel! equ 0 (
            echo     [OK] %%P
        ) else (
            echo     [FAIL] %%P
        )
    )
) else (
    echo   All packages installed.
)

:: Optional packages
echo.
echo   Installing optional packages...
%PIP_CMD% install WMI screeninfo --quiet 2>nul


:: ============================================================
:: STEP 5: Verify
:: ============================================================
echo.
echo [5/6] Verifying installation...

set ALL_OK=1

python -c "import cv2" 2>nul
if %errorlevel% neq 0 (echo   [FAIL] OpenCV & set ALL_OK=0) else (echo   [OK] OpenCV)

python -c "import mediapipe" 2>nul
if %errorlevel% neq 0 (echo   [FAIL] MediaPipe & set ALL_OK=0) else (echo   [OK] MediaPipe)

python -c "import numpy" 2>nul
if %errorlevel% neq 0 (echo   [FAIL] NumPy & set ALL_OK=0) else (echo   [OK] NumPy)

python -c "import PyQt6" 2>nul
if %errorlevel% neq 0 (echo   [FAIL] PyQt6 & set ALL_OK=0) else (echo   [OK] PyQt6)

python -c "import psutil" 2>nul
if %errorlevel% neq 0 (echo   [FAIL] psutil & set ALL_OK=0) else (echo   [OK] psutil)


:: ============================================================
:: STEP 6: Done
:: ============================================================
echo.
echo [6/6] Setup complete!
echo.
if %ALL_OK% equ 1 (
    echo  ========================================
    echo   Setup Successful!
    echo  ========================================
    echo.
    echo   To run OpenBroadcast:
    echo     Double-click "run.bat"
    echo.
    echo   Or in this terminal, type:
    echo     python main.py
    echo.
) else (
    echo  ========================================
    echo   Setup completed with warnings.
    echo  ========================================
    echo.
    echo   Some packages may need manual install:
    echo     pip install opencv-python mediapipe numpy PyQt6 psutil
    echo.
)

pause
exit /b 0


:: ============================================================
:: AUTO-INSTALL PYTHON
:: ============================================================
:install_python

:: Create temp folder
if not exist "%TEMP%\ob_setup" mkdir "%TEMP%\ob_setup"

:: Download Python installer
set PYTHON_URL=https://www.python.org/ftp/python/3.12.5/python-3.12.5-amd64.exe
set PYTHON_EXE=%TEMP%\ob_setup\python-installer.exe

echo   Downloading Python 3.12.5...
echo   URL: %PYTHON_URL%
echo.

:: Try certutil (built into Windows)
certutil -urlcache -split -f "%PYTHON_URL%" "%PYTHON_EXE%" >nul 2>nul

:: Check if download succeeded
if not exist "%PYTHON_EXE%" (
    echo.
    echo   [ERROR] Download failed!
    echo.
    echo   Please install Python manually:
    echo.
    echo   1. Open your browser and go to:
    echo      https://www.python.org/downloads/
    echo.
    echo   2. Click "Download Python 3.12.x"
    echo.
    echo   3. Run the installer
    echo      IMPORTANT: Check "Add python.exe to PATH" at the bottom!
    echo      Then click "Install Now"
    echo.
    echo   4. After install, CLOSE this window and open a NEW terminal
    echo.
    echo   5. Run: setup.bat
    echo.
    pause
    exit /b 1
)

echo   Download complete! Installing...
echo.
echo   [!] If a UAC prompt appears, click "Yes"
echo.

:: Install Python silently with PATH addition
"%PYTHON_EXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0 DefaultJustForMeTargetDir="%LOCALAPPDATA%\Programs\Python\Python312"

:: Wait for installer to finish
echo   Waiting for installer...
timeout /t 10 /nobreak >nul

:: Refresh PATH
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"

:: Verify Python is now available
set PYTHON_FOUND=0

:: Check common locations again
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "C:\Python312\python.exe"
) do (
    if exist %%P (
        set PYTHON_FOUND=1
        set PIP_CMD=%%P -m pip
        for /f "tokens=*" %%i in ('%%P -c "import sys; print(sys.version)"') do set PY_VERSION=%%i
        echo   Python installed: !PY_VERSION!
        goto :check_pip
    )
)

:: Try py launcher
py -c "import sys; print('ok')" >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_FOUND=1
    set PIP_CMD=py -m pip
    for /f "tokens=*" %%i in ('py -c "import sys; print(sys.version)"') do set PY_VERSION=%%i
    echo   Python installed via py: !PY_VERSION!
    goto :check_pip
)

:: Try python
python -c "import sys; print('ok')" >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_FOUND=1
    set PIP_CMD=python -m pip
    for /f "tokens=*" %%i in ('python -c "import sys; print(sys.version)"') do set PY_VERSION=%%i
    echo   Python installed: !PY_VERSION!
    goto :check_pip
)

:: Python installed but not in PATH yet
echo.
echo   Python was installed but may need a fresh terminal.
echo.
echo   Please:
echo     1. CLOSE this window
echo     2. OPEN a NEW Command Prompt
echo     3. Navigate to this folder
echo     4. Run: setup.bat
echo.
echo   Or try: python --version
echo   If that works, then run: setup.bat
echo.
pause
exit /b 0
