@echo off
title OpenBroadcast - One-Click Installer & Launcher
color 0B
setlocal enabledelayedexpansion

:: ================================================================
::  OPENBROADCAST - FULLY AUTOMATED INSTALLER & LAUNCHER
::  Just double-click this file. That's it.
:: ================================================================

cls
echo.
echo   ============================================
echo    OpenBroadcast - Eye Gaze Correction
echo    Automated Installer ^& Launcher
echo   ============================================
echo.
echo   This will:
echo     1. Install Python (if not present)
echo     2. Install all required packages
echo     3. Launch the application
echo.
echo   Please wait, this may take a few minutes...
echo.
echo   ============================================
echo.

:: ================================================================
::  PHASE 1: FIND OR INSTALL PYTHON
:: ================================================================
echo  [Phase 1] Checking for Python...
echo.

set PYTHON_CMD=
set NEED_INSTALL=0

:: --- Try to find existing Python ---

:: Check common install paths
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    if not defined PYTHON_CMD (
        if exist %%P (
            %%P -c "import sys" >nul 2>nul
            if !errorlevel! equ 0 (
                set PYTHON_CMD=%%P
            )
        )
    )
)

:: Try py launcher
if not defined PYTHON_CMD (
    py -c "import sys" >nul 2>nul
    if !errorlevel! equ 0 (
        set PYTHON_CMD=py
    )
)

:: Try python command
if not defined PYTHON_CMD (
    python -c "import sys" >nul 2>nul
    if !errorlevel! equ 0 (
        set PYTHON_CMD=python
    )
)

:: --- If not found, install it ---
if not defined PYTHON_CMD (
    echo    Python not found. Installing automatically...
    echo.
    goto :install_python
) else (
    for /f "tokens=*" %%i in ('!PYTHON_CMD! -c "import sys; print(sys.version)"') do set PY_VER=%%i
    echo    Python found: !PY_VER!
    echo.
    goto :install_deps
)


:: ================================================================
::  PHASE 1B: DOWNLOAD AND INSTALL PYTHON
:: ================================================================
:install_python
echo.
echo    Downloading Python 3.12.5...
echo.

:: Create temp directory
if not exist "%TEMP%\ob_install" mkdir "%TEMP%\ob_install"

set PYTHON_URL=https://www.python.org/ftp/python/3.12.5/python-3.12.5-amd64.exe
set PYTHON_EXE=%TEMP%\ob_install\python-installer.exe

:: Download using certutil (built into all Windows versions)
certutil -urlcache -split -f "%PYTHON_URL%" "%PYTHON_EXE%" >nul 2>nul

if not exist "%PYTHON_EXE%" (
    echo.
    echo    [!] Download failed. Trying alternative method...
    echo.
    :: Try PowerShell as fallback
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_EXE%'" >nul 2>nul
)

if not exist "%PYTHON_EXE%" (
    echo    [ERROR] Could not download Python.
    echo    Please check your internet connection.
    echo.
    pause
    exit /b 1
)

echo    Download complete! Installing Python...
echo    (This is silent, no clicks needed)
echo.

:: Install Python silently
:: /quiet = no UI
:: InstallAllUsers=0 = just for this user
:: PrependPath=1 = add to PATH
:: Include_pip=1 = include pip
"%PYTHON_EXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0 DefaultJustForMeTargetDir="%LOCALAPPDATA%\Programs\Python\Python312"

:: Wait for installer to complete
echo    Waiting for installation to finish...
timeout /t 15 /nobreak >nul

:: Check if installer is still running
tasklist /fi "imagename eq python-3.12.5-amd64.exe" 2>nul | find /i "python-3.12.5" >nul
if %errorlevel% equ 0 (
    echo    Still installing, please wait...
    timeout /t 20 /nobreak >nul
)

:: Refresh PATH for this session
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"

:: Verify Python is available
set PYTHON_CMD=

:: Check install paths
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
) do (
    if not defined PYTHON_CMD (
        if exist %%P (
            %%P -c "import sys" >nul 2>nul
            if !errorlevel! equ 0 (
                set PYTHON_CMD=%%P
            )
        )
    )
)

:: Try py/python again
if not defined PYTHON_CMD (
    py -c "import sys" >nul 2>nul
    if !errorlevel! equ 0 set PYTHON_CMD=py
)
if not defined PYTHON_CMD (
    python -c "import sys" >nul 2>nul
    if !errorlevel! equ 0 set PYTHON_CMD=python
)

if not defined PYTHON_CMD (
    echo.
    echo    Python installed but not in current PATH.
    echo    Relaunching setup...
    echo.
    :: Re-launch this script with fresh environment
    cmd /c "set PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%" "%~f0"
    exit /b
)

for /f "tokens=*" %%i in ('!PYTHON_CMD! -c "import sys; print(sys.version)"') do set PY_VER=%%i
echo    Python installed: !PY_VER!
echo.


:: ================================================================
::  PHASE 2: INSTALL DEPENDENCIES
:: ================================================================
:install_deps
echo  [Phase 2] Installing required packages...
echo.

:: Upgrade pip first
echo    Upgrading pip...
!PYTHON_CMD! -m pip install --upgrade pip --quiet 2>nul

:: Install all required packages
echo    Installing OpenCV, MediaPipe, PyQt6, and other dependencies...
echo    (This may take 2-5 minutes on first install)
echo.

!PYTHON_CMD! -m pip install opencv-python mediapipe numpy PyQt6 psutil py-cpuinfo --quiet 2>nul
if %errorlevel% neq 0 (
    echo    Installing packages one by one...
    for %%P in (opencv-python mediapipe numpy PyQt6 psutil py-cpuinfo) do (
        echo      - %%P
        !PYTHON_CMD! -m pip install %%P --quiet 2>nul
    )
)

:: Install optional packages (don't fail if these don't work)
echo    Installing optional packages...
!PYTHON_CMD! -m pip install pyvirtualcam onnxruntime WMI screeninfo --quiet 2>nul

echo.
echo    Dependencies installed!
echo.


:: ================================================================
::  PHASE 3: VERIFY
:: ================================================================
echo  [Phase 3] Verifying installation...
echo.

set PASS=0
set FAIL=0

!PYTHON_CMD! -c "import cv2" >nul 2>nul && (echo    [OK] OpenCV & set /a PASS+=1) || (echo    [!!] OpenCV - will try to install & set /a FAIL+=1)
!PYTHON_CMD! -c "import mediapipe" >nul 2>nul && (echo    [OK] MediaPipe & set /a PASS+=1) || (echo    [!!] MediaPipe - will try to install & set /a FAIL+=1)
!PYTHON_CMD! -c "import numpy" >nul 2>nul && (echo    [OK] NumPy & set /a PASS+=1) || (echo    [!!] NumPy - will try to install & set /a FAIL+=1)
!PYTHON_CMD! -c "import PyQt6" >nul 2>nul && (echo    [OK] PyQt6 & set /a PASS+=1) || (echo    [!!] PyQt6 - will try to install & set /a FAIL+=1)
!PYTHON_CMD! -c "import psutil" >nul 2>nul && (echo    [OK] psutil & set /a PASS+=1) || (echo    [!!] psutil - will try to install & set /a FAIL+=1)

:: Retry failed packages
if %FAIL% gtr 0 (
    echo.
    echo    Retrying failed packages...
    !PYTHON_CMD! -m pip install opencv-python mediapipe numpy PyQt6 psutil py-cpuinfo --quiet 2>nul
)

echo.


:: ================================================================
::  PHASE 4: LAUNCH APPLICATION
:: ================================================================
echo  [Phase 4] Launching OpenBroadcast...
echo.
echo   ============================================
echo    Starting application...
echo   ============================================
echo.

!PYTHON_CMD! main.py

:: If app crashed
if %errorlevel% neq 0 (
    echo.
    echo   ============================================
    echo    Application exited with an error.
    echo   ============================================
    echo.
    echo   Try running this script again,
    echo   or check the error messages above.
    echo.
    pause
)
