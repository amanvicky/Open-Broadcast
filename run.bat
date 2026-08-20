@echo off
title OpenBroadcast
color 0A
setlocal enabledelayedexpansion

echo.
echo  ========================================
echo   OpenBroadcast - Eye Gaze Correction
echo  ========================================
echo.

:: Find Python
set PYTHON_CMD=

:: Check common locations first
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "C:\Python312\python.exe"
) do (
    if exist %%P (
        set PYTHON_CMD=%%P
        goto :run
    )
)

:: Try py launcher
py -c "import sys; print('ok')" >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=py
    goto :run
)

:: Try python
python -c "import sys; print('ok')" >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    goto :run
)

:: Not found
echo   [ERROR] Python not found!
echo.
echo   Please run "setup.bat" first.
echo.
pause
exit /b 1

:run
:: Quick dependency check
echo   Checking dependencies...
!PYTHON_CMD! -c "import cv2, mediapipe, numpy, PyQt6, psutil" >nul 2>nul
if %errorlevel% neq 0 (
    echo   [!] Missing dependencies. Running setup...
    echo.
    call setup.bat
)

:: Launch the app
echo   Starting OpenBroadcast...
echo.
!PYTHON_CMD! main.py

if %errorlevel% neq 0 (
    echo.
    echo   [!] App exited with error.
    echo.
    echo   Try running: setup.bat
    echo.
    pause
)
