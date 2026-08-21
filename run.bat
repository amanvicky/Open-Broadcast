@echo off
title OpenBroadcast - Eye Gaze Correction
cd /d "%~dp0"

:: Check for embedded Python
if exist "python\python.exe" (
    echo Starting OpenBroadcast...
    "python\python.exe" main.py
) else (
    echo Python not found in app directory.
    echo Trying system Python...
    python main.py
    if errorlevel 1 (
        echo.
        echo ERROR: Python not found.
        echo Please install Python 3.10+ from https://python.org
        echo or run setup.bat to install dependencies.
        pause
    )
)
