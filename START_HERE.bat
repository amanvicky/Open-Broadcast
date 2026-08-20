@echo off
title OpenBroadcast
color 0A
cls

echo.
echo   ============================================
echo    OpenBroadcast - Eye Gaze Correction
echo   ============================================
echo.

:: Check if Python is available
set FOUND=0

:: Quick check all known paths
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "C:\Python312\python.exe"
) do (
    if exist %%P (set FOUND=1 & goto :launch)
)

py -c "import sys" >nul 2>nul && (set FOUND=1 & set PY=py & goto :launch_check)
python -c "import sys" >nul 2>nul && (set FOUND=1 & set PY=python & goto :launch_check)

:: Python not found - redirect to full installer
if %FOUND% equ 0 (
    echo   Python not found. Running full installer...
    echo.
    call install_and_run.bat
    goto :eof
)

:launch_check
set PY=py
goto :launch

:launch
echo   Starting OpenBroadcast...
echo.

:: Find python path
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "C:\Python312\python.exe"
) do (
    if exist %%P (
        %%P main.py
        goto :eof
    )
)

py main.py 2>nul && goto :eof
python main.py

:: If we get here, something went wrong
if %errorlevel% neq 0 (
    echo.
    echo   [!] Error starting application.
    echo   Running full installer...
    echo.
    call install_and_run.bat
)
