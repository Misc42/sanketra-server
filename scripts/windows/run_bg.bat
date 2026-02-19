@echo off
setlocal enabledelayedexpansion
REM Sanketra - Windows launcher
REM Background mode for web (trackpad works when minimized)

set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..\..\
set VENV_PYTHON=%ROOT_DIR%venv\Scripts\python.exe
set VENV_PYTHONW=%ROOT_DIR%venv\Scripts\pythonw.exe

REM Check venv exists
if not exist "%VENV_PYTHON%" (
    echo Error: Virtual environment not found. Run setup.py first:
    echo   python setup.py
    exit /b 1
)

REM If no arguments, run interactive Python launcher
if "%1"=="" (
    "%VENV_PYTHON%" "%ROOT_DIR%src\run.py"
    exit /b
)

REM Parse mode
set MODE=%1
shift

if "%MODE%"=="cli" (
    "%VENV_PYTHON%" "%ROOT_DIR%src\stt_vad.py" %1 %2 %3 %4 %5
    exit /b
)

if "%MODE%"=="web" (
    echo.
    echo   Starting web mode in background...
    echo   (Trackpad works when minimized)
    echo.
    echo   QR code will open as image - scan with phone
    echo   To stop: Task Manager ^> End pythonw.exe
    echo.
    start "" "%VENV_PYTHONW%" "%ROOT_DIR%src\server_async.py" --background
    timeout /t 3 /nobreak >nul
    echo   Server started! QR image should open automatically.
    echo.
    exit /b
)

if "%MODE%"=="-h" goto :help
if "%MODE%"=="--help" goto :help

echo Unknown mode: %MODE%
echo Use 'cli' or 'web', or run without args for interactive mode
exit /b 1

:help
echo.
echo   Sanketra (Windows)
echo   ---------------------
echo.
echo   Usage:
echo     run_bg.bat              Interactive launcher
echo     run_bg.bat cli          Local mic mode
echo     run_bg.bat web          Web mode (phone mic + trackpad)
echo.
echo   Model selection happens interactively when the app starts.
echo.
exit /b 0
