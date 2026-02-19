@echo off
REM mic_on_term - Update script (Windows)

set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..\..\
cd /d "%ROOT_DIR%"

echo.
echo   Updating mic_on_term...
echo   -------------------------
echo.

REM Check if git repo
if not exist ".git" (
    echo   Error: Not a git repository
    exit /b 1
)

REM Pull latest
echo   Pulling latest changes...
git pull

if errorlevel 1 (
    echo.
    echo   Error: git pull failed
    exit /b 1
)

REM Check if venv exists
set VENV_PIP=%ROOT_DIR%venv\Scripts\pip.exe
if exist "%VENV_PIP%" (
    echo.
    echo   Updating packages...
    "%VENV_PIP%" install -r requirements.txt -q --upgrade
    echo   Done.
) else (
    echo.
    echo   Note: venv not found. Run setup.py first:
    echo     python setup.py
)

echo.
echo   -------------------------
echo   Update complete!
echo.
