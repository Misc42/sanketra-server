@echo off
setlocal enabledelayedexpansion
title Sanketra Installer
color 0A

echo.
echo  ========================================
echo   Sanketra Installer for Windows
echo  ========================================
echo.

set "INSTALL_DIR=%USERPROFILE%\sanketra"
set "REPO_URL=https://github.com/Misc42/sanketra-server.git"
set "BRANCH=master"

:: ── Check for admin (not required, but warn) ────────────────────────
net session >nul 2>&1
if %errorlevel% equ 0 (
    echo  [i] Running as Administrator
) else (
    echo  [i] Running as normal user ^(recommended^)
)
echo.

:: ── Check/Install Python ─────────────────────────────────────────────
echo  -- Checking Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Python not found. Installing via winget...
    winget install --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent
    if !errorlevel! neq 0 (
        echo  [X] Python install failed.
        echo      Download manually: https://www.python.org/downloads/
        echo      IMPORTANT: Check "Add Python to PATH" during install.
        pause
        exit /b 1
    )
    echo  [i] Refreshing PATH...
    call :RefreshPath
)
:: Verify python works
python --version >nul 2>&1
if %errorlevel% neq 0 (
    :: Try python3
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo  [X] Python still not in PATH. Please restart this installer.
        echo      If that doesn't work, install Python manually and check "Add to PATH".
        pause
        exit /b 1
    )
    set "PYTHON=python3"
) else (
    set "PYTHON=python"
)
for /f "tokens=*" %%i in ('%PYTHON% --version 2^>^&1') do echo  [OK] %%i

:: ── Check/Install Git ────────────────────────────────────────────────
echo.
echo  -- Checking Git...
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Git not found. Installing via winget...
    winget install --id Git.Git --accept-source-agreements --accept-package-agreements --silent
    if !errorlevel! neq 0 (
        echo  [X] Git install failed.
        echo      Download manually: https://git-scm.com/download/win
        pause
        exit /b 1
    )
    echo  [i] Refreshing PATH...
    call :RefreshPath
)
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo  [X] Git still not in PATH. Please restart this installer.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('git --version 2^>^&1') do echo  [OK] %%i

:: ── Clone or Update Repository ───────────────────────────────────────
echo.
echo  -- Setting up Sanketra...
if exist "%INSTALL_DIR%\.git" (
    echo  [i] Existing install found — updating...
    cd /d "%INSTALL_DIR%"
    git fetch origin
    git reset --hard "origin/%BRANCH%"
) else (
    echo  [i] Cloning repository...
    git clone --depth 1 -b "%BRANCH%" "%REPO_URL%" "%INSTALL_DIR%"
)
echo  [OK] Repository ready at %INSTALL_DIR%

:: ── Run setup.py ─────────────────────────────────────────────────────
echo.
echo  -- Running setup ^(venv, dependencies, GPU detection^)...
cd /d "%INSTALL_DIR%"
set PYTHONIOENCODING=utf-8
%PYTHON% setup.py
if %errorlevel% neq 0 (
    echo  [X] Setup failed. Check the output above for errors.
    pause
    exit /b 1
)
echo  [OK] Setup complete

:: ── Install Service ──────────────────────────────────────────────────
echo.
echo  -- Installing service ^(auto-start on login^)...
%PYTHON% setup.py --install-service
if %errorlevel% neq 0 (
    echo  [!] Service install had issues. Server may need manual start.
)

:: ── Add Firewall Rules ───────────────────────────────────────────────
echo.
echo  -- Adding firewall rules...
set "FW_OK=1"
netsh advfirewall firewall delete rule name="sanketra" >nul 2>&1
netsh advfirewall firewall delete rule name="sanketra-udp" >nul 2>&1
netsh advfirewall firewall add rule name="sanketra" dir=in action=allow protocol=tcp localport=5000 >nul 2>&1
if %errorlevel% neq 0 set "FW_OK=0"
netsh advfirewall firewall add rule name="sanketra-udp" dir=in action=allow protocol=udp localport=5001 >nul 2>&1
if %errorlevel% neq 0 set "FW_OK=0"
if "%FW_OK%"=="1" (
    echo  [OK] Firewall rules added
) else (
    echo  [!] Firewall rules need admin. Right-click installer and Run as admin.
)

:: ── Start Service ────────────────────────────────────────────────────
echo.
echo  -- Starting server...
schtasks /run /tn "sanketra" >nul 2>&1
if %errorlevel% equ 0 (
    echo  [OK] Server started
) else (
    echo  [!] Could not auto-start. Server will start on next login.
)

:: ── Done ─────────────────────────────────────────────────────────────
echo.
echo  ========================================
echo   Sanketra is ready!
echo.
echo   Open the Sanketra app on your phone.
echo   Make sure your phone is on the same WiFi.
echo   The app will find this computer automatically.
echo  ========================================
echo.
pause
exit /b 0

:: ── Helper: Refresh PATH from registry ──────────────────────────────
:RefreshPath
set "SYS_PATH="
set "USR_PATH="
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%b"
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USR_PATH=%%b"
if defined SYS_PATH (
    set "PATH=%SYS_PATH%;%USR_PATH%;%WINDIR%;%WINDIR%\System32"
) else (
    :: Registry query failed — keep existing PATH, just append common install dirs
    set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%ProgramFiles%\Git\cmd"
)
goto :eof
