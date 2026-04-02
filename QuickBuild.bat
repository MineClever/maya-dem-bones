@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
:: QuickBuild.bat
:: Detects Maya from the registry and installs the package
:: into that Maya's Python environment via pip.
::
:: To target a specific version without prompts, set env vars
:: before running this script:
::   set MAYA_VERSION=2023
::   set MAYA_PYTHON_VERSION=3   (optional override)
:: ============================================================

call "%~dp0detect_maya.bat"
if errorlevel 1 (
    echo [QuickBuild] Aborted: Maya detection failed.
    pause
    exit /b 1
)

:: Pass Maya version info to scikit-build / CMake
set "CMAKE_ARGS=-DMAYA_VERSION=%MAYA_VERSION% -DMAYA_PYTHON_VERSION=%MAYA_PYTHON_VERSION% -DCMAKE_POLICY_VERSION_MINIMUM=3.10"

echo Killing running Maya processes...
taskkill /f /im maya.exe /t 2>nul

echo.
echo Installing build dependencies...
if "%MAYA_PYTHON_VERSION%"=="2" (
    :: mypy requires Python 3 -- skip it for Python 2 builds
    "%MAYA_PYTHON_EXECUTABLE%" -m pip install scikit-build setuptools wheel ninja cmake
) else (
    "%MAYA_PYTHON_EXECUTABLE%" -m pip install scikit-build setuptools wheel ninja cmake mypy
)

echo.
echo Building and installing dem_bones for Maya %MAYA_VERSION% (Python %MAYA_PYTHON_VERSION%)...
"%MAYA_PYTHON_EXECUTABLE%" -m pip install .

echo.
echo Done.
pause
