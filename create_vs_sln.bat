@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

:: ============================================================
:: create_vs_sln.bat
:: Detects Maya from the registry using the same selection flow
:: as QuickBuild.bat/build_core.bat, then generates a Visual
:: Studio 2022 solution in the build\ directory for IDE-based
:: editing and debugging.
::
:: To target a specific version without prompts, set env vars
:: before running this script:
::   set MAYA_VERSION=2023
::   set MAYA_PYTHON_VERSION=3   (optional override)
:: ============================================================

call "%~dp0detect_maya.bat"
if errorlevel 1 (
    echo [create_vs_sln] Aborted: Maya detection failed.
    pause
    exit /b 1
)

if "%MAYA_PYTHON_VERSION%"=="2" (
    if not exist "%REPO_ROOT%\extern\pybind11_py27\CMakeLists.txt" (
        echo [create_vs_sln] Missing Python 2 pybind11 submodule: extern\pybind11_py27
        echo [create_vs_sln] Run:
        echo                 git submodule update --init extern/pybind11_py27
        pause
        exit /b 1
    )
    set "CMAKE_POLICY_VERSION_MINIMUM=3.5"
) else (
    set "CMAKE_POLICY_VERSION_MINIMUM=3.10"
)

set "BUILD_DIR=%REPO_ROOT%\build"

:: Locate cmake (prefer system cmake, fallback to PATH)
set "CMAKE_EXE=cmake"
if exist "C:\Program Files\CMake\bin\cmake.exe" (
    set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"
)

:: Clean old build directory
if exist "%BUILD_DIR%" (
    echo Removing old build directory...
    rmdir /s /q "%BUILD_DIR%"
)
mkdir "%BUILD_DIR%"

echo Generating Visual Studio 2022 solution  --  Maya %MAYA_VERSION%  Python %MAYA_PYTHON_VERSION%
echo.

"%CMAKE_EXE%" ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DMAYA_VERSION=%MAYA_VERSION% ^
    -DMAYA_PYTHON_VERSION=%MAYA_PYTHON_VERSION% ^
    -DMAYA_INSTALL_BASE_PATH="%MAYA_LOCATION%\.." ^
    -DCMAKE_POLICY_VERSION_MINIMUM=%CMAKE_POLICY_VERSION_MINIMUM% ^
    -B "%BUILD_DIR%" ^
    "%REPO_ROOT%"

if errorlevel 1 (
    echo.
    echo [create_vs_sln] CMake configuration failed.
    pause
    exit /b 1
)

echo.
echo Visual Studio solution generated in: %BUILD_DIR%
echo Open the .sln file with Visual Studio 2022.
pause
