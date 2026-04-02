@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
:: QuickBuildWheel.bat
:: Detects Maya from the registry and builds a distributable
:: Python wheel (.whl) into the dist\ directory.
::
:: Python 3 path : scikit-build via setup.py bdist_wheel
:: Python 2.7 path: cmake direct + setup_prebuilt.py bdist_wheel
::
:: To target a specific version without prompts, set env vars
:: before running this script:
::   set MAYA_VERSION=2023
::   set MAYA_PYTHON_VERSION=3   (optional override)
:: ============================================================

call "%~dp0detect_maya.bat"
if errorlevel 1 (
    echo [QuickBuildWheel] Aborted: Maya detection failed.
    pause
    exit /b 1
)

if "%MAYA_PYTHON_VERSION%"=="2" goto :build_py27

:: ---- Python 3: scikit-build path ----
set "CMAKE_ARGS=-DMAYA_VERSION=%MAYA_VERSION% -DMAYA_PYTHON_VERSION=%MAYA_PYTHON_VERSION% -DCMAKE_POLICY_VERSION_MINIMUM=3.10"

echo Installing build dependencies (Python 3)...
"%MAYA_PYTHON_EXECUTABLE%" -m pip install scikit-build setuptools wheel ninja cmake mypy

echo.
echo Building wheel for Maya %MAYA_VERSION% (Python %MAYA_PYTHON_VERSION%)...
"%MAYA_PYTHON_EXECUTABLE%" setup.py bdist_wheel
goto :done

:: ---- Python 2.7: cmake-direct path (scikit-build not compatible) ----
:build_py27
set "CMAKE_EXE=cmake"
if exist "C:\Program Files\CMake\bin\cmake.exe" set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"

set "PY27_BUILD_DIR=%~dp0_build_py27"
echo Cleaning Python 2.7 build directory...
if exist "%PY27_BUILD_DIR%" rmdir /s /q "%PY27_BUILD_DIR%"
mkdir "%PY27_BUILD_DIR%"

echo Configuring cmake for Maya %MAYA_VERSION% Python 2.7...
"%CMAKE_EXE%" -G "Visual Studio 17 2022" -A x64 ^
    -DMAYA_VERSION=%MAYA_VERSION% ^
    -DMAYA_PYTHON_VERSION=2 ^
    -DMAYA_INSTALL_BASE_PATH="%MAYA_LOCATION%\.." ^
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
    -B "%PY27_BUILD_DIR%" "%~dp0"
if errorlevel 1 (
    echo [QuickBuildWheel] cmake configure failed.
    pause & exit /b 1
)

echo.
echo Building _core extension...
"%CMAKE_EXE%" --build "%PY27_BUILD_DIR%" --config Release
if errorlevel 1 (
    echo [QuickBuildWheel] cmake build failed.
    pause & exit /b 1
)

echo.
echo Copying .pyd to package source tree...
for /f "delims=" %%f in ('dir /b /s "%PY27_BUILD_DIR%\Release\_core*.pyd" 2^>nul') do (
    copy /y "%%f" "%~dp0src\dem_bones\"
    echo Copied: %%f
)

echo.
echo Building wheel (Python 2.7, no scikit-build)...
"%MAYA_PYTHON_EXECUTABLE%" "%~dp0setup_prebuilt.py" bdist_wheel

:done
echo.
echo Wheel written to dist\
pause
