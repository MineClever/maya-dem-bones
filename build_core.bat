@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
:: build_core.bat  <mode>
::
::   mode = "install"  (default)
::             Python 3 : mayapy -m pip install .
::             Python 2 : cmake direct + setup_prebuilt.py install
::
::   mode = "wheel"
::             Python 3 : mayapy setup.py bdist_wheel  -> dist\
::             Python 2 : cmake direct + setup_prebuilt.py bdist_wheel -> dist\
::
:: Maya version and Python version are selected interactively
:: via detect_maya.bat (registry detection + numbered menu).
:: Pre-set MAYA_VERSION / MAYA_PYTHON_VERSION env vars to skip prompts.
:: ============================================================

set "_MODE=%~1"
if "!_MODE!"=="" (
    echo.
    echo  Select build mode:
    echo  ------------------------------------------------
    echo   [1]  Install into Maya Python environment
    echo   [2]  Build distributable wheel  ^(dist\^)
    echo  ------------------------------------------------
    echo.
    set /p "_MODE_CHOICE= Select mode (Enter = default [1]): "
    if "!_MODE_CHOICE!"=="2" (
        set "_MODE=wheel"
    ) else (
        set "_MODE=install"
    )
    echo.
)

:: ---- Detect Maya (shows version/Python menu as needed) ------
call "%~dp0detect_maya.bat"
if errorlevel 1 (
    echo [build] Aborted: Maya detection failed.
    pause
    exit /b 1
)

:: Kill Maya only when installing directly into its environment
if "!_MODE!"=="install" (
    echo Killing running Maya processes...
    taskkill /f /im maya.exe /t 2>nul
    echo.
)

if "%MAYA_PYTHON_VERSION%"=="2" goto :build_py27

:: ==============================================================
:: Python 3 path  --  scikit-build
:: ==============================================================
set "CMAKE_ARGS=-DMAYA_VERSION=%MAYA_VERSION% -DMAYA_PYTHON_VERSION=%MAYA_PYTHON_VERSION% -DCMAKE_POLICY_VERSION_MINIMUM=3.10"

echo Installing build dependencies (Python 3)...
"%MAYA_PYTHON_EXECUTABLE%" -m pip install scikit-build setuptools wheel ninja cmake mypy
echo.

if "!_MODE!"=="wheel" (
    echo Building wheel  --  Maya %MAYA_VERSION%  Python %MAYA_PYTHON_VERSION%
    "%MAYA_PYTHON_EXECUTABLE%" "%~dp0setup.py" bdist_wheel
) else (
    echo Installing  --  Maya %MAYA_VERSION%  Python %MAYA_PYTHON_VERSION%
    "%MAYA_PYTHON_EXECUTABLE%" -m pip install "%~dp0." --no-build-isolation --user
)
goto :done

:: ==============================================================
:: Python 2.7 path  --  cmake direct  (scikit-build incompatible)
:: ==============================================================
:build_py27
set "CMAKE_EXE=cmake"
if exist "C:\Program Files\CMake\bin\cmake.exe" set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"

set "PY27_BUILD_DIR=%~dp0_build_py27"
echo Cleaning Python 2.7 build directory...
if exist "%PY27_BUILD_DIR%" rmdir /s /q "%PY27_BUILD_DIR%"
mkdir "%PY27_BUILD_DIR%"

echo.
echo Configuring cmake  --  Maya %MAYA_VERSION%  Python 2.7
"%CMAKE_EXE%" -G "Visual Studio 17 2022" -A x64 ^
    -DMAYA_VERSION=%MAYA_VERSION% ^
    -DMAYA_PYTHON_VERSION=2 ^
    -DMAYA_INSTALL_BASE_PATH="%MAYA_LOCATION%\.." ^
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
    -B "%PY27_BUILD_DIR%" "%~dp0"
if errorlevel 1 (
    echo [build] cmake configure failed.
    pause & exit /b 1
)

echo.
echo Building _core extension...
"%CMAKE_EXE%" --build "%PY27_BUILD_DIR%" --config Release
if errorlevel 1 (
    echo [build] cmake build failed.
    pause & exit /b 1
)

echo.
echo Copying .pyd to package source tree...
set "_COPIED=0"
for /f "delims=" %%f in ('dir /b /s "%PY27_BUILD_DIR%\Release\_core*.pyd" 2^>nul') do (
    copy /y "%%f" "%~dp0src\dem_bones\" >nul
    echo   %%f
    set "_COPIED=1"
)
if "!_COPIED!"=="0" (
    echo [build] ERROR: _core*.pyd not found in %PY27_BUILD_DIR%\Release\
    pause & exit /b 1
)
echo.

if "!_MODE!"=="wheel" (
    echo Building wheel  --  Python 2.7  ^(no scikit-build^)
    "%MAYA_PYTHON_EXECUTABLE%" "%~dp0setup_prebuilt.py" bdist_wheel
) else (
    echo Installing via setuptools  --  Python 2.7
    "%MAYA_PYTHON_EXECUTABLE%" "%~dp0setup_prebuilt.py" install
)

:done
echo.
if "!_MODE!"=="wheel" (
    echo Wheel written to dist\
) else (
    echo Installation complete.
)
pause
