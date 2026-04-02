@echo off
:: ============================================================
:: detect_maya.bat
:: Detects installed Maya versions from the Windows registry.
::
:: Exported variables (via endlocal passback):
::   MAYA_VERSION          e.g. 2025
::   MAYA_LOCATION         e.g. C:\Program Files\Autodesk\Maya2025
::   MAYA_PYTHON_VERSION   2 or 3
::   MAYA_PYTHON_EXECUTABLE  full path to mayapy.exe (or mayapy2.exe)
::
:: Callers can pre-set MAYA_VERSION and/or MAYA_PYTHON_VERSION as
:: environment variables to skip the interactive prompts.
:: ============================================================

setlocal enabledelayedexpansion

set "_MAYA_VERSION=%MAYA_VERSION%"
set "_MAYA_PYTHON_VERSION=%MAYA_PYTHON_VERSION%"

:: ---- Step 1: resolve version --------------------------------
if defined _MAYA_VERSION (
    echo [detect_maya] Using pre-set MAYA_VERSION=!_MAYA_VERSION!
    goto :resolve_location
)

:: Query registry for installed Maya versions (4-digit year keys)
set "_COUNT=0"
for /f "delims=" %%v in ('powershell -NoProfile -Command ^
    "Get-ChildItem 'HKLM:\SOFTWARE\Autodesk\Maya' -ErrorAction SilentlyContinue ^
    | Where-Object { $_.PSChildName -match '^\d{4}$' } ^
    | Sort-Object { [int]$_.PSChildName } -Descending ^
    | ForEach-Object { $_.PSChildName }" 2^>nul') do (
    set /a "_COUNT+=1"
    set "_VER_!_COUNT!=%%v"
)

if !_COUNT! == 0 (
    echo [detect_maya] ERROR: No Maya installation found in registry.
    echo              Please set MAYA_VERSION manually before calling this script.
    endlocal
    exit /b 1
)

if !_COUNT! == 1 (
    set "_MAYA_VERSION=!_VER_1!"
    echo [detect_maya] Found Maya !_MAYA_VERSION!
    goto :resolve_location
)

:: Multiple versions found -- show menu
echo.
echo [detect_maya] Multiple Maya installations found:
for /l %%i in (1,1,!_COUNT!) do (
    echo   [%%i] Maya !_VER_%%i!
)
echo.
set /p "_CHOICE=Select version (1=latest, Enter=default 1): "
if "!_CHOICE!"=="" set "_CHOICE=1"

:: Validate choice is a number in range
set "_OK=0"
for /l %%i in (1,1,!_COUNT!) do (
    if "!_CHOICE!"=="%%i" set "_OK=1"
)
if "!_OK!"=="0" (
    echo [detect_maya] Invalid choice '!_CHOICE!', using default (1).
    set "_CHOICE=1"
)

for %%i in (!_CHOICE!) do set "_MAYA_VERSION=!_VER_%%i!"
echo [detect_maya] Selected Maya !_MAYA_VERSION!

:resolve_location
:: ---- Step 2: get install path from registry -----------------
set "_MAYA_LOCATION="
for /f "delims=" %%p in ('powershell -NoProfile -Command ^
    "(Get-ItemProperty 'HKLM:\SOFTWARE\Autodesk\Maya\!_MAYA_VERSION!\Setup\InstallPath' ^
    -ErrorAction SilentlyContinue).MAYA_INSTALL_LOCATION" 2^>nul') do (
    set "_MAYA_LOCATION=%%p"
)

if not defined _MAYA_LOCATION (
    echo [detect_maya] ERROR: Could not read INSTALLDIR from registry for Maya !_MAYA_VERSION!.
    endlocal
    exit /b 1
)

:: Strip trailing backslash if present
if "!_MAYA_LOCATION:~-1!"=="\" set "_MAYA_LOCATION=!_MAYA_LOCATION:~0,-1!"

:: ---- Step 3: resolve Python version -------------------------
if defined _MAYA_PYTHON_VERSION (
    echo [detect_maya] Using pre-set MAYA_PYTHON_VERSION=!_MAYA_PYTHON_VERSION!
    goto :resolve_executable
)

:: Default: >=2022 -> Python 3, older -> Python 2
if !_MAYA_VERSION! GEQ 2022 (
    set "_MAYA_PYTHON_VERSION=3"
) else (
    set "_MAYA_PYTHON_VERSION=2"
)

:: Maya 2022 ships both Python 2 and 3 -- ask the user
if "!_MAYA_VERSION!"=="2022" (
    echo.
    set /p "_PY_CHOICE=Maya 2022 supports Python 2 and 3. Choose version [2/3, default=3]: "
    if "!_PY_CHOICE!"=="2" set "_MAYA_PYTHON_VERSION=2"
    if "!_PY_CHOICE!"=="3" set "_MAYA_PYTHON_VERSION=3"
    :: any other input keeps the default (3)
)

:resolve_executable
:: ---- Step 4: locate mayapy executable -----------------------
if "!_MAYA_VERSION!"=="2022" if "!_MAYA_PYTHON_VERSION!"=="2" (
    set "_MAYA_PYTHON_EXECUTABLE=!_MAYA_LOCATION!\bin\mayapy2.exe"
) else (
    set "_MAYA_PYTHON_EXECUTABLE=!_MAYA_LOCATION!\bin\mayapy.exe"
)

if not defined _MAYA_PYTHON_EXECUTABLE (
    set "_MAYA_PYTHON_EXECUTABLE=!_MAYA_LOCATION!\bin\mayapy.exe"
)

if not exist "!_MAYA_PYTHON_EXECUTABLE!" (
    echo [detect_maya] ERROR: mayapy not found at: !_MAYA_PYTHON_EXECUTABLE!
    endlocal
    exit /b 1
)

echo [detect_maya] Maya     : !_MAYA_VERSION!
echo [detect_maya] Location : !_MAYA_LOCATION!
echo [detect_maya] Python   : !_MAYA_PYTHON_VERSION!
echo [detect_maya] mayapy   : !_MAYA_PYTHON_EXECUTABLE!
echo.

:: ---- Passback variables to calling scope --------------------
endlocal ^
  & set "MAYA_VERSION=%_MAYA_VERSION%" ^
  & set "MAYA_LOCATION=%_MAYA_LOCATION%" ^
  & set "MAYA_PYTHON_VERSION=%_MAYA_PYTHON_VERSION%" ^
  & set "MAYA_PYTHON_EXECUTABLE=%_MAYA_PYTHON_EXECUTABLE%"

exit /b 0
