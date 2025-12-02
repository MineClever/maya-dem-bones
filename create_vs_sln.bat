@echo off
setlocal

REM === 配置路径 ===
set MAYA_VERSION=2022
set "MAYA_LOCATION=C:/Program Files/Autodesk/Maya2022"

set PROJECT_DIR=%~dp0
set BUILD_DIR=%PROJECT_DIR%build
set "CMAKE_EXE=C:/Program Files/CMake/bin/cmake.exe"

REM === 清理旧的 build 目录 ===
if exist "%BUILD_DIR%" (
    echo Removing old build directory...
    rmdir /s /q "%BUILD_DIR%"
)

REM === 创建新的 build 目录 ===
mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

REM === 调用 CMake 生成 Visual Studio 2022 的解决方案 ===
"%CMAKE_EXE%" -G "Visual Studio 17 2022" -A x64 %PROJECT_DIR%

echo.
echo Visual Studio solution has been generated in: %BUILD_DIR%
echo Open the .sln file with Visual Studio 2022.
pause
