chcp 65001

set "MAYA_VERSION=2022"
set "MAYAY_PATH=C:/Progra~1/Autodesk/Maya%MAYA_VERSION%"
set "MAYA_PYTHON_EXECUTABLE=%MAYAY_PATH%/bin/mayapy.exe"

set "CMAKE_ARGS=-DCMAKE_POLICY_VERSION_MINIMUM=3.10"

taskkill /f /im maya.exe /t

"%MAYA_PYTHON_EXECUTABLE%" -m pip install scikit-build setuptools wheel ninja cmake mypy
::"%MAYA_PYTHON_EXECUTABLE%" setup.py bdist_wheel
"%MAYA_PYTHON_EXECUTABLE%" -m pip install .

pause