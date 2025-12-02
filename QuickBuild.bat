chcp 65001

set "MAYA_VERSION=2022"
set "MAYAY_PATH=C:/Progra~1/Autodesk/Maya%MAYA_VERSION%"
set "MAYA_PYTHON_EXECUTABLE=%MAYAY_PATH%/bin/mayapy.exe"

"%MAYAY_PATH%/bin/mayapy.exe" -m pip install scikit-build setuptools wheel ninja cmake
"%MAYAY_PATH%/bin/mayapy.exe" setup.py bdist_wheel
"%MAYAY_PATH%/bin/mayapy.exe" -m pip install .

pause