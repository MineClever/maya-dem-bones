# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Maya Python bindings for the [DemBones](https://github.com/electronicarts/dem-bones) algorithm — automatically extracts Linear Blend Skinning (LBS) weights and bone transformations from animated mesh sequences. The core is a C++17 pybind11 extension (`_core`) that wraps DemBones and interfaces with the Maya C++ SDK, exposed as a Python package (`dem_bones`).

## Build Commands

All builds use Maya's Python interpreter (`mayapy.exe`), not the system Python. The three main
scripts auto-detect installed Maya versions from the Windows registry — no hardcoded paths needed.

**Install into Maya's Python (one-click):**
```bat
QuickBuild.bat
```
Detects installed Maya versions, shows a numbered menu if multiple are found (Enter = latest),
kills any running Maya process, installs build deps, then runs `pip install .`.

**Build a distributable wheel:**
```bat
QuickBuildWheel.bat
```
Same detection logic; outputs `.whl` to `dist/`.

**Generate a Visual Studio 2022 solution:**
```bat
create_vs_sln.bat
```
Generates `build/` with a `.sln` targeting x64; cleans old build dir first.

**Targeting a specific version without prompts:**
```bat
set MAYA_VERSION=2023
QuickBuild.bat
```
Setting `MAYA_PYTHON_VERSION=2` or `=3` additionally overrides Python version detection.

**Maya version → Python version mapping:**
- Maya < 2022 → Python 2 (`mayapy.exe`)
- Maya ≥ 2022 → Python 3 (`mayapy.exe`)
- Maya 2022 + Python 2 override → `mayapy2.exe`

**CMake args forwarded by scikit-build:**
`MAYA_VERSION`, `MAYA_PYTHON_VERSION` are passed via the `CMAKE_ARGS` env var so
`FindMaya.cmake` / `FindMayaPython.cmake` pick up the right SDK automatically.

**Stub regeneration** (after changing C++ API):
```python
# Run inside mayapy
python src/dem_bones/gen_stub.py
```

## Architecture

### Key layers

```
Python user code
    ↓
src/dem_bones/__init__.py      # Python wrapper (DemBones class, timing/RMSE logging)
src/dem_bones/dem_bones_ui.py  # PySide2 Qt UI (show() opens the tool window in Maya)
    ↓
src/dem_bones/_core (pybind11 .pyd)  # C++ extension built from src/main.cpp
    ↓
extern/DemBones/               # EA DemBones algorithm (header-only)
extern/Eigen/                  # Linear algebra (header-only)
Maya C++ SDK (found via FindMaya.cmake)
```

### C++ extension (`src/main.cpp`)

The single class `DemBonesModel : public DemBonesExt<double, float>` does all the heavy work:

- **`compute(source, target, start_frame, end_frame)`** — main entry: calls `extractSource()` (reads skinned mesh + skeleton), `extractTarget()` (reads deformed mesh animation per frame), runs the DemBones solver, then optionally `updateResultSkinWeight()`.
- **`extractSource()`** — reads the Maya skin cluster: influences, bind matrices, locked weights via `demLock` color set.
- **`extractTarget()`** — samples the deforming mesh vertex positions at each frame.
- **`InitBones()`** — when no skeleton exists on the source, auto-creates joints based on `numBones`.
- **`updateResultSkinWeight()`** — writes computed weights back to Maya via `MFnSkinCluster::setWeights`.
- **`bindMatrix()` / `animMatrix()`** — return computed 4×4 matrices (16 floats, column-major) for use in Python to set keyframes.

OpenMP parallelises per-vertex loops. A `std::mutex` guards all Maya API calls that create scene objects (joint creation is not thread-safe).

### Python package (`src/dem_bones/`)

- `__init__.py` — subclasses `_core.DemBones`, overrides `compute()` to add timing and RMSE logging; this is what users import.
- `dem_bones_ui.py` — standalone PySide2 dialog; `show()` is the entry point.
- `_core.pyi` — type stubs (Chinese + English docstrings); authoritative parameter reference.
- `gen_stub.py` — regenerates `_core.pyi` using `mypy.stubgenc`.

### External dependencies (git submodules)

Initialise with:
```bash
git submodule update --init --recursive
```
- `extern/DemBones` — EA's header-only SSDR algorithm
- `extern/Eigen` — header-only linear algebra
- `extern/pybind11` — Python ↔ C++ bindings

### CMake find modules (`modules/`)

- `FindMaya.cmake` — locates Maya SDK by `MAYA_VERSION` env/cache var; sets `Maya::Maya` target.
- `FindMayaPython.cmake` — locates `mayapy.exe` and Maya Python libraries.
- `FindEigen3.cmake` — falls back to the bundled submodule if Eigen is not system-installed.

## Weight/Transform Locking

- **Weight locking**: add a `demLock` color set to the mesh; grayscale value 1.0 = fully locked.
- **Bone transform locking**: add a `demLock` boolean attribute to an influence joint.

Both attribute names are configurable via `lock_weights_set` and `lock_bone_attr_name` on the `DemBones` instance.

## Important Notes

- The package only runs inside Maya (requires Maya C++ SDK at both build and runtime).
- Python 2.7 Maya builds are not supported due to a scikit-build incompatibility.
- The C++ module name is `_core`; the public API is always accessed through `dem_bones.DemBones`.
- `bind_matrix()` and `anim_matrix()` return 16-float tuples in **column-major** order.
- `weights` is a flat list ordered `[vertex0_bone0, vertex0_bone1, ..., vertexN_boneM]`.
