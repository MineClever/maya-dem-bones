# -*- coding: utf-8 -*-
import sys
from typing import Optional

from maya import cmds
from maya.api import OpenMaya

# Import the pybind11 module (exposed as dem_bones.DemBones)
import dem_bones

# PySide2 for UI inside Maya
from PySide2 import QtCore, QtWidgets
from shiboken2 import wrapInstance

# Get Maya main window as a QWidget
def _maya_main_window() -> Optional[QtWidgets.QWidget]:
    import maya.OpenMayaUI as omui
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget) if ptr else None


def validate_selection():
    """
    Validate Maya selection for source skinned mesh and target animated mesh.
    Returns (source_mesh_name, target_mesh_name) or (None, None) if invalid.
    """
    sel = OpenMaya.MGlobal.getActiveSelectionList() # type: OpenMaya.MSelectionList
    if sel.length() < 2:
        OpenMaya.MGlobal.displayWarning("Select a skinned mesh first, then the target animated mesh.")
        return None, None

    # First selected: skinned mesh
    sk_mesh_dag = sel.getDagPath(0) # type: OpenMaya.MDagPath
    sk_mesh_name = sk_mesh_dag.partialPathName()
    sk_mesh_dag.extendToShape()

    # Last selected: target animated mesh
    vtx_anim_mesh_dag = sel.getDagPath(sel.length() - 1) # type: OpenMaya.MDagPath
    vtx_anim_mesh_name = vtx_anim_mesh_dag.partialPathName()
    vtx_anim_mesh_dag.extendToShape()

    # Vertex count check
    sk_mesh_fn = OpenMaya.MFnMesh(sk_mesh_dag)
    vtx_anim_mesh_fn = OpenMaya.MFnMesh(vtx_anim_mesh_dag)
    if sk_mesh_fn.numVertices != vtx_anim_mesh_fn.numVertices:
        OpenMaya.MGlobal.displayWarning(
            "Vertex count mismatch: <%s> vs <%s>" %
            (sk_mesh_dag.partialPathName(), vtx_anim_mesh_dag.partialPathName())
        )
        return None, None

    return sk_mesh_name, vtx_anim_mesh_name


def run_dem_bones(source_mesh: str, target_mesh: str, start_frame: int, end_frame: int,
                  apply_anim: bool, update_weights: bool):
    """
    Run DemBones compute and optionally apply animation keys and weights via C++ API.
    """
    db = dem_bones.DemBones()
    db.compute(source_mesh, target_mesh, start_frame=start_frame, end_frame=end_frame)

    # Apply animation curves and weights directly via C++ for performance
    OpenMaya.MGlobal.displayInfo("trying to apply animation and weights...")
    if apply_anim:
        db.apply_animation_and_weights(source_mesh,update_weights)

    # Feedback
    OpenMaya.MGlobal.displayInfo(
        "DemBones done. Frames %d-%d, influences: %d, RMSE: %.6f"
        % (db.start_frame, db.end_frame, len(db.influences), db.rmse())
    )


class DemBonesUI(QtWidgets.QDialog):
    """
    Simple PySide2 UI for DemBones workflow.
    """

    def __init__(self, parent=_maya_main_window()):
        super(DemBonesUI, self).__init__(parent)
        self.setWindowTitle("DemBones Decomposition")
        self.setObjectName("DemBonesDecompositionUI")
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        self.setMinimumWidth(360)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        # Widgets
        self.source_label = QtWidgets.QLabel("Source skinned mesh:")
        self.source_line = QtWidgets.QLineEdit()
        self.target_label = QtWidgets.QLabel("Target animated mesh:")
        self.target_line = QtWidgets.QLineEdit()

        self.pick_btn = QtWidgets.QPushButton("Use selection")
        self.start_spin = QtWidgets.QSpinBox()
        self.end_spin = QtWidgets.QSpinBox()
        self.start_spin.setRange(-100000, 100000)
        self.end_spin.setRange(-100000, 100000)
        self.start_spin.setValue(0)
        self.end_spin.setValue(100)

        self.apply_anim_chk = QtWidgets.QCheckBox("Apply animation keys")
        self.apply_anim_chk.setChecked(True)
        self.update_weights_chk = QtWidgets.QCheckBox("Update skin weights")
        self.update_weights_chk.setChecked(True)

        self.run_btn = QtWidgets.QPushButton("Run DemBones")
        self.close_btn = QtWidgets.QPushButton("Close")

        # Layouts
        form = QtWidgets.QFormLayout()
        form.addRow(self.source_label, self.source_line)
        form.addRow(self.target_label, self.target_line)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Start frame:"), 0, 0)
        grid.addWidget(self.start_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("End frame:"), 1, 0)
        grid.addWidget(self.end_spin, 1, 1)

        opts = QtWidgets.QVBoxLayout()
        opts.addWidget(self.apply_anim_chk)
        opts.addWidget(self.update_weights_chk)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.close_btn)

        main = QtWidgets.QVBoxLayout(self)
        main.addLayout(form)
        main.addWidget(self.pick_btn)
        main.addLayout(grid)
        main.addLayout(opts)
        main.addSpacing(8)
        main.addLayout(buttons)

    def _connect_signals(self):
        self.pick_btn.clicked.connect(self._fill_from_selection)
        self.run_btn.clicked.connect(self._on_run)
        self.close_btn.clicked.connect(self.close)

    def _fill_from_selection(self):
        src, tgt = validate_selection()
        if src and tgt:
            self.source_line.setText(src)
            self.target_line.setText(tgt)

    def _on_run(self):
        src = self.source_line.text().strip()
        tgt = self.target_line.text().strip()
        start_frame = int(self.start_spin.value())
        end_frame = int(self.end_spin.value())
        apply_anim = self.apply_anim_chk.isChecked()
        update_weights = self.update_weights_chk.isChecked()

        if not src or not tgt:
            OpenMaya.MGlobal.displayWarning("Please specify source and target meshes.")
            return
        if start_frame >= end_frame:
            OpenMaya.MGlobal.displayWarning("Start frame must be less than end frame.")
            return

        try:
            run_dem_bones(src, tgt, start_frame, end_frame, apply_anim, update_weights)
        except Exception as e:
            OpenMaya.MGlobal.displayError("DemBones failed: %s" % e)

    @staticmethod
    def show_window():
        # Ensure single instance
        for w in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(w, DemBonesUI):
                w.close()
                break
        ui = DemBonesUI()
        ui.show()
        return ui


def show():
    """
    Entry point to show the UI in Maya.
    """
    return DemBonesUI.show_window()


if __name__ == "__main__":
    # For direct execution in mayapy or Script Editor
    DemBonesUI.show_window()
