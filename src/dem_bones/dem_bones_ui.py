from __future__ import annotations

import math
from maya import cmds
from maya import OpenMayaUI as omui
from maya.api import OpenMaya as om2
from maya.api import OpenMayaAnim as omanim2

from shiboken2 import wrapInstance
from PySide2 import QtWidgets, QtCore
import dem_bones


class CollapsibleGroupBox(QtWidgets.QWidget):
    """A simple collapsible group box with a clickable header.

    Usage:
        g = CollapsibleGroupBox("Title", collapsed=True)
        g.setContentLayout(a_layout)
        parent_layout.addWidget(g)
    """
    def __init__(self, title: str = "", parent=None, collapsed: bool = True):
        super(CollapsibleGroupBox, self).__init__(parent)

        self._toggle = QtWidgets.QToolButton(text=title)
        self._toggle.setStyleSheet("QToolButton { border: none; }")
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        # checked == expanded
        self._toggle.setChecked(not collapsed)
        self._toggle.setArrowType(QtCore.Qt.DownArrow if not collapsed else QtCore.Qt.RightArrow)
        self._toggle.clicked.connect(self._on_toggled)

        self._content = QtWidgets.QWidget()
        self._content.setVisible(not collapsed)
        # when collapsed we set maximum height to 0 so layouts shrink
        if collapsed:
            self._content.setMaximumHeight(0)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._toggle)
        lay.addWidget(self._content)

    def setContentLayout(self, layout: QtWidgets.QLayout) -> None:
        # replace existing layout if any
        old = self._content.layout()
        if old is not None:
            QtWidgets.QWidget().setLayout(old)
        self._content.setLayout(layout)
        # update maximum height according to current toggle state so
        # parent layouts can shrink when collapsed
        try:
            if not self._toggle.isChecked():
                self._content.setMaximumHeight(0)
            else:
                # allow content to size to its hint
                self._content.setMaximumHeight(self._content.sizeHint().height() or 16777215)
        except Exception:
            pass
        try:
            self.updateGeometry()
            win = self.window()
            if win is not None:
                win.adjustSize()
        except Exception:
            pass

    def _on_toggled(self):
        checked = self._toggle.isChecked()
        self._content.setVisible(checked)
        self._toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        try:
            if not checked:
                self._content.setMaximumHeight(0)
            else:
                # restore to content preferred height
                self._content.setMaximumHeight(self._content.sizeHint().height() or 16777215)
        except Exception:
            pass
        # notify layouts and top-level window to adjust size
        try:
            self.updateGeometry()
            win = self.window()
            if win is not None:
                win.adjustSize()
        except Exception:
            pass


def get_maya_main_window():
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget)


class DemBonesUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        if parent is None:
            parent = get_maya_main_window()
        super(DemBonesUI, self).__init__(parent)
        self.setWindowTitle("DemBones UI")
        self.setMinimumWidth(420)
        # unique object name so we can find any existing instance
        self.setObjectName("DemBonesUI_MainWindow")
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Source / Target Row
        row = QtWidgets.QHBoxLayout()
        self.source_le = QtWidgets.QLineEdit()
        self.source_btn = QtWidgets.QPushButton("Select Source")
        self.source_btn.clicked.connect(self.select_source)
        row.addWidget(QtWidgets.QLabel("VertexAnimMesh:"))
        row.addWidget(self.source_le)
        row.addWidget(self.source_btn)
        layout.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        self.target_le = QtWidgets.QLineEdit()
        self.target_btn = QtWidgets.QPushButton("Select Target")
        self.target_btn.clicked.connect(self.select_target)
        row2.addWidget(QtWidgets.QLabel("SkinnedMesh:"))
        row2.addWidget(self.target_le)
        row2.addWidget(self.target_btn)
        layout.addLayout(row2)

        # Detected SkinCluster
        sc_row = QtWidgets.QHBoxLayout()
        self.skincluster_le = QtWidgets.QLineEdit()
        self.skincluster_le.setReadOnly(False)
        self.detect_sc_btn = QtWidgets.QPushButton("Detect from Source")
        self.detect_sc_btn.clicked.connect(self.detect_skincluster)
        sc_row.addWidget(QtWidgets.QLabel("SkinCluster:"))
        sc_row.addWidget(self.skincluster_le)
        sc_row.addWidget(self.detect_sc_btn)
        layout.addLayout(sc_row)

        # Parameters

        #--------------------- Time Range ---------------------#
        time_range_group = CollapsibleGroupBox("Basic Parameters", collapsed=False)
        time_range_layout = QtWidgets.QGridLayout()
        
        self.start_frame_sb = QtWidgets.QSpinBox()
        self.start_frame_sb.setRange(-100000, 100000)
        self.start_frame_sb.setValue(0)
        self.end_frame_sb = QtWidgets.QSpinBox()
        self.end_frame_sb.setRange(-100000, 100000)
        self.end_frame_sb.setValue(100)
        
        time_range_layout.addWidget(QtWidgets.QLabel("start_frame:"), 0, 0)
        time_range_layout.addWidget(self.start_frame_sb, 0, 1)
        time_range_layout.addWidget(QtWidgets.QLabel("end_frame:"), 1, 0)
        time_range_layout.addWidget(self.end_frame_sb, 1, 1)
        
        # button to set from timeline
        self.timeline_btn = QtWidgets.QPushButton("Use Timeline Range")
        self.timeline_btn.clicked.connect(self.use_timeline_range)
        time_range_layout.addWidget(self.timeline_btn, 2, 1)

        # Options and Run
        self.apply_weights_cb = QtWidgets.QCheckBox("Apply weights to SkinCluster")
        self.apply_weights_cb.setChecked(True)
        # Option to create animation keys for influences
        self.create_keys_cb = QtWidgets.QCheckBox("Create animation keys")
        self.create_keys_cb.setChecked(True)
        
        time_range_layout.addWidget(self.apply_weights_cb, 3, 0)
        time_range_layout.addWidget(self.create_keys_cb, 4, 0)
        
        time_range_group.setContentLayout(time_range_layout)
        layout.addWidget(time_range_group)

        #--------------------- Skin Parameters ---------------------#
        skin_params_group = CollapsibleGroupBox("DemBones Skin Parameters", collapsed=True)
        skin_params_layout = QtWidgets.QGridLayout()

        max_influences_label = QtWidgets.QLabel("max_influences:")
        self.max_influences_sb = QtWidgets.QSpinBox()
        self.max_influences_sb.setRange(1, 10)
        self.max_influences_sb.setValue(4)

        weights_smooth_label = QtWidgets.QLabel("weights_smooth:")
        self.weights_smooth_sb = QtWidgets.QDoubleSpinBox()
        self.weights_smooth_sb.setRange(0.0, 1.0)
        self.weights_smooth_sb.setDecimals(4)
        self.weights_smooth_sb.setValue(1e-4)

        weights_smooth_step_label = QtWidgets.QLabel("weights_smooth_step:")
        self.weights_smooth_step_sb = QtWidgets.QDoubleSpinBox()
        self.weights_smooth_step_sb.setRange(0.0, 10.0)
        self.weights_smooth_step_sb.setDecimals(2)
        self.weights_smooth_step_sb.setValue(1.0)
        
        
        skin_params_layout.addWidget(max_influences_label, 0, 0)
        skin_params_layout.addWidget(self.max_influences_sb, 0, 1)

        skin_params_layout.addWidget(weights_smooth_label, 1, 0)
        skin_params_layout.addWidget(self.weights_smooth_sb, 1, 1)
        skin_params_layout.addWidget(weights_smooth_step_label, 2, 0)
        skin_params_layout.addWidget(self.weights_smooth_step_sb, 2, 1)
        
        skin_params_group.setContentLayout(skin_params_layout)
        layout.addWidget(skin_params_group)

        #--------------------- DemBones Parameters ------------------#
        params_group = CollapsibleGroupBox("DemBones Parameters", collapsed=True)
        params_layout = QtWidgets.QGridLayout()

        self.num_iter_sb = QtWidgets.QSpinBox()
        self.num_iter_sb.setRange(1, 1000)
        self.num_iter_sb.setValue(5)

        self.num_transform_sb = QtWidgets.QSpinBox()
        self.num_transform_sb.setRange(0, 1000)
        self.num_transform_sb.setValue(3)

        self.num_weight_sb = QtWidgets.QSpinBox()
        self.num_weight_sb.setRange(0, 1000)
        self.num_weight_sb.setValue(3)


        self.affine_norm_sb = QtWidgets.QDoubleSpinBox()
        self.affine_norm_sb.setRange(1.0, 10.0)
        self.affine_norm_sb.setValue(4.0)

        self.affine_sb = QtWidgets.QDoubleSpinBox()
        self.affine_sb.setRange(0.0, 1000.0)
        self.affine_sb.setValue(10.0)
        
        params_layout.addWidget(QtWidgets.QLabel("num_iterations:"), 0, 0)
        params_layout.addWidget(self.num_iter_sb, 0, 1)
        params_layout.addWidget(QtWidgets.QLabel("num_transform_iterations:"), 1, 0)
        params_layout.addWidget(self.num_transform_sb, 1, 1)
        params_layout.addWidget(QtWidgets.QLabel("num_weight_iterations:"), 2, 0)
        params_layout.addWidget(self.num_weight_sb, 2, 1)
        params_layout.addWidget(QtWidgets.QLabel("translation_affine:"), 3, 0)
        params_layout.addWidget(self.affine_sb, 3, 1)
        params_layout.addWidget(QtWidgets.QLabel("translation_affine_norm:"), 4, 0)
        params_layout.addWidget(self.affine_norm_sb, 4, 1)
        
        params_group.setContentLayout(params_layout)
        layout.addWidget(params_group)

        bone_params_group = CollapsibleGroupBox("DemBones Init Parameters", collapsed=True)
        bone_params_layout = QtWidgets.QGridLayout()
        
        init_iterations_label = QtWidgets.QLabel("init_iterations:")
        init_iterations_label.setToolTip("Number of iterations during initialization phase.")
        self.init_iterations_sb = QtWidgets.QSpinBox()
        self.init_iterations_sb.setRange(1, 1000)
        self.init_iterations_sb.setValue(10)
        
        # bind_update: switched from checkbox to combo box supporting integer values (now 0/1/2)
        self.bind_combo = QtWidgets.QComboBox()
        # Add items: display text, userData = int value
        # Bind transformation update, 0=keep original, 1=set translations to p-norm centroids (using #transAffineNorm) and rotations to identity, 2=do 1 and group joints, default = 0
        self.bind_combo.addItem("0 - keep original", 0)
        self.bind_combo.setItemData(0, "keep original after compute.", QtCore.Qt.ToolTipRole)
        self.bind_combo.addItem("1 - update Translate&Rotate", 1)
        self.bind_combo.setItemData(1, "set translations to p-norm centroids (using #transAffineNorm) and rotations to identity.", QtCore.Qt.ToolTipRole)
        self.bind_combo.addItem("2 - do 1 and group joints", 2)
        self.bind_combo.setItemData(2, "set translations to p-norm centroids (using #transAffineNorm) and rotations to identity.Also group to cluster", QtCore.Qt.ToolTipRole)
        self.bind_combo.setCurrentIndex(0)
        self.bind_combo.setToolTip("Select bind update mode (returns int 0/1/2)")
        self.bind_combo.currentIndexChanged.connect(self._on_bind_combo_changed)


        self.add_joint_root = QtWidgets.QCheckBox()
        self.add_joint_root.setChecked(True)

        self.num_bone_sb = QtWidgets.QSpinBox()
        self.num_bone_sb.setRange(-1, 99999)
        self.num_bone_sb.setValue(100)

        root_label = QtWidgets.QLabel("auto_root:")
        root_label.setToolTip("If checked, a root joint will be created to parent all generated bones if needed.")
        root_label.setWordWrap(True)
        
        bone_params_layout.addWidget(QtWidgets.QLabel("bind_update:"), 0, 0)
        bone_params_layout.addWidget(self.bind_combo, 0, 1)

        bone_params_layout.addWidget(root_label, 1, 0)
        bone_params_layout.addWidget(self.add_joint_root, 1, 1)
        bone_params_layout.addWidget(QtWidgets.QLabel("num_bones:"), 2, 0)
        bone_params_layout.addWidget(self.num_bone_sb, 2, 1)

        bone_params_layout.addWidget(init_iterations_label, 3, 0)
        bone_params_layout.addWidget(self.init_iterations_sb, 3, 1)

        
        bone_params_group.setContentLayout(bone_params_layout)
        layout.addWidget(bone_params_group)


        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.clicked.connect(self.run)
        self.close_btn = QtWidgets.QPushButton("Close")
        # close the dialog (destroy) instead of hide
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def select_source(self):
        sel = cmds.ls(selection=True)
        if not sel:
            cmds.warning("Please select a source object first.")
            return
        src = sel[0]
        self.source_le.setText(src)


    def select_target(self):
        sel = cmds.ls(selection=True)
        if not sel:
            cmds.warning("Please select a target object first.")
            return
        tgt = sel[0]
        self.target_le.setText(tgt)
        # auto detect skinCluster
        self._auto_detect_skincluster_from(tgt)

    def detect_skincluster(self):
        tat = self.target_le.text().strip()
        if not cmds.objExists(tat):
            self.target_le.clear()
        
        if not tat:
            cmds.warning("Target is empty. Select a source or click Select Target first.")
            return
        self._auto_detect_skincluster_from(tat)

    def _auto_detect_skincluster_from(self, mesh_name: str):
        # If transform provided, try shape
        if not cmds.objExists(mesh_name):
            self.skincluster_le.clear()
        
        shapes = cmds.listRelatives(mesh_name, shapes=True, fullPath=True) or []
        candidates = []
        if shapes:
            for s in shapes:
                hist = cmds.listHistory(s) or []
                sc = [n for n in hist if cmds.nodeType(n) == 'skinCluster']
                if sc:
                    candidates.extend(sc)
        # also try history on transform
        hist2 = cmds.listHistory(mesh_name) or []
        sc2 = [n for n in hist2 if cmds.nodeType(n) == 'skinCluster']
        candidates.extend(sc2)

        candidates = list(dict.fromkeys(candidates))
        if candidates:
            self.skincluster_le.setText(candidates[0])
            cmds.inViewMessage(amg='Detected skinCluster: <hl>{}</hl>'.format(candidates[0]), pos='midCenter', fade=True, fadeInTime=0.1, fadeStayTime=1.0, fadeOutTime=0.2)
        else:
            self.skincluster_le.clear()
            cmds.inViewMessage(amg='No skinCluster found for <hl>{}</hl>'.format(mesh_name), pos='midCenter', fade=True, fadeInTime=0.1, fadeStayTime=1.0, fadeOutTime=0.2)

    def _on_bind_combo_changed(self, index: int):
        """Update the bind description label when the combo selection changes."""
        try:
            desc = self.bind_combo.itemData(index, QtCore.Qt.ToolTipRole) or ""
            cmds.inViewMessage(amg='Bind Update mode: <hl>{}</hl>'.format(desc), pos='midCenter', fade=True, fadeInTime=0.1, fadeStayTime=1.0, fadeOutTime=0.2)
        except Exception:
            # safe-guard: ignore UI update errors
            pass

    def _create_root_joint_if_needed(self, db):
        # type: (dem_bones.DemBones) -> None
        if self.add_joint_root.isChecked() == False:
            return
        
        if (db.bind_update == 2):
            cmds.warning("will not Create root joint, bind_update == 2.")
            return

        if not cmds.objExists("root"):
            dagMod = om2.MDagModifier()
            joint_root_node = dagMod.createNode("joint") #type: om2.MObject
            dagMod.doIt()
            om2.MFnDagNode(joint_root_node).setName("root")
            # NOTE: Add attaribute to lock bones, and show it in channel box
            numAttr = om2.MFnNumericAttribute()
            demLockAttr = numAttr.create("demLock", "demLock", om2.MFnNumericData.kBoolean, 1)
            numAttr.keyable = True
            fnNode = om2.MFnDependencyNode(joint_root_node)
            fnNode.addAttribute(demLockAttr)
        else:
            maya_temp_selection_list = om2.MSelectionList()
            maya_temp_selection_list.add("root")
            joint_root_node = maya_temp_selection_list.getDependNode(0)  #type: om2.MObject
        
        maya_selection_list = om2.MSelectionList()
        
        for jnt in db.influences:
            maya_selection_list.add(jnt)

        select_iter = om2.MItSelectionList(maya_selection_list)
        
        while not select_iter.isDone():
            jnt_obj = select_iter.getDependNode() # type: om2.MObject
            parent_obj = om2.MFnDagNode(jnt_obj).parent(0)# type: om2.MObject
            parent_name = om2.MFnDagNode(parent_obj).name() # type: str
            if parent_name == "world":
                dagMod.reparentNode(jnt_obj, joint_root_node)
            select_iter.next()
        dagMod.doIt()

    def _apply_weights_to_skincluster(self, db):
        # type: (dem_bones.DemBones) -> None
        tgt_name = self.target_le.text().strip()
        
        if not tgt_name:
            cmds.warning("No target specified to apply weights.")
            return
        
        if not cmds.objExists(tgt_name):
            self.skincluster_le.clear()
            cmds.warning("No target specified to apply weights.")
            return
        
        self._auto_detect_skincluster_from(tgt_name)
        
        # NOTE: Create skinCluster if not existing?
        # TODO: If we set a skinMesh as VertexAnimMesh, we should create a copy mesh for skinCluster application
        if not self.skincluster_le.text().strip():
            cmds.warning("No skinCluster specified to apply weights. Create one first for <%s>." % db.skin_mesh_shape_name)

            for influence in db.influences:
                matrix = om2.MMatrix(db.bind_matrix(influence))
                cmds.xform(influence, matrix=matrix, os=True) # set bind pose
            cmds.skinCluster(tgt_name, *db.influences, tsb=True, mi=db.max_influences) # type: list[str]

            self._auto_detect_skincluster_from(tgt_name)
        
        self._create_root_joint_if_needed(db)
        if int(self.bind_combo.currentData()) >= 1:
            cmds.warning("Need Update bindPose.")
        
        db.update_result_skin_weight(tgt_name)
        

    def closeEvent(self, event):
        # Allow destruction: clear module-level reference so a new dialog can be created later
        global _dlg
        try:
            _dlg = None
        except NameError:
            pass
        # Ensure the dialog is destroyed by Maya/PySide: accept the close event and schedule deletion
        try:
            event.accept()
        except Exception:
            pass
        try:
            # schedule object for deletion on the Qt event loop
            self.deleteLater()
        except Exception:
            # as a last resort, try to hide
            try:
                self.hide()
            except Exception:
                pass

    def use_timeline_range(self):
        try:
            min_t = cmds.playbackOptions(q=True, minTime=True)
            max_t = cmds.playbackOptions(q=True, maxTime=True)
            # set spinboxes to integer frame values
            self.start_frame_sb.setValue(int(round(min_t)))
            self.end_frame_sb.setValue(int(round(max_t)))
            cmds.inViewMessage(amg='Timeline range set: <hl>{}</hl> - <hl>{}</hl>'.format(int(min_t), int(max_t)), pos='midCenter', fade=True, fadeInTime=0.1, fadeStayTime=4.0, fadeOutTime=0.2)
        except Exception as e:
            cmds.warning('Failed to read timeline range: {}'.format(e))

    def run(self):
        src = self.source_le.text().strip()
        tgt = self.target_le.text().strip()
        if not src or not tgt:
            QtWidgets.QMessageBox.warning(self, 'Missing', 'Please set both Source and Target.')
            return

        db = dem_bones.DemBones()
        db.num_iterations = int(self.num_iter_sb.value())
        db.num_transform_iterations = int(self.num_transform_sb.value())
        db.num_weight_iterations = int(self.num_weight_sb.value())
        # bind_update is read from the combo's userData (int value)
        try:
            db.bind_update = int(self.bind_combo.currentData())
        except Exception:
            db.bind_update = 0
        db.num_bones = int(self.num_bone_sb.value())
        db.init_iterations = int(self.init_iterations_sb.value())
        db.max_influences = int(self.max_influences_sb.value())


        start_frame = int(self.start_frame_sb.value())
        end_frame = int(self.end_frame_sb.value())

        try:
            db.compute(tgt, src, start_frame=start_frame, end_frame=end_frame)
            cmds.inViewMessage(amg='DemBones compute finished for <hl>{}</hl> -> <hl>{}</hl>'.format(src, tgt), pos='midCenter', fade=True, fadeInTime=0.1, fadeStayTime=1.0, fadeOutTime=0.2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Compute Error', str(e))
            return

        # Optionally create animation keys for influences
        if self.create_keys_cb.isChecked():
            # Use db's read-only start/end if available, otherwise use local values
            try:
                for frame in range(db.start_frame, db.end_frame + 1):
                    for influence in db.influences:
                        matrix = om2.MTransformationMatrix(om2.MMatrix(db.anim_matrix(influence, frame)))
                        translate = matrix.translation(om2.MSpace.kWorld)
                        rotate = matrix.rotation().asVector()

                        cmds.setKeyframe("{}.translateX".format(influence), time=frame, value=translate.x)
                        cmds.setKeyframe("{}.translateY".format(influence), time=frame, value=translate.y)
                        cmds.setKeyframe("{}.translateZ".format(influence), time=frame, value=translate.z)
                        cmds.setKeyframe("{}.rotateX".format(influence), time=frame, value=math.degrees(rotate.x))
                        cmds.setKeyframe("{}.rotateY".format(influence), time=frame, value=math.degrees(rotate.y))
                        cmds.setKeyframe("{}.rotateZ".format(influence), time=frame, value=math.degrees(rotate.z))
                        
                cmds.inViewMessage(amg='Created animation keys for influences', pos='midCenter', fade=True, fadeInTime=0.1, fadeStayTime=1.0, fadeOutTime=0.2)
            except Exception as e:
                cmds.warning('Failed to create animation keys: {}'.format(e))

        # Optionally apply weights
        if self.apply_weights_cb.isChecked():
            self._apply_weights_to_skincluster(db)

        QtWidgets.QMessageBox.information(self, 'Done', 'DemBones run complete.')

        if db.rmse() > 1.0:
            QtWidgets.QMessageBox.information(self, 'OK', "High RMSE detected: {:.4f}".format(db.rmse()))


_dlg = None


def show():
    global _dlg
    app = QtWidgets.QApplication.instance()
    # look for any existing top-level widget with our objectName and reuse it
    if app is not None:
        for w in app.topLevelWidgets():
            try:
                if w.objectName() == 'DemBonesUI_MainWindow' and isinstance(w, DemBonesUI):
                    _dlg = w
                    _dlg.show()
                    _dlg.raise_()
                    _dlg.activateWindow()
                    return
            except Exception:
                # some widgets may not have objectName; ignore
                pass

    # fallback: create or show the module-level dialog
    if _dlg is None:
        _dlg = DemBonesUI()
    if not _dlg.isVisible():
        _dlg.show()
    _dlg.raise_()
    _dlg.activateWindow()


if __name__ == '__main__':
    show()
