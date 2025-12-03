import math
from maya import cmds
from maya.api import OpenMaya, OpenMayaAnim

import dem_bones

def ProcessFromSelection(b_update_skin_weight = False, start_frame=0, end_frame=100):
    sel = OpenMaya.MGlobal.getActiveSelectionList() # type: OpenMaya.MSelectionList
    
    if not sel.length() > 1: return
    last_selected = sel.length() - 1 # type: int
    
    sk_mesh_obj = sel.getDependNode(0)
    skin_cluster_fn = OpenMayaAnim.MFnSkinCluster(sk_mesh_obj)

    sk_mesh_dag = sel.getDagPath(0)  # type: OpenMaya.MDagPath
    sk_mesh_dag.extendToShape()

    vtx_anim_mesh_dag = sel.getDagPath(last_selected) # type: OpenMaya.MDagPath
    vtx_anim_mesh_dag.extendToShape()
    
    # NOTE: Pre-Check if same count vertex from reference.
    sk_mesh_fn = OpenMaya.MFnMesh(sk_mesh_dag)
    vtx_anim_mesh_fn = OpenMaya.MFnMesh(vtx_anim_mesh_dag)
    b_same_vtx_count = sk_mesh_fn.numVertices == vtx_anim_mesh_fn.numVertices
    if not b_same_vtx_count:
        OpenMaya.MGlobal.displayWarning("Not same count vertex from <%s> between <%s>" % (sk_mesh_dag.partialPathName(), vtx_anim_mesh_dag.partialPathName()))
        return
    
    db = dem_bones.DemBones()
    db.compute(sk_mesh_dag.partialPathName(), vtx_anim_mesh_dag.partialPathName(), start_frame=start_frame, end_frame=1)

    # TODO: Move to C++ API
    for frame in range(db.start_frame, db.end_frame + 1):
        for influence in db.influences:
            matrix = OpenMaya.MMatrix(db.anim_matrix(influence, frame))
            matrix = OpenMaya.MTransformationMatrix(matrix)
            translate = matrix.translation(OpenMaya.MSpace.kWorld)
            rotate = matrix.rotation().asVector()

            cmds.setKeyframe("{}.translateX".format(influence), time=frame, value=translate.x)
            cmds.setKeyframe("{}.translateY".format(influence), time=frame, value=translate.y)
            cmds.setKeyframe("{}.translateZ".format(influence), time=frame, value=translate.z)
            cmds.setKeyframe("{}.rotateX".format(influence), time=frame, value=math.degrees(rotate.x))
            cmds.setKeyframe("{}.rotateY".format(influence), time=frame, value=math.degrees(rotate.y))
            cmds.setKeyframe("{}.rotateZ".format(influence), time=frame, value=math.degrees(rotate.z))
    
    if b_update_skin_weight:
        skin_cluster_fn.setWeights(
            sk_mesh_dag,
            OpenMaya.MObject(),
            OpenMaya.MIntArray(range(len(db.influences))),
            OpenMaya.MDoubleArray(db.weights)
        )
        # C++ method
        # db.update_result_skin_weight(sk_mesh_dag.partialPathName())
if __name__ == "__main__":
    ProcessFromSelection()
