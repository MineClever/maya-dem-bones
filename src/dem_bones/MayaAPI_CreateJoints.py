
# -*- coding: utf-8 -*-

from maya import OpenMaya
from maya import OpenMayaAnim
om = OpenMaya
omani = OpenMayaAnim
import maya.cmds as cmds


# -----------------------------
# OpenMaya (API 1.0) skinCluster helpers
# -----------------------------
def _get_skincluster_name(mesh):
    """使用 maya.cmds 查找网格关联的 skinCluster 名称（若有）。"""
    # 尝试取 shape
    shapes = cmds.listRelatives(mesh, shapes=True, fullPath=True) or []
    shape = shapes[0] if shapes else mesh
    sc = cmds.ls(cmds.listHistory(shape), type='skinCluster') or []
    return sc[0] if sc else None


def _get_mobject(node_name):
    sel = om.MSelectionList()
    try:
        sel.add(node_name)
    except Exception:
        return None
    mobj = om.MObject()
    try:
        sel.getDependNode(0, mobj)
    except Exception:
        return None
    return mobj


def _get_dagpath(node_name):
    sel = om.MSelectionList()
    try:
        sel.add(node_name)
    except Exception:
        return None
    dag = om.MDagPath()
    try:
        sel.getDagPath(0, dag)
    except Exception:
        return None
    return dag


def ensure_skincluster_api1(joint_names, mesh, skin_name=None):
    """确保网格有 skinCluster：若没有用 cmds 创建并返回 skinCluster 名称。
    返回 (skinName, skinMObject, skinMFn)
    """
    sc_name = _get_skincluster_name(mesh)
    if not sc_name:
        # 使用 cmds 创建最简单可靠
        sc = cmds.skinCluster(joint_names, mesh, toSelectedBones=True, normalizeWeights=1, name=skin_name or '')
        sc_name = sc[0] if isinstance(sc, (list, tuple)) else sc

    sc_mobj = _get_mobject(sc_name)
    if not sc_mobj or sc_mobj.isNull():
        om.MGlobal.displayWarning("Cannot get MObject for skinCluster: %s" % sc_name)
        return (sc_name, None, None)

    try:
        mfn = omani.MFnSkinCluster(sc_mobj)
    except Exception as e:
        om.MGlobal.displayWarning("Failed to create MFnSkinCluster: %s" % e)
        return (sc_name, sc_mobj, None)

    return (sc_name, sc_mobj, mfn)


def add_joints_to_skin_api1(joint_names, mesh):
    """把关节列表添加到网格的 skinCluster（若无则创建）。
    使用 cmds 做 addInfluence（简单且可靠），随后可以用 OpenMaya 设置权重。
    返回 skinCluster 名称。
    """
    if isinstance(joint_names, str):
        joint_names = [joint_names]

    sc_name, sc_mobj, sc_mfn = ensure_skincluster_api1(joint_names, mesh)
    if not sc_name:
        return None

    # 确保每个关节作为影响存在（cmds 更稳妥）
    existing = cmds.skinCluster(sc_name, query=True, influence=True) or []
    for j in joint_names:
        if j in existing:
            continue
        try:
            cmds.skinCluster(sc_name, edit=True, addInfluence=j, weight=0.0)
        except Exception as e:
            om.MGlobal.displayWarning("Failed to add influence %s: %s" % (j, e))

    return sc_name


def set_all_verts_to_joint_api1(skin_name, mesh, joint_name):
    """用 OpenMaya (API 1.0) 将网格上所有顶点的权重设置为某个关节的 1.0（其它影响为 0）。
    说明：这个函数假设关节已作为 skin 的影响存在。
    """
    # 获取 skin cluster MFn
    sc_mobj = _get_mobject(skin_name)
    if not sc_mobj:
        om.MGlobal.displayWarning("skinCluster not found: %s" % skin_name)
        return False

    try:
        sc_fn = omani.MFnSkinCluster(sc_mobj)
    except Exception as e:
        om.MGlobal.displayWarning("MFnSkinCluster failed: %s" % e)
        return False

    # 获取 mesh 的 dagPath（shape）
    shapes = cmds.listRelatives(mesh, shapes=True, fullPath=True) or []
    shape = shapes[0] if shapes else mesh
    dag = _get_dagpath(shape)
    if not dag:
        om.MGlobal.displayWarning("Cannot get dagPath for mesh: %s" % mesh)
        return False

    # 获取顶点数量
    try:
        mfnMesh = om.MFnMesh(dag)
        numVerts = mfnMesh.numVertices()
    except Exception as e:
        om.MGlobal.displayWarning("MFnMesh failed: %s" % e)
        return False

    # 创建顶点 component
    compFn = om.MFnSingleIndexedComponent()
    comp = compFn.create(om.MFn.kMeshVertComponent)
    intArray = om.MIntArray()
    for i in range(numVerts):
        intArray.append(i)
    compFn.addElements(intArray)

    # 找到 joint 在 influence 列表中的索引
    inf_objs = om.MObjectArray()
    try:
        sc_fn.influenceObjects(inf_objs)
    except Exception:
        # 某些 API 版本没有 influenceObjects；退回到查询 via cmds
        inf_names = cmds.skinCluster(skin_name, query=True, influence=True) or []
        try:
            inf_index = inf_names.index(joint_name)
        except ValueError:
            om.MGlobal.displayWarning("Joint %s is not an influence of %s" % (joint_name, skin_name))
            return False
        inf_indices = om.MIntArray()
        inf_indices.append(inf_index)
        # weights 数组为 numVerts * len(inf_indices) ; 这里只是单个影响
        weights = om.MDoubleArray()
        for _ in range(numVerts):
            weights.append(1.0)
        try:
            sc_fn.setWeights(dag, comp, inf_indices, weights, True)
        except Exception as e:
            om.MGlobal.displayWarning("setWeights failed (cmds-derived index): %s" % e)
            return False
        return True

    # 如果成功取得 MObjectArray，就寻找匹配的 MObject
    inf_index = -1
    try:
        sel = om.MSelectionList()
        sel.add(joint_name)
        joint_mobj = om.MObject()
        sel.getDependNode(0, joint_mobj)
    except Exception:
        joint_mobj = None

    for idx in range(inf_objs.length()):
        if joint_mobj and inf_objs[idx] == joint_mobj:
            inf_index = idx
            break

    if inf_index == -1:
        om.MGlobal.displayWarning("Joint %s not found in influence list of %s" % (joint_name, skin_name))
        return False

    inf_indices = om.MIntArray()
    inf_indices.append(inf_index)

    # 构造 weights 数组（numVerts * numInf）——单影响则为 numVerts
    weights = om.MDoubleArray()
    for _ in range(numVerts):
        weights.append(1.0)

    try:
        sc_fn.setWeights(dag, comp, inf_indices, weights, True)
    except Exception as e:
        om.MGlobal.displayWarning("setWeights failed: %s" % e)
        return False

    return True



def createJoints(names, radius):
    dagMod = om.MDagModifier()
    created = []

    for j in range(len(names)):
        print("Create joint <%s>" % names[j])

        jointObj = dagMod.createNode("joint", om.MObject())  # type: om.MObject
        created.append(jointObj)

    # 关键：先真正把节点加入到DAG
    dagMod.doIt()

    # 现在再设置每个joint的属性
    for j, jointObj in enumerate(created):
        fnJoint = omani.MFnIkJoint(jointObj)

        # 设置名称与初始位置（可选）
        fnJoint.setName(names[j])
        rootPos = om.MVector(0.0, 0.0, 0.0)
        fnJoint.setTranslation(rootPos, om.MSpace.kTransform)
        fnJoint.setRotationOrder(om.MTransformationMatrix.kXYZ, True)

        # 查找并设置半径
        
        radiusPlug = fnJoint.findPlug("radius", True)  # type: om.MPlug
        if not radiusPlug.isNull():
            radiusPlug.setFloat(radius)
        else:
            om.MGlobal.displayWarning("Joint %s has no 'radius' plug." % names[j])


if __name__ == "__main__":
    """小脚本示例：
    - 创建一组关节
    - 查找当前选择的网格（若无选择则回退到 'pSphere1'，否则提示退出）
    - 为网格创建/获取 skinCluster 并添加关节为影响
    - 将所有顶点的权重赋给第一个关节（作为示例，可按需注释）
    在 Maya 中把本文件作为模块导入或直接在 Script Editor 里运行此段。
    """


    joints = []
    for i in range(100):
        joints.append("joint_%d" % i)
    radius = 2.0

    print("Creating joints:", joints)
    createJoints(joints, radius)

    # 尝试从当前选择中找一个网格（transform）
    sel = cmds.ls(selection=True, long=True) or []
    target_mesh = None
    for s in sel:
        # 若选择是 transform，检查是否有 shape
        shapes = cmds.listRelatives(s, shapes=True, fullPath=True) or []
        if shapes:
            target_mesh = s
            break

    # 回退：如果没有选择，则尝试 pSphere1
    if not target_mesh:
        if cmds.objExists('pSphere1'):
            target_mesh = 'pSphere1'
            print("No selection found — falling back to 'pSphere1'")
        else:
            print("No mesh selected and 'pSphere1' not found. Select a mesh and rerun.")
            raise SystemExit(1)

    print("Target mesh:", target_mesh)

    # 添加关节到 skin（若无则创建 skinCluster）
    sc = add_joints_to_skin_api1(joints, target_mesh)
    if not sc:
        print("Failed to create or find a skinCluster on", target_mesh)
        raise SystemExit(1)

    print("SkinCluster:", sc)

    # 示例：把网格所有顶点的权重赋值给第一个关节（可按需禁用）
    try:
        ok = set_all_verts_to_joint_api1(sc, target_mesh, joints[0])
        if ok:
            print("Assigned all verts of", target_mesh, "to", joints[0])
        else:
            print("set_all_verts_to_joint_api1 returned False; check the warnings in Script Editor.")
    except Exception as e:
        print("Exception while setting weights:", e)
        print("You can set weights manually using cmds.skinPercent or the Weight Editor.")
