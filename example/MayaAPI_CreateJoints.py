
# -*- coding: utf-8 -*-

from maya import OpenMaya
from maya import OpenMayaAnim
om = OpenMaya
omani = OpenMayaAnim


def createJoints(names, radius):
    dagMod = om.MDagModifier()
    created = []

    for j in range(len(names)):
        print("Create joint <%s>" % names[j])

        jointObj = dagMod.createNode("joint", om.MObject())  # MObject
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
        radiusPlug = fnJoint.findPlug("radius", True)  # MPlug
        if not radiusPlug.isNull():
            radiusPlug.setFloat(radius)
        else:
            om.MGlobal.displayWarning("Joint %s has no 'radius' plug." % names[j])


if __name__ == "__main__":
    joints = ["joint_01", "joint_02", "joint_03"]
    createJoints(joints, 4.0)
