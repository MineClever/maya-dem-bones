# Stubs for pybind11 module _core (dem_bones)
# 提供 DemBones 类的 Python 类型提示和文档说明

from typing import List, Tuple

class DemBones:
    """
    DemBones 是一个基于 DemBones 算法的 Maya 绑定类，
    用于自动化骨骼动画分解、权重更新和关键帧应用。
    """

    def __init__(self) -> None:
        """
        初始化 DemBones 实例。
        默认参数包括迭代次数、容忍度等。
        """
        ...

    # ---------------- 属性 ----------------
    bind_update: int
    """绑定更新模式:
       0 = 保持原始绑定,
       1 = 平移设为质心，旋转设为单位矩阵,
       2 = 在模式 1 基础上分组关节。"""

    num_iterations: int
    """全局迭代次数，默认 30。"""

    num_transform_iterations: int
    """每次全局迭代中骨骼变换更新的迭代次数，默认 5。"""

    translation_affine: float
    """平移仿射软约束系数，默认 10.0。"""

    translation_affine_norm: float
    """平移仿射软约束的 p-范数，默认 4.0。"""

    num_weight_iterations: int
    """每次全局迭代中权重更新的迭代次数，默认 3。"""

    max_influences: int
    """每个顶点的最大非零权重数，默认 8。"""

    weights_smooth: float
    """权重平滑软约束系数，默认 1e-4。"""

    weights_smooth_step: float
    """权重平滑软约束的步长，默认 1.0。"""

    weights_epsilon: float
    """权重求解器的 epsilon，默认 1e-15。"""

    tolerance: float
    """收敛容忍度，默认 1e-3。"""

    patience: int
    """收敛等待迭代次数，默认 3。"""

    lock_weights_set: str
    """用于锁定权重的颜色集名称，默认 'demLock'。"""

    lock_bone_attr_name: str
    """用于锁定骨骼变换的属性名称，默认 'demLockBones'。"""

    start_frame: int
    """求解的起始帧。"""

    end_frame: int
    """求解的结束帧。"""

    influences: List[str]
    """所有骨骼影响的名字列表。"""

    weights: List[float]
    """所有顶点和骨骼的权重列表。"""

    # ---------------- 方法 ----------------
    def rmse(self) -> float:
        """
        计算重建误差的均方根 (Root Mean Squared Error)。
        返回值: float
        """
        ...

    def compute(self, source: str, target: str, start_frame: int, end_frame: int) -> None:
        """
        执行 DemBones 算法，分解权重和骨骼变换。

        参数:
            source: 源网格名称 (带 skinCluster 的 mesh)
            target: 目标动画网格名称
            start_frame: 起始帧
            end_frame: 结束帧
        """
        ...

    def apply_animation_and_weights(self, skin_mesh: str, b_update_joint_weight:bool = False) -> None:
        """
        将计算得到的动画和权重应用到指定的 skin mesh。

        功能:
            - 为骨骼插入关键帧 (translate/rotate)
            - 更新 skinCluster 权重
        """
        ...

    def bind_matrix(self, influence: str) -> Tuple[float, ...]:
        """
        获取指定骨骼的绑定矩阵。

        参数:
            influence: 骨骼名字
        返回值:
            16 个浮点数的矩阵 (列主序)
        """
        ...

    def anim_matrix(self, influence: str, frame: int) -> Tuple[float, ...]:
        """
        获取指定骨骼在某一帧的动画矩阵。

        参数:
            influence: 骨骼名字
            frame: 帧号
        返回值:
            16 个浮点数的矩阵 (列主序)
        """
        ...

    def update_result_skin_weight(self, skin_mesh: str) -> None:
        """
        更新指定 skin mesh 的权重。

        参数:
            skin_mesh: mesh 名称
        功能:
            调用 MFnSkinCluster::setWeights 更新权重。
        """
        ...
