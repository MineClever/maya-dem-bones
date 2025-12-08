# Stubs for pybind11 module _core (dem_bones)
# 提供 DemBones 类的 Python 类型提示和文档说明

from typing import List, Tuple

class DemBones:
    """
    DemBones 是一个基于 DemBones 算法的 Maya 绑定类，
    用于自动化骨骼动画分解、权重更新和关键帧应用。
    
    主要功能包括：
    •	nIters — 全局迭代次数，默认 30
    作用：外层交替优化轮数。每轮先更新骨变换（computeTranformations），再更新权重（computeWeights）。
    建议：增大可提高拟合精度但线性增加总耗时；若结果已收敛可减小或用回调 cbIterEnd() 提前停止。
    •	nInitIters — 初始化阶段聚类细化迭代次数，默认 10
    作用：在无初始权重/变换时，LBG/VQ 聚类后对分割（split）得到的簇执行的局部细化次数（computeTransFromLabel() + computeLabel() + pruneBones()）。
    建议：增大会使初始骨和标签更稳定，耗时增加；通常保留默认即可。
    •	nTransIters — 每次全局迭代中更新骨变换的内部迭代次数，默认 5
    作用：在一次 computeTranformations() 调用中重复局部求解骨变换的次数（外层循环内的子迭代次数）。
    建议：若变换收敛慢可增大；设为 0 可跳过变换更新（仅更新权重）。
    •	transAffine — 平移亲和性（平移正则化）权重，默认 10.0
    作用：在计算 vuT 时对骨平移引入软约束，抑制因稀疏权重或数据噪声导致的不稳定大平移。数值越大对平移的“拉回”越强。
    建议：当骨平移估计出现异常（大幅抖动）时增大；若希望更自由的平移（更接近数据），减小该值。
    •	transAffineNorm — 参与平移亲和性时对权重的 p‑范数指数，默认 4.0
    作用：在 compute_vuT() 中用 pow(w, transAffineNorm) 对权重加权以计算平移先验（强调较大权重的贡献）。增大该值会让少数较大权重主导平移亲和项。
    建议：常用默认；对重心明显由少数骨主导的情况可以适当增大，以增强稀疏骨的平移约束。
    •	nWeightsIters — 每次全局迭代中更新权重的内部迭代次数，默认 3
    作用：在一次 computeWeights() 中重复权重优化的轮数（包括平滑、构建线性系统、稀疏求解）。
    建议：影响权重收敛速度与耗时；设为 0 可跳过权重更新（仅更新变换）。
    •	nnz — 每个顶点允许的非零骨数量，默认 8
    作用：稀疏约束：每个顶点在权重求解时最多保留 nnz 个非零分量（通过 wSolver.init(nnz) 与索引截断实现）。
    建议：较大 nnz 可提高表达能力但增加每顶点求解成本与内存；常见取 4–16，根据模型复杂度和性能权衡。
    •	weightsSmooth — 权重平滑的正则化强度，默认 1e-4
    作用：在构造每顶点的正规化线性系统时，用于惩罚与先验平滑权重（ws）的偏离（见 aTa 与 aTb 的构造）。较大值会让最终权重更靠近平滑先验。
    建议：增大可获得更平滑、稳定但可能欠拟合的权重；对噪声数据或纹理连续性要求高时适当增大。
    •	weightsSmoothStep — 用于构造 Laplacian 平滑矩阵的步长缩放，默认 1.0
    作用：在 computeSmoothSolver() 里将 Laplacian 乘以 weightsSmoothStep 并加上单位矩阵，控制隐式平滑强度与数值稳定性（影响 ws 计算）。
    建议：可用于整体放大/缩小 Laplacian 对权重平滑的影响；通常保持 1，遇到平滑不足或过强可微调。
    •	weightEps — 权重求解的数值阈值，默认 1e-15
    作用：多个地方用作小值阈值：例如在选择 nnzi（非零个数）时丢弃非常小的 ws 分量，在 compute_aTb() 中判断 ws(j,i) > weightEps 以跳过数值为 0 的项。用于数值稳定与稀疏判定。
    建议：保持非常小（默认即可）；若碰到数值不稳定或异常舍入，可适当放大到 1e-12 等以避免极小值影响判断。
    补充要点（交互与性能）：
    •	计算量结构：总耗时 ≈ nIters * (nTransIters * cost_update_transform + nWeightsIters * cost_update_weights)。nnz 和 nV、nB、nF 对成本影响很大。
    •	若目标是快速原型或交互调参：可以把 nIters、nTransIters、nWeightsIters 调小，或减小 nnz。若追求质量，先用较少迭代调参，再放大迭代次数精化。
    •	对低质量数据或希望更平滑的权重，增大 weightsSmooth / weightsSmoothStep；对需要更精确拟合的场景，减小这些正则项并适当增大迭代次数。
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

    num_bones: int
    """强制初始化的骨骼数量。如果目标的网格的未找到蒙皮骨骼,则使用此数量。默认 -1 (不强制)。"""

    init_iterations: int
    """初始化阶段的迭代次数，默认 10。"""

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
    """用于锁定骨骼变换的属性名称，默认 'demLock'。"""

    bone_name_prefix: str
    """Init创建骨骼时的骨骼名称前缀"""

    start_frame: int
    """求解的起始帧。"""

    end_frame: int
    """求解的结束帧。"""

    influences: List[str]
    """所有骨骼影响的名字列表。"""

    weights: List[float]
    """所有顶点和骨骼的权重列表。"""

    skin_mesh_shape_name: str
    """蒙皮网格的形状节点名称。"""

    vertex_anim_mesh_shape_name: str
    """动画网格的形状节点名称。"""

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
