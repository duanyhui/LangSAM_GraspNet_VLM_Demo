""" 演示程序：展示抓取预测结果
    功能说明：
    - 该演示程序用于展示GraspNet基线模型的抓取预测结果
    - 加载预训练模型，处理输入的RGB-D图像数据
    - 生成抓取候选，进行碰撞检测，并可视化最优抓取姿态
    作者: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
# 模型检查点路径参数，必须提供预训练模型的路径
parser.add_argument('--checkpoint_path', required=True, help='模型检查点文件路径')
# 点云中点的数量，默认20000个点，用于网络输入
parser.add_argument('--num_point', type=int, default=20000, help='点云点数 [默认: 20000]')
# 视角数量，默认300个视角，用于多视角抓取检测
parser.add_argument('--num_view', type=int, default=300, help='视角数量 [默认: 300]')
# 碰撞检测阈值，用于过滤与物体发生碰撞的抓取姿态
parser.add_argument('--collision_thresh', type=float, default=0.01, help='碰撞检测中的碰撞阈值 [默认: 0.01]')
# 体素大小，在碰撞检测前对点云进行体素化处理的网格尺寸
parser.add_argument('--voxel_size', type=float, default=0.01, help='碰撞检测前处理点云的体素大小 [默认: 0.01]')
cfgs = parser.parse_args()


def get_net():
    """
    初始化并加载GraspNet神经网络模型
    
    功能说明：
    - 创建GraspNet模型实例，配置网络参数
    - 加载预训练的模型权重
    - 设置模型为评估模式，准备进行推理
    
    返回值：
    - net: 已加载权重的GraspNet模型，处于eval模式
    """
    # 初始化模型
    # input_feature_dim=0: 不使用额外的特征维度
    # num_view: 视角数量，影响抓取姿态的采样密度
    # num_angle=12: 抓取角度的离散化数量（每30度一个角度）
    # num_depth=4: 抓取深度的离散化层次
    # cylinder_radius=0.05: 圆柱形抓取区域的半径（5cm）
    # hmin=-0.02: 最小抓取高度（-2cm，表示可以稍微向下抓取）
    # hmax_list: 不同层次的最大抓取高度列表
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    # 设置计算设备（优先使用GPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # 加载预训练权重
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> 已加载检查点 %s (轮次: %d)"%(cfgs.checkpoint_path, start_epoch))
    # 设置模型为评估模式（关闭dropout和batch normalization的训练行为）
    net.eval()
    return net

def get_and_process_data(data_dir):
    """
    加载和预处理RGB-D数据
    
    功能说明：
    - 从指定目录加载彩色图像、深度图像、工作区域掩码和相机内参
    - 生成3D点云并进行预处理
    - 对点云进行采样以匹配网络输入要求
    
    参数：
    - data_dir: 包含输入数据的目录路径
    
    返回值：
    - end_points: 包含网络输入数据的字典
    - cloud: Open3D格式的完整点云（用于可视化）
    """
    # 加载输入数据
    # 彩色图像：RGB值归一化到[0,1]范围
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    # 深度图像：单位通常为毫米
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    # 工作区域掩码：标记有效的抓取区域
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    # 元数据：包含相机内参和深度缩放因子
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']  # 3x3相机内参矩阵
    factor_depth = meta['factor_depth']   # 深度值缩放因子

    # 生成点云
    # 创建相机信息对象，包含图像尺寸、焦距、主点坐标等
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    # 从深度图生成有组织的点云（保持像素对应关系）
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # 获取有效点
    # 结合工作区域掩码和深度有效性，筛选出有效的3D点
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]      # 有效的3D点坐标
    color_masked = color[mask]      # 对应的颜色信息

    # 点云采样
    # 如果点数足够，随机采样指定数量的点
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        # 如果点数不足，先取全部点，然后重复采样补足
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # 转换数据格式
    # 创建Open3D点云对象用于可视化
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    
    # 准备网络输入数据
    end_points = dict()
    # 转换为PyTorch张量并添加batch维度
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    """
    使用神经网络生成抓取候选
    
    功能说明：
    - 通过GraspNet模型进行前向推理
    - 解码网络输出为具体的抓取姿态参数
    - 返回抓取组对象，包含所有候选抓取的位置、方向和质量分数
    
    参数：
    - net: 已训练好的GraspNet模型
    - end_points: 包含输入点云的数据字典
    
    返回值：
    - gg: GraspGroup对象，包含所有预测的抓取姿态
    """
    # 前向传播（推理模式）
    with torch.no_grad():
        # 网络前向传播，预测抓取相关特征
        end_points = net(end_points)
        # 解码网络输出为抓取姿态参数
        # 包括抓取位置、旋转矩阵、开合宽度、质量分数等
        grasp_preds = pred_decode(end_points)
    
    # 转换为numpy数组并移到CPU
    gg_array = grasp_preds[0].detach().cpu().numpy()
    # 创建抓取组对象，方便后续处理和可视化
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    """
    碰撞检测：过滤与环境发生碰撞的抓取姿态
    
    功能说明：
    - 使用无模型碰撞检测器检测抓取器与物体的潜在碰撞
    - 过滤掉会导致碰撞的抓取姿态，提高抓取成功率
    - 基于体素化的快速碰撞检测算法
    
    参数：
    - gg: 输入的抓取组对象
    - cloud: 场景的点云数据
    
    返回值：
    - gg: 过滤掉碰撞抓取后的抓取组对象
    """
    # 创建无模型碰撞检测器
    # voxel_size: 体素化的网格大小，影响检测精度和速度
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    
    # 执行碰撞检测
    # approach_dist=0.05: 抓取器接近物体时的安全距离（5cm）
    # collision_thresh: 碰撞检测的敏感度阈值
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    
    # 过滤掉发生碰撞的抓取（保留collision_mask为False的抓取）
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    """
    可视化抓取结果
    
    功能说明：
    - 对抓取进行非最大值抑制，去除重复的抓取姿态
    - 按抓取质量分数排序，选择最优的抓取姿态
    - 使用Open3D进行3D可视化展示
    
    参数：
    - gg: 抓取组对象
    - cloud: 场景点云，用于背景显示
    """
    # 非最大值抑制：去除相近位置的重复抓取，避免聚集效应
    gg.nms()
    
    # 按抓取质量分数降序排序，优先显示高质量抓取
    gg.sort_by_score()
    
    # 选择前50个最优抓取进行可视化
    gg = gg[:50]
    
    # 将抓取姿态转换为Open3D的几何对象（抓取器模型）
    grippers = gg.to_open3d_geometry_list()
    
    # 使用Open3D可视化工具同时显示点云和抓取器
    # 用户可以交互式地查看抓取结果
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    """
    主演示函数：完整的抓取检测流程
    
    功能说明：
    - 整合所有步骤，从数据加载到结果可视化
    - 展示完整的抓取检测pipeline
    
    参数：
    - data_dir: 输入数据目录路径
    
    处理流程：
    1. 加载预训练的GraspNet模型
    2. 读取并预处理RGB-D数据
    3. 使用神经网络生成抓取候选
    4. 可选：进行碰撞检测过滤
    5. 可视化最终的抓取结果
    """
    # 步骤1: 初始化并加载神经网络模型
    net = get_net()
    
    # 步骤2: 加载和预处理输入数据
    end_points, cloud = get_and_process_data(data_dir)
    
    # 步骤3: 神经网络推理，生成抓取候选
    gg = get_grasps(net, end_points)
    
    # 步骤4: 可选的碰撞检测（如果阈值大于0）
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    print(gg)
    # 步骤5: 可视化最终结果
    vis_grasps(gg, cloud)

if __name__=='__main__':
    # 默认使用示例数据进行演示
    # 该目录包含一个完整的测试场景：彩色图、深度图、工作区域掩码和相机参数
    data_dir = 'doc/example_data'
    demo(data_dir)
