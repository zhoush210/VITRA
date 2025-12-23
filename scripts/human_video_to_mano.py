"""
将HaWoR输出的MANO手部数据转换为XHand机器人格式。

输入：读取thirdparty/HaWoR/example/dual_hand_short中的mano表示
处理：将MANO格式的旋转矩阵转换为欧拉角，并构建human hand格式
输出：使用vitra/datasets/robot_dataset.py中的transfer_human_to_xhand，把human hand格式转换成xhand格式
"""

import json
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from pathlib import Path

# 导入转换函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# 直接从robot_dataset.py复制转换函数和映射表，避免依赖问题
XHAND_HUMAN_MAPPING = [
    (10, 8, 1),
    (11, 14, 1),
    (12, 17, 1),
    (13, 23, 1),
    (16, 26, 1),
    (17, 32, 1),
    (14, 35, 1),
    (15, 41, 1),
    (7, 43, -1),
    (6, 44, 1),
    (8, 46, -1),
    (9, 7, 1),
]

def transfer_human_to_xhand(human_action: torch.Tensor) -> torch.Tensor:
    """
    Transfer human hand model format back to XHand robot format.
    
    Args:
        human_action: Human action sequence, shape (T, 192)
    
    Returns:
        xhand_action: Robot action sequence, shape (T, 36) - 18 dims per hand
    """
    T = human_action.shape[0]
    # Initialize output tensor: 18 dims per hand (6 EEF + 12 joints)
    xhand_action = torch.zeros((T, 36), dtype=torch.float32)
    
    # Transfer left hand end-effector 6-DoF pose (translation + rotation)
    xhand_action[:, 0:6] = human_action[:, 0:6]
    
    # Transfer right hand end-effector 6-DoF pose
    xhand_action[:, 18:24] = human_action[:, 51:57]
    
    # Transfer joint angles using reverse mapping
    for src, dst, sign in XHAND_HUMAN_MAPPING:
        # Left hand: human -> xhand
        xhand_action[:, src] = sign * human_action[:, dst]
        # Right hand: human -> xhand
        xhand_action[:, src+18] = sign * human_action[:, dst+51]
    
    return xhand_action


def rotation_matrix_to_euler(rot_matrix):
    """
    将旋转矩阵转换为欧拉角（xyz顺序，弧度制）。
    
    Args:
        rot_matrix: 旋转矩阵，形状可以是 [3, 3], [T, 3, 3], 或 [T, 15, 3, 3]
    
    Returns:
        euler_angles: 欧拉角，形状对应输入的形状
    """
    original_shape = rot_matrix.shape
    
    if len(original_shape) == 2:  # [3, 3]
        euler = R.from_matrix(rot_matrix).as_euler('xyz', degrees=False)
    elif len(original_shape) == 3:  # [T, 3, 3]
        T = original_shape[0]
        rot_flat = rot_matrix.reshape(-1, 3, 3)
        euler = R.from_matrix(rot_flat).as_euler('xyz', degrees=False)
        euler = euler.reshape(T, 3)
    elif len(original_shape) == 4:  # [T, 15, 3, 3]
        T, num_joints = original_shape[0], original_shape[1]
        rot_flat = rot_matrix.reshape(-1, 3, 3)
        euler = R.from_matrix(rot_flat).as_euler('xyz', degrees=False)
        euler = euler.reshape(T, num_joints, 3)
    else:
        raise ValueError(f"Unsupported rotation matrix shape: {original_shape}")
    
    return euler


def load_mano_data(data_dir, hand_side):
    """
    加载HaWoR输出的MANO数据。
    
    Args:
        data_dir: 数据目录路径（如 thirdparty/HaWoR/example/dual_hand_short）
        hand_side: 'left' (0) 或 'right' (1)
    
    Returns:
        dict: 包含MANO参数的字典
    """
    hand_idx = 0 if hand_side == 'left' else 1
    json_path = Path(data_dir) / 'cam_space' / str(hand_idx)
    
    # 查找json文件
    json_files = list(json_path.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {json_path}")
    
    json_file = json_files[0]
    print(f"Loading {hand_side} hand data from: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 转换为numpy数组
    mano_data = {
        'init_root_orient': np.array(data['init_root_orient']),  # [1, T, 3, 3]
        'init_hand_pose': np.array(data['init_hand_pose']),      # [1, T, 15, 3, 3]
        'init_trans': np.array(data['init_trans']),              # [1, T, 3]
        'init_betas': np.array(data['init_betas']),              # [1, T, 10]
    }
    
    return mano_data


def mano_to_human_hand_format(mano_data_left, mano_data_right):
    """
    将MANO格式转换为human hand格式（欧拉角表示）。
    
    Human hand格式说明：
        - 每帧的维度：102（左右手各51维）
        - 每只手51维：
            * [0:3]   transl: 手腕位置 (x, y, z)
            * [3:6]   global_orient: 全局旋转（欧拉角xyz）
            * [6:51]  hand_pose: 15个关节 × 3欧拉角 = 45维
    
    Args:
        mano_data_left: 左手MANO数据
        mano_data_right: 右手MANO数据
    
    Returns:
        human_action: torch.Tensor, 形状 [T, 102]
    """
    # 获取时间步数
    T = mano_data_left['init_trans'].shape[1]
    
    # 初始化human hand格式的动作张量
    human_action = torch.zeros((T, 192), dtype=torch.float32)
    
    # 处理左手数据
    transl_left = mano_data_left['init_trans'][0]  # [T, 3]
    root_orient_left = mano_data_left['init_root_orient'][0]  # [T, 3, 3]
    hand_pose_left = mano_data_left['init_hand_pose'][0]  # [T, 15, 3, 3]
    
    # 转换为欧拉角
    root_orient_euler_left = rotation_matrix_to_euler(root_orient_left)  # [T, 3]
    hand_pose_euler_left = rotation_matrix_to_euler(hand_pose_left)  # [T, 15, 3]
    hand_pose_euler_left = hand_pose_euler_left.reshape(T, -1)  # [T, 45]
    
    # 填充左手数据到human_action
    human_action[:, 0:3] = torch.from_numpy(transl_left)
    human_action[:, 3:6] = torch.from_numpy(root_orient_euler_left)
    human_action[:, 6:51] = torch.from_numpy(hand_pose_euler_left)
    
    # 处理右手数据
    transl_right = mano_data_right['init_trans'][0]  # [T, 3]
    root_orient_right = mano_data_right['init_root_orient'][0]  # [T, 3, 3]
    hand_pose_right = mano_data_right['init_hand_pose'][0]  # [T, 15, 3, 3]
    
    # 转换为欧拉角
    root_orient_euler_right = rotation_matrix_to_euler(root_orient_right)  # [T, 3]
    hand_pose_euler_right = rotation_matrix_to_euler(hand_pose_right)  # [T, 15, 3]
    hand_pose_euler_right = hand_pose_euler_right.reshape(T, -1)  # [T, 45]
    
    # 填充右手数据到human_action
    human_action[:, 51:54] = torch.from_numpy(transl_right)
    human_action[:, 54:57] = torch.from_numpy(root_orient_euler_right)
    human_action[:, 57:102] = torch.from_numpy(hand_pose_euler_right)
    
    return human_action


def main():
    """主函数：将HaWoR的MANO数据转换为XHand格式"""
    
    # 数据路径
    data_dir = Path(__file__).parent.parent / 'thirdparty' / 'HaWoR' / 'example' / 'dual_hand_short'
    output_dir = Path(__file__).parent.parent / 'examples'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Converting MANO data to XHand format")
    print("=" * 60)
    
    # 1. 加载左右手的MANO数据
    print("\n[Step 1] Loading MANO data...")
    mano_data_left = load_mano_data(data_dir, 'left')
    mano_data_right = load_mano_data(data_dir, 'right')
    
    T_left = mano_data_left['init_trans'].shape[1]
    T_right = mano_data_right['init_trans'].shape[1]
    print(f"  Left hand: {T_left} frames")
    print(f"  Right hand: {T_right} frames")
    
    # 2. 转换为human hand格式
    print("\n[Step 2] Converting to human hand format...")
    human_action = mano_to_human_hand_format(mano_data_left, mano_data_right)
    print(f"  Human action shape: {human_action.shape}")  # [T, 192]
    
    # 3. 使用transfer_human_to_xhand转换为XHand格式
    print("\n[Step 3] Converting to XHand format...")
    xhand_action = transfer_human_to_xhand(human_action)
    print(f"  XHand action shape: {xhand_action.shape}")  # [T, 36]
    
    # 4. 保存结果
    output_file = output_dir / 'xhand_action.npy'
    np.save(output_file, xhand_action.numpy())
    print(f"\n[Step 4] Saved XHand action to: {output_file}")
    
    # 打印一些统计信息
    print("\n" + "=" * 60)
    print("Conversion Statistics:")
    print("=" * 60)
    print(f"Time steps: {xhand_action.shape[0]}")
    print(f"Action dimensions: {xhand_action.shape[1]} (18 per hand)")
    print(f"\nLeft hand action range:")
    print(f"  Translation: [{xhand_action[:, 0:3].min():.4f}, {xhand_action[:, 0:3].max():.4f}]")
    print(f"  Rotation: [{xhand_action[:, 3:6].min():.4f}, {xhand_action[:, 3:6].max():.4f}]")
    print(f"  Joints: [{xhand_action[:, 6:18].min():.4f}, {xhand_action[:, 6:18].max():.4f}]")
    print(f"\nRight hand action range:")
    print(f"  Translation: [{xhand_action[:, 18:21].min():.4f}, {xhand_action[:, 18:21].max():.4f}]")
    print(f"  Rotation: [{xhand_action[:, 21:24].min():.4f}, {xhand_action[:, 21:24].max():.4f}]")
    print(f"  Joints: [{xhand_action[:, 24:36].min():.4f}, {xhand_action[:, 24:36].max():.4f}]")
    print("=" * 60)


if __name__ == '__main__':
    main()