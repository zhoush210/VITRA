import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import json
import sys
import cv2
from pathlib import Path

# 从 visualize_xhand_action.py 导入辅助函数
sys.path.insert(0, str(Path(__file__).parent))
from visualize_xhand_action import (
    xhand_to_hand_pose,
    transform_camera_to_robot_base,
    map_xhand_fingers_to_dex3
)

DT = 0.02
PUPPET_GRIPPER_JOINT_OPEN = 1.4

# 条件性导入 unitree 相关模块
try:
    sys.path.append(os.path.expanduser('~/VITRA/unitree_sdk2_python/example/g1/high_level'))
    sys.path.append(os.path.expanduser('~/VITRA/avp_teleoperate_il/teleop/robot_control'))
    import test_act   
    import robot_hand_unitree_dex3 as hand
    UNITREE_AVAILABLE = True
    print("Unitree SDK loaded successfully")
except ImportError as e:
    print(f"Warning: Unitree modules not available: {e}")
    UNITREE_AVAILABLE = False
    # 创建占位符
    class MockModule:
        def __init__(self):
            pass
        def __getattr__(self, name):
            return MockModule()
    test_act = MockModule()
    hand = MockModule()

import IPython
import time
from PIL import Image
from datetime import datetime
import select
import termios
import tty

e = IPython.embed

def get_key_nonblocking():
    """非阻塞地检查是否有按键输入"""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main(args):
    eval_bc_dual_arm(args)
    return

def eval_bc_dual_arm(args):
    if not UNITREE_AVAILABLE:
        print("Unitree SDK not available, cannot run robot evaluation")
        return
    
    # 加载 XHand 动作数据
    xhand_file = Path(args['xhand_file'])
    if not xhand_file.exists():
        raise FileNotFoundError(f"找不到文件: {xhand_file}")
    
    print(f"加载 XHand 动作数据: {xhand_file}")
    xhand_actions = np.load(xhand_file)
    print(f"数据形状: {xhand_actions.shape}")
    print(f"  时间步数: {xhand_actions.shape[0]}")
    print(f"  动作维度: {xhand_actions.shape[1]} (左手18维 + 右手18维)")
    
    # 截取指定范围的帧
    if args.get('start_frame') is not None or args.get('end_frame') is not None:
        start = args.get('start_frame', 0)
        end = args.get('end_frame', len(xhand_actions))
        xhand_actions = xhand_actions[start:end]
        print(f"使用帧范围: [{start}, {end}), 共 {len(xhand_actions)} 帧")
    
    max_timesteps = len(xhand_actions)
    print(f"总帧数: {max_timesteps}")
    
    # 需要G1机器人IK模块
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from g1.robot_arm_ik import G1_29_ArmIK
        print("G1 IK模块加载成功")
    except ImportError as e:
        print(f"错误: 无法导入 G1 IK模块: {e}")
        print("请确保 g1/robot_arm_ik.py 文件存在")
        return
    
    # 初始化IK求解器
    arm_ik = G1_29_ArmIK(faster=True)
    
    # === 从第一帧计算初始位姿 ===
    print("\n从第一帧数据计算初始位姿...")
    
    # 1. 提取第一帧的左右手姿态
    first_left_pose = xhand_to_hand_pose(xhand_actions[0], 'left')
    first_right_pose = xhand_to_hand_pose(xhand_actions[0], 'right')
    
    # 2. 坐标系转换（如果需要）
    if args.get('transform_to_robot', True):
        first_left_pose = transform_camera_to_robot_base(first_left_pose, 'left')
        first_right_pose = transform_camera_to_robot_base(first_right_pose, 'right')
    
    # 2.5. 应用高度偏移（与主循环中保持一致）
    height_offset = args.get('height_offset', 0.0)
    first_left_pose[2, 3] += height_offset
    first_right_pose[2, 3] += height_offset
    
    print(f"  左手初始位置（偏移后）: [{first_left_pose[0,3]:.3f}, {first_left_pose[1,3]:.3f}, {first_left_pose[2,3]:.3f}]")
    print(f"  右手初始位置（偏移后）: [{first_right_pose[0,3]:.3f}, {first_right_pose[1,3]:.3f}, {first_right_pose[2,3]:.3f}]")
    
    # 3. IK求解获取初始关节角度
    init_lr_arm_q = np.zeros(14)   # 14个关节角度（左臂7 + 右臂7）
    init_lr_arm_dq = np.zeros(14)  # 14个关节速度
    
    init_sol_q, init_sol_tauff = arm_ik.solve_ik(
        first_left_pose, first_right_pose,
        init_lr_arm_q, init_lr_arm_dq
    )
    
    init_left_arm = init_sol_q[0:7]   # 左臂关节角度
    init_right_arm = init_sol_q[7:14]  # 右臂关节角度
    
    print(f"  左臂初始关节角度: {init_left_arm}")
    print(f"  右臂初始关节角度: {init_right_arm}")
    
    # 4. 提取第一帧的手指姿态
    if args.get('use_finger_mapping', True):
        init_left_hand = map_xhand_fingers_to_dex3(xhand_actions[0], 'left')
        init_right_hand = map_xhand_fingers_to_dex3(xhand_actions[0], 'right')
        print(f"  左手初始关节角度: {init_left_hand}")
        print(f"  右手初始关节角度: {init_right_hand}")
    else:
        # 使用默认的手部初始姿态（张开）
        init_left_hand = np.zeros(7)
        init_right_hand = np.zeros(7)
    
    # 初始化机器人
    test_act.ChannelFactoryInitialize(0)
    custom = test_act.Custom()
    custom.Init()

    # 初始化手部控制
    left_state_arr = hand.Array('d', 7, lock=False)
    right_state_arr = hand.Array('d', 7, lock=False)
    action_arr = hand.Array('d', 14, lock=False)
    state_lock = hand.Lock()

    hand_ctrl = hand.Dex3_1_Controller(
        right_hand_state_array=right_state_arr,
        left_hand_state_array=left_state_arr,
        dual_hand_action_array=action_arr,
        fps=100.0,
        Unit_Test=False
    )
    
    # 设置初始位姿
    custom.init_armpos(init_left_arm, init_right_arm)
    hand_ctrl.ctrl_dual_hand(init_left_hand, init_right_hand)
    time.sleep(0.5)
    print("初始位姿设置完成\n")

    with torch.inference_mode():
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            print("按回车键开始推理，按 's' 键结束推理")
            input()
            
            # 使用第一帧计算出的关节状态初始化
            current_lr_arm_q = init_sol_q.copy()  # 使用第一帧的关节角度
            current_lr_arm_dq = np.zeros(14)      # 初始速度为0
            
            for t in range(max_timesteps):
                print(f"\n=== 帧 {t}/{max_timesteps} ===")
                key = get_key_nonblocking()
                if key == 's':
                    print("检测到按键 's'，退出循环")
                    break
                
                start_time = time.time()

                # 从XHand数据获取当前帧的动作
                # 1. 提取左右手姿态
                left_pose = xhand_to_hand_pose(xhand_actions[t], 'left')
                right_pose = xhand_to_hand_pose(xhand_actions[t], 'right')
                
                # 2. 坐标系转换（如果需要）
                if args.get('transform_to_robot', True):
                    left_pose = transform_camera_to_robot_base(left_pose, 'left')
                    right_pose = transform_camera_to_robot_base(right_pose, 'right')

                # 应用高度偏移
                left_pose[2, 3] += height_offset
                right_pose[2, 3] += height_offset
                
                # 打印调试信息
                if t == 0 or t % 30 == 0:
                    print(f"  左手位置: [{left_pose[0,3]:.3f}, {left_pose[1,3]:.3f}, {left_pose[2,3]:.3f}]")
                    print(f"  右手位置: [{right_pose[0,3]:.3f}, {right_pose[1,3]:.3f}, {right_pose[2,3]:.3f}]")
                
                # 3. IK求解获取关节角度
                sol_q, sol_tauff = arm_ik.solve_ik(
                    left_pose, right_pose, 
                    current_lr_arm_q, current_lr_arm_dq
                )
                
                # 4. 更新关节状态
                if t > 0:
                    current_lr_arm_dq = (sol_q - current_lr_arm_q) * args.get('fps', 20.0)
                current_lr_arm_q = sol_q.copy()
                
                # 5. 提取手指动作（如果使用手指映射）
                if args.get('use_finger_mapping', True):
                    left_hand_action = map_xhand_fingers_to_dex3(xhand_actions[t], 'left')
                    right_hand_action = map_xhand_fingers_to_dex3(xhand_actions[t], 'right')
                    
                    if t == 0:
                        print(f"\n手指映射:")
                        print(f"  左手dex3关节: {left_hand_action}")
                        print(f"  右手dex3关节: {right_hand_action}")
                else:
                    # 使用固定的gripper值
                    left_hand_action = init_left_hand
                    right_hand_action = init_right_hand
                
                # 动作平滑 - 双臂版本
                if t == 0:
                    print(f"\n初始左臂关节: {sol_q[0:7]}")
                    print(f"初始右臂关节: {sol_q[7:14]}")
                    prev_left_arm = sol_q[0:7]
                    prev_right_arm = sol_q[7:14]
                    prev_left_hand = left_hand_action
                    prev_right_hand = right_hand_action
                    smoothed_left_arm = prev_left_arm
                    smoothed_right_arm = prev_right_arm
                    smoothed_left_hand = prev_left_hand
                    smoothed_right_hand = prev_right_hand
                else:
                    alpha = args.get('smoothing_alpha', 0.0)  # 平滑系数
                    smoothed_left_arm = alpha * prev_left_arm + (1-alpha) * sol_q[0:7]
                    smoothed_right_arm = alpha * prev_right_arm + (1-alpha) * sol_q[7:14]
                    smoothed_left_hand = alpha * prev_left_hand + (1-alpha) * left_hand_action
                    smoothed_right_hand = alpha * prev_right_hand + (1-alpha) * right_hand_action
                    prev_left_arm = smoothed_left_arm
                    prev_right_arm = smoothed_right_arm
                    prev_left_hand = smoothed_left_hand
                    prev_right_hand = smoothed_right_hand

                # 控制双臂机器人
                # target_qpos 格式: [左臂7, 右臂7, 腰部3]
                target_qpos = np.concatenate([smoothed_left_arm, smoothed_right_arm, np.zeros(3)], axis=0)
                custom.set_arm_pose(target_qpos, enable_sdk=True)
                
                # 控制双手
                hand_ctrl.ctrl_dual_hand(smoothed_left_hand, smoothed_right_hand)

                # 控制频率
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(20)) - time_elapsed)
                time.sleep(sleep_time)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    print("\n执行完成!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="在真机上replay XHand轨迹"
    )
    
    # 输入文件
    parser.add_argument(
        '--xhand_file', 
        type=str, 
        default='examples/xhand_action.npy',
        help='XHand 动作文件路径 (.npy 格式)'
    )
    
    # 帧范围
    parser.add_argument(
        '--start_frame', 
        type=int, 
        default=None,
        help='起始帧索引（默认: 0）'
    )
    parser.add_argument(
        '--end_frame', 
        type=int, 
        default=None,
        help='结束帧索引（默认: 最后一帧）'
    )
    
    # 控制参数
    parser.add_argument(
        '--fps', 
        type=float, 
        default=20.0,
        help='控制频率（默认: 20 Hz）'
    )
    parser.add_argument(
        '--transform_to_robot', 
        action='store_true',
        default=True,
        help='将相机坐标系转换为机器人基座坐标系（默认开启）'
    )
    parser.add_argument(
        '--use_finger_mapping',
        action='store_true',
        default=True,
        help='使用XHand手指关节数据映射到dex3灵巧手（默认开启）'
    )
    parser.add_argument(
        '--smoothing_alpha',
        type=float,
        default=0.3,
        help='动作平滑系数 (0-1, 0表示无平滑)'
    )
    parser.add_argument(
        '--height_offset',
        type=float,
        default=0.0,
        help='末端执行器高度偏移（米），正值表示抬高（默认: 0.0）'
    )
    
    args = parser.parse_args()
    main(vars(args))
