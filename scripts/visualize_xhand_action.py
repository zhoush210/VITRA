"""
可视化从 HaWoR MANO 数据转换得到的 XHand 机器人动作序列。

输入：examples/xhand_action.npy (T, 36) - XHand格式的双手动作序列
输出：3D 可视化显示机器人双手的运动轨迹

# 简化可视化（推荐先试这个）
python scripts/visualize_xhand_action.py --simple_vis

# 使用 G1 机器人模型可视化（需要 G1 模块）
python scripts/visualize_xhand_action.py

# 指定特定帧范围
python scripts/visualize_xhand_action.py --simple_vis --start_frame 0 --end_frame 50

# 调整播放速度
python scripts/visualize_xhand_action.py --fps 15

"""

import argparse
import numpy as np
import torch
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# 检查是否可以导入 G1 机器人可视化模块
try:
    from g1.robot_arm_ik import G1_29_ArmIK
    from g1.visualize_arm_episodes import G1_29_Vis_Episode, show_current_q
    from g1.map_gripper_to_hand import map_gripper_to_hand
    HAS_G1_MODULES = True
except ImportError:
    print("警告: 无法导入 G1 机器人模块，将使用简化的可视化")
    HAS_G1_MODULES = False


def map_xhand_fingers_to_dex3(xhand_state, hand_side='left'):
    """
    将XHand手指关节映射到G1的dex3三指手
    
    Dex3手有7个自由度：
        0: thumb_0_joint (大拇指根部旋转 - 未使用，设为0)
        1: thumb_1_joint (大拇指第一关节 - 对应PIP)
        2: thumb_2_joint (大拇指第二关节 - 对应DIP)
        3: middle_0_joint (中指第一关节 - 对应MCP)
        4: middle_1_joint (中指第二关节 - 对应PIP/DIP)
        5: index_0_joint (食指第一关节 - 对应MCP)
        6: index_1_joint (食指第二关节 - 对应PIP)

        open 的状态定义为：
        left hand: [0, 0, 0, 0, 0, 0, 0, 0]
        right hand: [0, 0, 0, 0, 0, 0, 0, 0]

        close 的状态定义为：
        left hand: [0, 0.2, 1.0, -1.0, -1.0, -1.0, -1.0]
        right hand: [0, -0.2, -1.0, 1.0, 1.0, 1.0, 1.0]
    
    XHand手指关节映射（每只手12个关节，索引6-17）：
        大拇指: XHand[11]=thumb_pip_z, XHand[12]=thumb_dip_z
        食指:   XHand[13]=index_mcp_z, XHand[16]=index_pip_z
        中指:   XHand[14]=middle_mcp_z, XHand[15]=middle_dip_z
    
    Args:
        xhand_state: XHand状态 [18] 或 [36] (双手)
        hand_side: 'left' 或 'right'
    
    Returns:
        dex3_joints: numpy array [7], dex3手的关节角度
    """
    if len(xhand_state.shape) == 1:
        # 单帧数据
        if xhand_state.shape[0] == 36:  # 双手数据
            if hand_side == 'left':
                finger_joints = xhand_state[6:18]  # 左手手指关节
            else:
                finger_joints = xhand_state[24:36]  # 右手手指关节
        else:  # 单手数据 [18]
            finger_joints = xhand_state[6:18]
    else:
        # 多帧数据，只处理当前帧
        raise ValueError("请传入单帧数据")
    
    # 初始化dex3关节角度
    dex3_joints = np.zeros(7, dtype=np.float32)

    # 归一化因子：将XHand关节角度放大到dex3的合理范围
    # XHand关节可能范围较小，需要放大以匹配dex3的-1到1范围
    scale = 1.0  # 可以根据实际数据调整

    # 映射 XHand 手指关节到 dex3
    if hand_side == 'left':
        # 左手：弯曲时关节角度为负值
        dex3_joints[1] = -finger_joints[5] * scale  # thumb_1_joint <- thumb_pip_z (XHand[11])
        dex3_joints[2] = -finger_joints[6] * scale  # thumb_2_joint <- thumb_dip_z (XHand[12])
        dex3_joints[3] = finger_joints[8] * scale  # middle_0_joint <- middle_mcp_z (XHand[14])
        dex3_joints[4] = finger_joints[9] * scale  # middle_1_joint <- middle_dip_z (XHand[15])
        dex3_joints[5] = finger_joints[7] * scale  # index_0_joint <- index_mcp_z (XHand[13])
        dex3_joints[6] = finger_joints[10] * scale # index_1_joint <- index_pip_z (XHand[16])
    else:  # right
        # 右手：弯曲时关节角度为正值
        dex3_joints[1] = -finger_joints[5] * scale   # thumb_1_joint <- thumb_pip_z (XHand[11])
        dex3_joints[2] = -finger_joints[6] * scale   # thumb_2_joint <- thumb_dip_z (XHand[12])
        dex3_joints[3] = finger_joints[8] * scale   # middle_0_joint <- middle_mcp_z (XHand[14])
        dex3_joints[4] = finger_joints[9] * scale   # middle_1_joint <- middle_dip_z (XHand[15])
        dex3_joints[5] = finger_joints[7] * scale   # index_0_joint <- index_mcp_z (XHand[13])
        dex3_joints[6] = finger_joints[10] * scale  # index_1_joint <- index_pip_z (XHand[16])
    # thumb_0_joint 设为0（未使用）
    dex3_joints[0] = 0.0
    
    return dex3_joints

def xhand_to_hand_pose(xhand_state, hand_side='left'):
    """
    将 XHand 格式的状态转换为 4x4 变换矩阵。
    
    XHand 格式 (每只手18维):
        [0:3]   translation (x, y, z)
        [3:6]   rotation (euler angles in radians, xyz order)
        [6:18]  joint angles (12 joints)
    
    Args:
        xhand_state: XHand 状态向量 [36,] 或动作序列 [T, 36]
        hand_side: 'left' 或 'right'
    
    Returns:
        4x4 变换矩阵或 [T, 4, 4] 变换矩阵序列
    """
    if xhand_state.ndim == 1:
        # 单帧处理
        if hand_side == 'left':
            trans = xhand_state[0:3]
            rot_euler = xhand_state[3:6]
        else:  # right
            trans = xhand_state[18:21]
            rot_euler = xhand_state[21:24]
        
        # 构建 4x4 变换矩阵
        pose = np.eye(4)
        pose[:3, :3] = R.from_euler('xyz', rot_euler).as_matrix()
        pose[:3, 3] = trans
        
        return pose
    else:
        # 多帧处理
        T = xhand_state.shape[0]
        poses = np.zeros((T, 4, 4))
        for t in range(T):
            poses[t] = xhand_to_hand_pose(xhand_state[t], hand_side)
        return poses


def transform_camera_to_robot_base(hand_pose, hand_side='left', apply_transform=True):
    """
    将相机坐标系中的手部姿态转换到机器人基座坐标系。
    
    相机坐标系: X右, Y下, Z前
    机器人坐标系: X前, Y左, Z上
    
    Args:
        hand_pose: 4x4 变换矩阵或 [T, 4, 4]
        hand_side: 'left' 或 'right'，用于处理左右手的镜像关系
        apply_transform: 是否应用坐标系转换
    
    Returns:
        转换后的变换矩阵
    """
    if not apply_transform:
        return hand_pose.copy()
    
    if hand_pose.ndim == 2:
        # 单帧
        transformed = hand_pose.copy()
        
        # 坐标系转换矩阵: 相机 → 机器人
        # 相机坐标系: X右, Y下, Z前
        # 机器人坐标系: X前, Y左, Z上
        # 转换: X_robot = Z_cam, Y_robot = -X_cam, Z_robot = -Y_cam
        
        R_cam_to_robot = np.array([
            [ 0,  0,  1],  # X_robot = Z_cam
            [-1,  0,  0],  # Y_robot = -X_cam
            [ 0, -1,  0]   # Z_robot = -Y_cam
        ])
        
        # 应用旋转变换到位置
        transformed[:3, 3] = R_cam_to_robot @ transformed[:3, 3]
        
        # 应用旋转变换到姿态
        transformed[:3, :3] = R_cam_to_robot @ transformed[:3, :3]
        
        # 右手需要额外的180度旋转（绕Z轴），因为在人手数据中右手朝向相反
        if hand_side == 'right':
            # 绕Z轴旋转180度
            R_flip = np.array([
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0,  1]
            ])
            transformed[:3, :3] = transformed[:3, :3] @ R_flip
        
        # 高度偏移 (相机在头部，需要向上平移)
        transformed[2, 3] += 0.25  # 假设相机距离机器人基座 0.25m
        
        return transformed
    else:
        # 多帧
        T = hand_pose.shape[0]
        transformed = np.zeros_like(hand_pose)
        for t in range(T):
            transformed[t] = transform_camera_to_robot_base(hand_pose[t], hand_side, True)
        return transformed


def visualize_with_g1(xhand_actions, args):
    """使用 G1 机器人模型进行可视化"""
    print("使用 G1 机器人模型进行可视化...")
    
    g1_urdf_path = "g1/assets/g1_body29_hand14.urdf"
    hand_type = "dex3"
    
    vis_model = G1_29_Vis_Episode(urdf=g1_urdf_path, hand_type=hand_type)
    arm_ik = G1_29_ArmIK(faster=True)
    
    # 初始化关节状态
    current_lr_arm_q = np.zeros(14)   # 14个关节角度
    current_lr_arm_dq = np.zeros(14)  # 14个关节速度
    
    T = len(xhand_actions)
    
    # 如果需要保存视频，使用 meshcat 的录制功能
    frames = []
    if args.save_video:
        print(f"将录制 {T} 帧并保存为视频...")
        print("提示: 请确保浏览器窗口可见以正确录制")
        
        # 尝试导入必要的库
        try:
            import cv2
            has_cv2 = True
        except ImportError:
            print("警告: 未安装 opencv-python")
            print("请运行: pip install opencv-python")
            has_cv2 = False
            args.save_video = False
    
    for t in range(T):
        start_time = time.time()
        
        # 提取当前帧的左右手姿态
        left_pose = xhand_to_hand_pose(xhand_actions[t], 'left')
        right_pose = xhand_to_hand_pose(xhand_actions[t], 'right')
        
        # 坐标系转换（注意传入 hand_side 参数）
        if args.transform_to_robot:
            left_pose = transform_camera_to_robot_base(left_pose, 'left')
            right_pose = transform_camera_to_robot_base(right_pose, 'right')
        
        # 打印调试信息
        if t == 0:
            print(f"\n帧 {t}/{T}:")
            print(f"  左手位置: [{left_pose[0,3]:.3f}, {left_pose[1,3]:.3f}, {left_pose[2,3]:.3f}]")
            print(f"  右手位置: [{right_pose[0,3]:.3f}, {right_pose[1,3]:.3f}, {right_pose[2,3]:.3f}]")
            # 打印朝向（X轴方向）
            left_dir = left_pose[:3, 0]
            right_dir = right_pose[:3, 0]
            print(f"  左手X轴朝向: [{left_dir[0]:.3f}, {left_dir[1]:.3f}, {left_dir[2]:.3f}]")
            print(f"  右手X轴朝向: [{right_dir[0]:.3f}, {right_dir[1]:.3f}, {right_dir[2]:.3f}]")
            print(f"  (注意：添加了右手180度翻转以修正朝向)")
        
        # IK 求解
        sol_q, sol_tauff = arm_ik.solve_ik(
            left_pose, right_pose, 
            current_lr_arm_q, current_lr_arm_dq
        )
        
        # 更新状态
        if t > 0:
            current_lr_arm_dq = (sol_q - current_lr_arm_q) * args.fps
        current_lr_arm_q = sol_q.copy()
        
        # 手指状态 - 从XHand数据中提取手指关节角度
        if args.use_finger_mapping:
            # 使用真实的手指关节数据
            left_q_target = map_xhand_fingers_to_dex3(xhand_actions[t], 'left')
            right_q_target = map_xhand_fingers_to_dex3(xhand_actions[t], 'right')
            
            if t == 0:
                print(f"\n手指关节映射 (帧0):")
                # 显示原始XHand手指关节值
                left_fingers = xhand_actions[t, 6:18]
                right_fingers = xhand_actions[t, 24:36]
                print(f"  左手XHand关节[6-17]: {left_fingers}")
                print(f"  右手XHand关节[24-35]: {right_fingers}")
                print(f"\n  映射详情:")
                print(f"  左手: thumb[{left_fingers[5]:.3f},{left_fingers[6]:.3f}] "
                      f"index[{left_fingers[7]:.3f},{left_fingers[10]:.3f}] "
                      f"middle[{left_fingers[8]:.3f},{left_fingers[9]:.3f}]")
                print(f"  右手: thumb[{right_fingers[5]:.3f},{right_fingers[6]:.3f}] "
                      f"index[{right_fingers[7]:.3f},{right_fingers[10]:.3f}] "
                      f"middle[{right_fingers[8]:.3f},{right_fingers[9]:.3f}]")
                print(f"\n  Dex3关节值:")
                print(f"  左手dex3: {left_q_target}")
                print(f"    thumb[{left_q_target[1]:.3f},{left_q_target[2]:.3f}] "
                      f"middle[{left_q_target[3]:.3f},{left_q_target[4]:.3f}] "
                      f"index[{left_q_target[5]:.3f},{left_q_target[6]:.3f}]")
                print(f"  右手dex3: {right_q_target}")
                print(f"    thumb[{right_q_target[1]:.3f},{right_q_target[2]:.3f}] "
                      f"middle[{right_q_target[3]:.3f},{right_q_target[4]:.3f}] "
                      f"index[{right_q_target[5]:.3f},{right_q_target[6]:.3f}]")
                print(f"\n  注意: 左手应该是负值弯曲，右手应该是正值弯曲")
                
            # 每30帧打印一次关节值变化
            if t > 0 and t % 30 == 0:
                print(f"\n帧{t}手指状态: L_dex3={left_q_target[1:3]} R_dex3={right_q_target[1:3]}")
        else:
            # 使用固定的gripper值
            left_gripper = args.left_gripper
            right_gripper = args.right_gripper
            left_q_target, right_q_target = map_gripper_to_hand(left_gripper, right_gripper)
        
        # 可视化
        show_current_q(vis_model, sol_q[:7], left_q_target, sol_q[7:], right_q_target)
        
        # 如果需要保存视频，尝试截取当前帧
        if args.save_video and has_cv2:
            try:
                # Meshcat 不直接提供帧缓冲访问
                # 我们需要使用屏幕截图或浏览器自动化
                # 这里提供一个基于 selenium 的方案
                if t == 0:
                    # 在第一帧尝试初始化 selenium
                    try:
                        from selenium import webdriver
                        from selenium.webdriver.chrome.options import Options
                        
                        # 获取 meshcat URL - 需要通过 viewer 属性访问
                        meshcat_url = vis_model.vis.viewer.url()
                        
                        # 配置无头浏览器
                        chrome_options = Options()
                        chrome_options.add_argument('--headless')
                        chrome_options.add_argument('--window-size=1920,1080')
                        chrome_options.add_argument('--disable-gpu')
                        
                        driver = webdriver.Chrome(options=chrome_options)
                        driver.get(meshcat_url)
                        time.sleep(2)  # 等待页面加载
                        
                        print(f"已连接到 meshcat: {meshcat_url}")
                        has_selenium = True
                    except Exception as e:
                        print(f"警告: 无法使用 selenium ({e})")
                        print("视频录制需要 selenium 和 chromedriver")
                        print("安装方法:")
                        print("  pip install selenium")
                        print("  # 并下载对应版本的 chromedriver")
                        has_selenium = False
                        args.save_video = False
                
                if args.save_video and has_selenium:
                    # 截取当前帧
                    screenshot = driver.get_screenshot_as_png()
                    import io
                    from PIL import Image
                    img = Image.open(io.BytesIO(screenshot))
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frames.append(frame)
                    
                    if (t + 1) % 10 == 0:
                        print(f"已录制 {t + 1}/{T} 帧...")
            except Exception as e:
                if t == 0:
                    print(f"警告: 录制失败 ({e})")
                    args.save_video = False
        
        # 控制帧率
        elapsed = time.time() - start_time
        sleep_time = max(0, (1.0 / args.fps) - elapsed)
        time.sleep(sleep_time)
    
    # 清理 selenium
    if args.save_video and 'driver' in locals():
        driver.quit()
    
    print(f"\n可视化完成! 共 {T} 帧")
    
    # 保存视频
    if args.save_video and len(frames) > 0:
        try:
            import cv2
            output_path = args.output_video
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, args.fps, (width, height))
            
            print(f"\n正在保存视频到: {output_path}")
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"✓ 视频已保存: {output_path}")
            print(f"  分辨率: {width}x{height}, 帧率: {args.fps} fps, 帧数: {len(frames)}")
        except Exception as e:
            print(f"错误: 保存视频失败 - {e}")


def visualize_simple(xhand_actions, args):
    """简单的可视化（不依赖 G1 模块）"""
    print("使用简化可视化模式...")
    print("提示: 安装 G1 机器人模块可获得完整的3D可视化效果")
    
    T = len(xhand_actions)
    
    # 提取轨迹
    left_traj = xhand_actions[:, 0:3]   # 左手位置轨迹
    right_traj = xhand_actions[:, 18:21]  # 右手位置轨迹
    
    print(f"\n动作序列统计 (共 {T} 帧):")
    print("="*60)
    
    print("\n左手轨迹:")
    print(f"  X 范围: [{left_traj[:, 0].min():.3f}, {left_traj[:, 0].max():.3f}] m")
    print(f"  Y 范围: [{left_traj[:, 1].min():.3f}, {left_traj[:, 1].max():.3f}] m")
    print(f"  Z 范围: [{left_traj[:, 2].min():.3f}, {left_traj[:, 2].max():.3f}] m")
    print(f"  移动距离: {np.linalg.norm(left_traj[-1] - left_traj[0]):.3f} m")
    
    print("\n右手轨迹:")
    print(f"  X 范围: [{right_traj[:, 0].min():.3f}, {right_traj[:, 0].max():.3f}] m")
    print(f"  Y 范围: [{right_traj[:, 1].min():.3f}, {right_traj[:, 1].max():.3f}] m")
    print(f"  Z 范围: [{right_traj[:, 2].min():.3f}, {right_traj[:, 2].max():.3f}] m")
    print(f"  移动距离: {np.linalg.norm(right_traj[-1] - right_traj[0]):.3f} m")
    
    print("\n关节角度范围:")
    left_joints = xhand_actions[:, 6:18]
    right_joints = xhand_actions[:, 24:36]
    print(f"  左手关节: [{left_joints.min():.3f}, {left_joints.max():.3f}] rad")
    print(f"  右手关节: [{right_joints.min():.3f}, {right_joints.max():.3f}] rad")
    
    print("="*60)
    
    # 尝试使用 matplotlib 绘制轨迹
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 5))
        
        # 3D 轨迹图
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(left_traj[:, 0], left_traj[:, 1], left_traj[:, 2], 
                'b-', label='left hand', linewidth=2)
        ax1.plot(right_traj[:, 0], right_traj[:, 1], right_traj[:, 2], 
                'r-', label='right hand', linewidth=2)
        ax1.scatter(left_traj[0, 0], left_traj[0, 1], left_traj[0, 2], 
                   c='blue', marker='o', s=100, label='left hand start point')
        ax1.scatter(right_traj[0, 0], right_traj[0, 1], right_traj[0, 2], 
                   c='red', marker='o', s=100, label='right hand start point')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('双手轨迹 (相机坐标系)')
        ax1.legend()
        ax1.grid(True)
        
        # 时间序列图
        ax2 = fig.add_subplot(122)
        time_steps = np.arange(T)
        ax2.plot(time_steps, left_traj[:, 0], 'b-', label='left hand X', alpha=0.7)
        ax2.plot(time_steps, left_traj[:, 1], 'g-', label='left hand Y', alpha=0.7)
        ax2.plot(time_steps, left_traj[:, 2], 'r-', label='left hand Z', alpha=0.7)
        ax2.set_xlabel('frame')
        ax2.set_ylabel('position/m')
        ax2.set_title('left hand position over time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = Path(args.xhand_file).parent / 'xhand_trajectory_plot.png'
        plt.savefig(output_path, dpi=150)
        print(f"\n轨迹图已保存到: {output_path}")
        
        # 显示（如果在支持GUI的环境中）
        try:
            plt.show()
        except:
            print("无法显示图形界面，但图像已保存")
            
    except ImportError:
        print("\n提示: 安装 matplotlib 可以查看轨迹可视化图表")
        print("  pip install matplotlib")


def main(args):
    # 加载 XHand 动作数据
    xhand_file = Path(args.xhand_file)
    if not xhand_file.exists():
        raise FileNotFoundError(f"找不到文件: {xhand_file}")
    
    print(f"加载 XHand 动作数据: {xhand_file}")
    xhand_actions = np.load(xhand_file)
    
    print(f"数据形状: {xhand_actions.shape}")
    print(f"  时间步数: {xhand_actions.shape[0]}")
    print(f"  动作维度: {xhand_actions.shape[1]} (左手18维 + 右手18维)")
    
    # 截取指定范围的帧
    if args.start_frame is not None or args.end_frame is not None:
        start = args.start_frame if args.start_frame is not None else 0
        end = args.end_frame if args.end_frame is not None else len(xhand_actions)
        xhand_actions = xhand_actions[start:end]
        print(f"使用帧范围: [{start}, {end}), 共 {len(xhand_actions)} 帧")
    
    # 选择可视化方法
    if HAS_G1_MODULES and not args.simple_vis:
        visualize_with_g1(xhand_actions, args)
    else:
        visualize_simple(xhand_actions, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="可视化 XHand 机器人动作序列"
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
    
    # 可视化参数
    parser.add_argument(
        '--fps', 
        type=float, 
        default=30.0,
        help='播放帧率（默认: 30）'
    )
    parser.add_argument(
        '--simple_vis', 
        action='store_true',
        help='使用简化可视化（不需要 G1 模块）'
    )
    
    # 坐标系转换
    parser.add_argument(
        '--transform_to_robot', 
        action='store_true',
        help='将相机坐标系转换为机器人基座坐标系'
    )
    
    # 手指控制
    parser.add_argument(
        '--left_gripper', 
        type=float, 
        default=0.0,
        help='左手抓取器开合度 (0.0-1.0)，仅在不使用手指映射时有效'
    )
    parser.add_argument(
        '--right_gripper', 
        type=float, 
        default=0.0,
        help='右手抓取器开合度 (0.0-1.0)，仅在不使用手指映射时有效'
    )
    parser.add_argument(
        '--use_finger_mapping',
        action='store_true',
        help='使用XHand手指关节数据映射到dex3灵巧手（大拇指和食指）'
    )
    
    # 视频保存
    parser.add_argument(
        '--save_video', 
        action='store_true',
        help='将可视化保存为视频文件'
    )
    parser.add_argument(
        '--output_video', 
        type=str, 
        default='output.mp4',
        help='输出视频文件路径'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
