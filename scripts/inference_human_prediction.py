import os
import sys
import cv2
import math
import json
import torch
import argparse
import numpy as np
from PIL import Image
from PIL import ImageOps
from pathlib import Path
import multiprocessing as mp
from vitra.models import VITRA_Paligemma, load_model
from vitra.utils.data_utils import resize_short_side_to_target, load_normalizer, recon_traj
from vitra.utils.config_utils import load_config
from vitra.datasets.human_dataset import pad_state_human, pad_action
from scipy.spatial.transform import Rotation as R
from vitra.datasets.dataset_utils import (
    compute_new_intrinsics_resize, 
    calculate_fov,
    ActionFeature,
    StateFeature,
)

repo_root = Path(__file__).parent.parent  # VITRA/
sys.path.insert(0, str(repo_root))

from visualization.visualize_core import HandVisualizer, normalize_camera_intrinsics, save_to_video, Renderer, process_single_hand_labels
from visualization.visualize_core import Config as HandConfig

def main():
    """
    Main execution function for hand action prediction and visualization.
    
    This function uses a multi-process architecture to separate hand reconstruction
    and VLA inference into independent processes, preventing CUDA conflicts.
    
    Workflow:
    1. Parse command-line arguments and load model configurations
    2. Initialize persistent services:
       - HandReconstructionService: Runs HAWOR + MOGE models in separate process
       - VLAInferenceService: Runs VLA model in separate process
    3. Load or reconstruct hand state:
       - Uses precomputed .npy file if available (same stem as image)
       - Otherwise runs hand reconstruction service
    4. Prepare input data:
       - Load and resize image
       - Extract hand state (translation, rotation, pose) for left/right hands
       - Create state and action masks based on which hands to predict
    5. Run VLA inference to predict future hand actions (multiple samples for diversity)
    6. Reconstruct absolute hand trajectories from relative actions
    7. Visualize predicted hand motions using MANO hand model
    8. Generate grid layout video showing all samples and save to file
    9. Cleanup: Shutdown persistent services and free GPU memory

    """
    parser = argparse.ArgumentParser(description="Hand VLA inference and visualization.")
    
    # Model Configuration
    parser.add_argument('--config_path', type=str, required=True, help='Path to model configuration JSON file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint (overrides config)')
    parser.add_argument('--statistics_path', type=str, default=None, help='Path to normalization statistics JSON (overrides config)')
    
    # Input/Output
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image file')
    parser.add_argument('--hand_path', type=str, default=None, help='Path to hand state .npy file (optional, will run reconstruction if not provided)')
    parser.add_argument('--video_path', type=str, default='./example_human_inf.mp4', help='Path to save output visualization video')
    
    # Hand Reconstruction Models
    parser.add_argument('--hawor_model_path', type=str, default='./weights/hawor/checkpoints/hawor.ckpt', help='Path to HAWOR model weights')
    parser.add_argument('--detector_path', type=str, default='./weights/hawor/external/detector.pt', help='Path to hand detector model')
    parser.add_argument('--moge_model_name', type=str, default='Ruicheng/moge-2-vitl', help='MOGE model name from Hugging Face')
    parser.add_argument('--mano_path', type=str, default='/home/t-qixiuli/repo/VITRA/weights/mano', help='Path to MANO model files')
    # parser.add_argument('--output_path', type=str, default='./recon_results.npy', help='Path to save reconstruction results')
    
    # Prediction Settings
    parser.add_argument('--use_left', action='store_true', help='Enable left hand prediction')
    parser.add_argument('--use_right', action='store_true', help='Enable right hand prediction')
    parser.add_argument('--instruction', type=str, default="Left: Put the trash into the garbage. Right: None.", help='Text instruction for hand motion')
    parser.add_argument('--sample_times', type=int, default=4, help='Number of action samples to generate for diversity')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second for output video')
    
    # Advanced Options
    parser.add_argument('--save_state_local', action='store_true', help='Save hand state locally as .npy file')

    # === Environment Configuration ===
    # Disable tokenizers parallelism to avoid deadlocks in multi-process data loading
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parser.parse_args()
    
    # Validate that at least one hand is selected
    if not args.use_left and not args.use_right:
        raise ValueError("At least one of --use_left or --use_right must be specified.")
    
    # Load configs
    configs = load_config(args.config_path)
    
    # Override config with command-line arguments if provided
    if args.model_path is not None:
        configs['model_load_path'] = args.model_path
    if args.statistics_path is not None:
        configs['statistics_path'] = args.statistics_path

    # Check if a precomputed hand reconstruction .npy (same stem as image) exists.
    image_path_obj = Path(args.image_path)
    npy_path = image_path_obj.with_suffix('.npy')

    # Initialize services
    print("Initializing services...")
    if npy_path.exists():
        # If precomputed .npy exists, load hand state from it
        # Hand State File Format
        # The hand state is stored as a .npy file containing a Python dictionary with the following structure:
        # Format: .npy
        # Content: dictionary with MANO-based hand pose parameters and camera FOV.
        # Structure:
        # {
        #     'left': {
        #         0: {
        #             'hand_pose': np.ndarray,      # [15, 3, 3] rotation matrices for MANO joints
        #             'global_orient': np.ndarray,  # [3, 3] global rotation matrix
        #             'transl': np.ndarray,         # [3] root translation in camera coordinates
        #             'beta': np.ndarray            # [10] MANO shape parameters
        #         }
        #     },
        #     'right': {                           # Same structure as 'left'
        #         0: {
        #             ...
        #         }
        #     },
        #     'fov_x': float                       # Horizontal field of view (in degrees)
        # }

        print(f"Found precomputed hand state results: {npy_path}. Using the state instead of running hand recon.")
        hand_data = np.load(npy_path, allow_pickle=True).item()

        hand_recon_service = None
    else:
        print(f"No precomputed hand state .npy found at {npy_path}. Starting hand reconstruction service.")
        
        # Start hand reconstruction service
        hand_recon_service = HandReconstructionService(args)
        hand_data = None


    # Start VLA service (normalizer and model are loaded inside the service)
    vla_service = VLAInferenceService(configs)
    
    # Visualization setup
    hand_config = HandConfig(args)
    hand_config.FPS = args.fps
    visualizer = HandVisualizer(hand_config, render_gradual_traj=False)

    try:
        # ============================================================================
        # 阶段1：手部3D重建 (Hand 3D Reconstruction)
        # ============================================================================
        # 功能：从输入图像中检测并重建双手的3D姿态
        # 子步骤：
        #   1.1 YOLO手部检测 - 定位图像中的左右手区域
        #   1.2 HaWoR姿态估计 - 估计每只手的3D位置、旋转和关节角度
        #   1.3 MoGe相机参数 - 估计相机的视场角(FoV)
        #   1.4 MANO对齐 - 将手部参数对齐到标准MANO模型
        # 输出：hand_data字典，包含左右手的姿态参数和相机FOV
        # ============================================================================
        
        if hand_data is None:
            # 如果没有预计算的手部状态，则运行手部重建服务
            print("Running hand reconstruction...")
            hand_data = hand_recon_service.reconstruct(args.image_path)
            
            if args.save_state_local:
                # 可选：将重建的手部状态保存到本地，供下次使用
                np.save(npy_path, hand_data, allow_pickle=True)
                print(f"Saved reconstructed hand state to {npy_path}")

        # 加载并预处理输入图像
        image = Image.open(args.image_path)
        ori_w, ori_h = image.size  # 保存原始图像尺寸

        # 处理EXIF方向信息（修正手机拍摄的旋转图像）
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass  # 如果EXIF处理失败，继续使用原始图像

        # 将图像短边缩放到224像素（保持宽高比）
        image_resized = resize_short_side_to_target(image, target=224)
        w, h = image_resized.size

        # 确定要预测哪只手（左手、右手或双手）
        use_right = args.use_right
        use_left = args.use_left

        # 初始化手部状态变量
        current_state_left = None
        current_state_right = None
        
        # 从重建结果中提取当前时刻的手部状态
        if use_right:
            # 右手状态：[x,y,z, rx,ry,rz, 45个关节角度] + [10个形状参数]
            current_state_right, beta_right, fov_x, _ = get_state(hand_data, hand_side='right')
        if use_left:
            # 左手状态：结构同右手
            current_state_left, beta_left, fov_x, _ = get_state(hand_data, hand_side='left')
        
        # 计算相机内参矩阵
        # 将水平视场角从度转换为弧度
        fov_x = fov_x * np.pi / 180
        # 根据原始图像尺寸计算焦距
        f_ori = ori_w / np.tan(fov_x / 2) / 2
        # 根据原始图像的宽高比计算垂直视场角
        fov_y = 2 * np.arctan(ori_h / (2 * f_ori))

        # 根据缩放后的图像尺寸重新计算焦距
        f = w / np.tan(fov_x / 2) / 2
        # 构建相机内参矩阵 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        intrinsics = np.array([
            [f, 0, w/2],      # fx和主点x坐标
            [0, f, h/2],      # fy和主点y坐标
            [0, 0, 1]         # 齐次坐标
        ])

        # ============================================================================
        # 阶段2：VLA模型推理 (VLA Model Inference)
        # ============================================================================
        # 功能：基于当前手部状态、图像和文本指令，预测未来的手部动作序列
        # 输入准备：
        #   2.1 状态向量构建 - 拼接左右手状态和形状参数
        #   2.2 掩码设置 - 指定哪些手需要预测
        #   2.3 数据标准化 - 归一化状态和动作空间
        # 模型推理：
        #   2.4 VLA扩散模型 - 使用DDIM采样生成多样化的动作预测
        #   2.5 多次采样 - 生成sample_times个不同的预测结果
        # 输出：未来16帧的手部动作增量（相对当前状态）
        # ============================================================================
        
        # 拼接左右手状态，如果某只手不使用则用零填充
        if current_state_left is None and current_state_right is None:
            raise ValueError("Both current_state_left and current_state_right are None")
        
        # 构建统一的状态向量（如果不使用某只手，用对侧手的形状填充零）
        state_left = current_state_left if use_left else np.zeros_like(current_state_right)
        beta_left = beta_left if use_left else np.zeros_like(beta_right)
        state_right = current_state_right if use_right else np.zeros_like(current_state_left)
        beta_right = beta_right if use_right else np.zeros_like(beta_left)
        
        # 最终状态向量：[左手状态(51) + 左手形状(10) + 右手状态(51) + 右手形状(10)] = 122维
        state = np.concatenate([state_left, beta_left, state_right, beta_right], axis=0)
        
        # 状态掩码：标记哪些手是有效的
        state_mask = np.array([use_left, use_right], dtype=bool)
        
        # 从配置中获取预测帧数（默认16帧）
        chunk_size = configs.get('fwd_pred_next_n', 16)
        
        # 动作掩码：为每一帧指定哪些手需要预测动作
        # shape: [16, 2] 表示16帧，每帧2只手（左、右）
        action_mask = np.tile(np.array([[use_left, use_right]], dtype=bool), (chunk_size, 1))

        # 相机视场角（水平和垂直）
        fov = np.array([fov_x, fov_y], dtype=np.float32)
        
        # 将PIL图像转换为numpy数组供模型使用
        image_resized_np = np.array(image_resized)

        # 使用命令行参数中的文本指令
        instruction = args.instruction

        # 调用VLA推理服务进行动作预测
        # 服务内部会自动处理：归一化、padding、扩散采样、反归一化
        print(f"Running VLA inference...")
        sample_times = args.sample_times  # 采样次数（生成多个多样化的预测）
        unnorm_action = vla_service.predict(
            image=image_resized_np,              # 输入图像
            instruction=instruction,              # 文本指令
            state=state,                          # 当前手部状态
            state_mask=state_mask,                # 状态有效性掩码
            action_mask=action_mask,              # 动作预测掩码
            fov=fov,                              # 相机视场角
            num_ddim_steps=10,                    # DDIM扩散采样步数
            cfg_scale=5.0,                        # 分类器自由引导强度
            sample_times=sample_times,            # 采样次数
        )
        
        # ============================================================================
        # 阶段3：轨迹重建 (Trajectory Reconstruction)
        # ============================================================================
        # 功能：将VLA预测的相对动作增量转换为绝对手部轨迹
        # 处理步骤：
        #   3.1 动作累积 - 将相对增量累加到初始状态，生成完整轨迹
        #   3.2 坐标转换 - 从欧拉角转换为旋转矩阵（供MANO使用）
        #   3.3 MANO生成 - 使用MANO模型生成每一帧的手部mesh顶点
        # 输出：每一帧的手部顶点坐标序列（用于渲染）
        # ============================================================================
        
        # 提取相机内参用于3D渲染
        fx_exo = intrinsics[0, 0]  # 焦距x
        fy_exo = intrinsics[1, 1]  # 焦距y
        # 初始化3D渲染器（用于将手部mesh投影到图像平面）
        renderer = Renderer(w, h, (fx_exo, fy_exo), 'cuda')

        # 时间步数 = 动作帧数 + 1（包含初始状态）
        T = len(action_mask) + 1
        
        # 为所有采样结果预分配轨迹存储空间
        # shape: [采样次数, 时间步, 51维状态]
        traj_right_list = np.zeros((sample_times, T, 51), dtype=np.float32)
        traj_left_list = np.zeros((sample_times, T, 51), dtype=np.float32)

        # 构建轨迹掩码（标记每一帧哪些手是有效的）
        traj_mask = np.tile(np.array([[use_left, use_right]], dtype=bool), (T, 1))
        left_hand_mask = traj_mask[:, 0]   # 左手的有效性掩码
        right_hand_mask = traj_mask[:, 1]  # 右手的有效性掩码

        # 将左右手掩码打包成元组，供可视化使用
        hand_mask = (left_hand_mask, right_hand_mask)

        # 存储所有采样结果的渲染帧
        all_rendered_frames = []
        
        # 遍历每一次采样结果，重建轨迹并生成可视化
        for i in range(sample_times):
            traj_right = traj_right_list[i]
            traj_left = traj_left_list[i]
            
            # 将相对动作累积到初始状态，重建完整的绝对轨迹
            if use_left:
                # 左手轨迹重建：从初始状态开始，逐帧累加相对动作
                traj_left = recon_traj(
                    state=state_left,                    # 初始状态 [51]
                    rel_action=unnorm_action[i, :, 0:51],  # 相对动作序列 [16, 51]
                )
            if use_right:
                # 右手轨迹重建：结构同左手
                traj_right = recon_traj(
                    state=state_right,
                    rel_action=unnorm_action[i, :, 51:102],
                )
        
            # 将轨迹数据转换为MANO模型所需的格式
            # 左手参数字典
            left_hand_labels = {
                'transl_worldspace': traj_left[:, 0:3],  # 位置轨迹 [T, 3]
                'global_orient_worldspace': R.from_euler('xyz', traj_left[:, 3:6]).as_matrix(),  # 全局旋转 [T, 3, 3]
                'hand_pose': euler_traj_to_rotmat_traj(traj_left[:, 6:51], T),  # 关节角度 [T, 15, 3, 3]
                'beta': beta_left,  # 形状参数 [10]
            }
            # 右手参数字典
            right_hand_labels = {
                'transl_worldspace': traj_right[:, 0:3],
                'global_orient_worldspace': R.from_euler('xyz', traj_right[:, 3:6]).as_matrix(),
                'hand_pose': euler_traj_to_rotmat_traj(traj_right[:, 6:51], T),
                'beta': beta_right,
            }

            # 使用MANO模型生成手部mesh顶点坐标
            # 返回：[T, 778, 3] 每帧778个顶点的3D坐标
            verts_left_worldspace, _ = process_single_hand_labels(left_hand_labels, left_hand_mask, visualizer.mano, is_left=True)
            verts_right_worldspace, _ = process_single_hand_labels(right_hand_labels, right_hand_mask, visualizer.mano, is_left=False)

            # ============================================================================
            # 阶段4：可视化渲染 (Visualization and Rendering)
            # ============================================================================
            # 功能：将预测的手部mesh渲染到图像上，生成可视化视频
            # 渲染步骤：
            #   4.1 准备相机外参 - 设置世界坐标到相机坐标的转换
            #   4.2 mesh投影 - 将3D手部顶点投影到2D图像平面
            #   4.3 颜色渲染 - 为手部mesh着色（左手和右手使用不同颜色）
            #   4.4 图像合成 - 将渲染的手部叠加到原始图像上
            # 多样性展示：
            #   4.5 网格布局 - 将多个采样结果排列成网格
            #   4.6 视频保存 - 输出为mp4格式视频文件
            # ============================================================================
            
            # 打包左右手的顶点轨迹
            hand_traj_wordspace = (verts_left_worldspace, verts_right_worldspace)
            
            # 设置相机外参（此处使用单位矩阵，即相机坐标系=世界坐标系）
            R_w2c = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()  # 旋转矩阵 [T, 3, 3]
            t_w2c = np.zeros((T, 3, 1), dtype=np.float32)          # 平移向量 [T, 3, 1]

            extrinsics = (R_w2c, t_w2c)

            # 准备背景图像（将RGB转换为BGR格式，因为OpenCV使用BGR）
            image_bgr = image_resized_np[..., ::-1]
            # 为每一帧复制相同的背景图像
            resize_video_frames = [image_bgr] * T
            
            # 渲染手部轨迹到图像序列
            # mode='first'表示只在第一帧显示完整手部，后续帧显示轨迹
            save_frames = visualizer._render_hand_trajectory(
                resize_video_frames,      # 背景图像序列
                hand_traj_wordspace,      # 手部顶点轨迹
                hand_mask,                # 手部有效性掩码
                extrinsics,               # 相机外参
                renderer,                 # 3D渲染器
                mode='first'              # 渲染模式
            )
        
            # 保存当前采样的渲染结果
            all_rendered_frames.append(save_frames)
        
        # ============================================================================
        # 最终输出：多样性展示
        # ============================================================================
        # 将所有采样结果在空间上拼接成网格布局的单个视频
        # 例如：4次采样 -> 2x2网格，9次采样 -> 3x3网格
        # ============================================================================
        
        # all_rendered_frames: 包含sample_times个帧列表的列表
        # 每个帧列表有T帧图像
        num_frames = len(all_rendered_frames[0])
        
        # 确定网格布局尺寸（尽可能接近正方形）
        # 例如：4个样本 -> 2列x2行，6个样本 -> 3列x2行
        grid_cols = math.ceil(math.sqrt(sample_times))  # 列数
        grid_rows = math.ceil(sample_times / grid_cols)  # 行数
        
        # 逐帧拼接所有采样结果
        combined_frames = []
        for frame_idx in range(num_frames):
            # 收集所有采样在当前时间步的帧
            sample_frames = [all_rendered_frames[i][frame_idx] for i in range(sample_times)]
            
            # 如果采样数不足以填满网格，用黑色帧填充
            while len(sample_frames) < grid_rows * grid_cols:
                black_frame = np.zeros_like(sample_frames[0])
                sample_frames.append(black_frame)
            
            # 将帧排列成网格
            rows = []
            for row_idx in range(grid_rows):
                # 获取当前行的所有帧
                row_frames = sample_frames[row_idx * grid_cols:(row_idx + 1) * grid_cols]
                # 水平拼接当前行的帧
                row_concat = np.concatenate(row_frames, axis=1)
                rows.append(row_concat)
            
            # 垂直拼接所有行，形成最终的网格布局
            combined_frame = np.concatenate(rows, axis=0)
            combined_frames.append(combined_frame)

        # 将拼接后的帧序列保存为视频文件
        save_to_video(combined_frames, f'{args.video_path}', fps=hand_config.FPS)
        print(f"Combined video with {sample_times} samples saved to {args.video_path}")
    
    finally:
        # Cleanup persistent services
        print("Shutting down services...")
        if hand_recon_service is not None:
            hand_recon_service.shutdown()
        vla_service.shutdown()
        print("All services shut down successfully")
    

def get_state(hand_data, hand_side='right'):
    """
    从手部重建数据中提取指定手的状态信息。
    
    功能说明：
        将MANO模型的旋转矩阵格式转换为欧拉角格式，便于VLA模型处理。
        提取的状态包括手腕位置、全局旋转和所有关节的角度。
    
    参数:
        hand_data (dict): 手部重建数据字典，包含'left'、'right'和'fov_x'键
        hand_side (str): 要提取的手，'left'或'right'，默认为'right'
        
    返回:
        tuple: (state_t0, beta, fov_x, None) 其中:
            - state_t0 (np.ndarray): [51]维手部状态向量，包含:
                * [0:3] - 手腕3D位置 (x, y, z)
                * [3:6] - 全局旋转（欧拉角 rx, ry, rz）
                * [6:51] - 手部姿态（15个关节 × 3个欧拉角 = 45维）
            - beta (np.ndarray): [10]维MANO形状参数（PCA系数）
            - fov_x (float): 水平视场角（度）
            - None: 占位符（用于可选的文本标注）
    """
    if hand_side not in ['left', 'right']:
        raise ValueError(f"hand_side must be 'left' or 'right', got '{hand_side}'")
    
    # 提取初始时刻(t0)的手部姿态（旋转矩阵格式）
    hand_pose_t0 = hand_data[hand_side][0]['hand_pose']  # [15, 3, 3] 15个关节的旋转矩阵
    
    # 将旋转矩阵转换为欧拉角（xyz顺序，弧度制）
    hand_pose_t0_euler = R.from_matrix(hand_pose_t0).as_euler('xyz', degrees=False)  # [15, 3]
    hand_pose_t0_euler = hand_pose_t0_euler.reshape(-1)  # 展平为 [45] 维向量
    
    # 提取全局旋转并转换为欧拉角
    global_orient_mat_t0 = hand_data[hand_side][0]['global_orient']  # [3, 3] 全局旋转矩阵
    R_t0_euler = R.from_matrix(global_orient_mat_t0).as_euler('xyz', degrees=False)  # [3] 欧拉角
    
    # 提取手腕的3D位置（相机坐标系）
    transl_t0 = hand_data[hand_side][0]['transl']  # [3] (x, y, z)
    
    # 拼接成完整的状态向量：位置(3) + 旋转(3) + 姿态(45) = 51维
    state_t0 = np.concatenate([transl_t0, R_t0_euler, hand_pose_t0_euler])
    
    # 提取相机视场角
    fov_x = hand_data['fov_x']

    return state_t0, hand_data[hand_side][0]['beta'], fov_x, None

def euler_traj_to_rotmat_traj(euler_traj, T):
    """
    将欧拉角轨迹转换为旋转矩阵轨迹。
    
    功能说明：
        VLA模型输出的是欧拉角表示的手部姿态，而MANO模型需要旋转矩阵作为输入。
        此函数完成从欧拉角到旋转矩阵的批量转换。
    
    参数:
        euler_traj (np.ndarray): 欧拉角格式的手部姿态轨迹
                                 形状: [T, 45] 其中T是时间步数
                                 45 = 15个关节 × 3个欧拉角/关节
        T (int): 轨迹的时间步数
        
    返回:
        np.ndarray: 旋转矩阵格式的姿态轨迹
                    形状: [T, 15, 3, 3]
                    其中每个 [3, 3] 块是一个关节的旋转矩阵
    
    转换过程:
        [T, 45] -> [T*15, 3] -> [T*15, 3, 3] -> [T, 15, 3, 3]
    """
    # 重塑为每个关节一行的格式
    hand_pose = euler_traj.reshape(-1, 3)  # [T*15, 3] 每行是一个关节的xyz欧拉角
    
    # 批量转换欧拉角为旋转矩阵
    pose_matrices = R.from_euler('xyz', hand_pose).as_matrix()  # [T*15, 3, 3]
    
    # 重塑回原始的时间和关节维度
    pose_matrices = pose_matrices.reshape(T, 15, 3, 3)  # [T, 15, 3, 3]

    return pose_matrices


def _hand_reconstruction_worker(args_dict, task_queue, result_queue):
    """
    Persistent worker for hand reconstruction that runs in a separate process.
    Keeps model loaded and processes multiple requests until shutdown signal.
    """
    from data.tools.hand_recon_core import Config, HandReconstructor
    
    hand_reconstructor = None
    
    try:
        # Reconstruct args object
        class ArgsObj:
            pass
        args_obj = ArgsObj()
        for key, value in args_dict.items():
            setattr(args_obj, key, value)
        
        # Initialize hand reconstructor once
        print("[HandRecon Process] Initializing hand reconstructor...")
        config = Config(args_obj)
        hand_reconstructor = HandReconstructor(config=config, device='cuda')
        print("[HandRecon Process] Hand reconstructor ready")
        
        # Signal ready
        result_queue.put({'type': 'ready'})
        
        # Process tasks in loop
        while True:
            task = task_queue.get()
            
            if task['type'] == 'shutdown':
                print("[HandRecon Process] Received shutdown signal")
                break
            
            elif task['type'] == 'reconstruct':
                try:
                    image_path = task['image_path']
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"Failed to load image from {image_path}")
                    
                    image_list = [image]
                    recon_results = hand_reconstructor.recon(image_list)
                    
                    result_queue.put({
                        'type': 'result',
                        'success': True,
                        'data': recon_results
                    })
                    
                except Exception as e:
                    import traceback
                    result_queue.put({
                        'type': 'result',
                        'success': False,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
        
    except Exception as e:
        import traceback
        result_queue.put({
            'type': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    finally:
        # Cleanup on shutdown
        if hand_reconstructor is not None:
            del hand_reconstructor
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[HandRecon Process] Cleaned up and exiting")


def _vla_inference_worker(configs_dict, task_queue, result_queue):
    """
    Persistent worker for VLA model inference that runs in a separate process.
    Keeps model loaded and processes multiple requests until shutdown signal.
    """
    from vitra.models import load_model
    from vitra.utils.data_utils import load_normalizer
    from vitra.datasets.human_dataset import pad_state_human, pad_action
    from vitra.datasets.dataset_utils import ActionFeature, StateFeature
    
    model = None
    normalizer = None
    
    try:
        # Load model and normalizer once
        print("[VLA Process] Loading VLA model...")
        model = load_model(configs_dict).cuda()
        model.eval()
        normalizer = load_normalizer(configs_dict)
        print(f"[VLA Process] VLA model ready.")
        
        # Signal ready
        result_queue.put({'type': 'ready'})
        
        # Process tasks in loop
        while True:
            task = task_queue.get()
            
            if task['type'] == 'shutdown':
                print("[VLA Process] Received shutdown signal")
                break
            
            elif task['type'] == 'predict':
                try:
                    image = task['image']
                    instruction = task['instruction']
                    state = task['state']
                    state_mask = task['state_mask']
                    action_mask = task['action_mask']
                    fov = task['fov']
                    num_ddim_steps = task.get('num_ddim_steps', 10)
                    cfg_scale = task.get('cfg_scale', 5.0)
                    sample_times = task.get('sample_times', 1)
                    
                    # Normalize state
                    norm_state = normalizer.normalize_state(state.copy())
                    
                    # Pad state and action
                    unified_action_dim = ActionFeature.ALL_FEATURES[1]  # 192
                    unified_state_dim = StateFeature.ALL_FEATURES[1]    # 212
                    
                    unified_state, unified_state_mask = pad_state_human(
                        state=norm_state,
                        state_mask=state_mask,
                        action_dim=normalizer.action_mean.shape[0],
                        state_dim=normalizer.state_mean.shape[0],
                        unified_state_dim=unified_state_dim,
                    )
                    _, unified_action_mask = pad_action(
                        actions=None,
                        action_mask=action_mask.copy(),
                        action_dim=normalizer.action_mean.shape[0],
                        unified_action_dim=unified_action_dim
                    )
                    
                    # Convert to torch and move to GPU
                    fov = torch.from_numpy(fov).unsqueeze(0)
                    unified_state = unified_state.unsqueeze(0)
                    unified_state_mask = unified_state_mask.unsqueeze(0)
                    unified_action_mask = unified_action_mask.unsqueeze(0)
                    
                    # Run inference
                    norm_action = model.predict_action(
                        image=image,
                        instruction=instruction,
                        current_state=unified_state,
                        current_state_mask=unified_state_mask,
                        action_mask_torch=unified_action_mask,
                        num_ddim_steps=num_ddim_steps,
                        cfg_scale=cfg_scale,
                        fov=fov,
                        sample_times=sample_times,
                    )
                    
                    # Extract and denormalize action
                    norm_action = norm_action[:, :, :102]
                    unnorm_action = normalizer.unnormalize_action(norm_action)
                    
                    # Convert to numpy for inter-process communication
                    if isinstance(unnorm_action, torch.Tensor):
                        unnorm_action_np = unnorm_action.cpu().numpy()
                    else:
                        unnorm_action_np = np.array(unnorm_action)
                    
                    result_queue.put({
                        'type': 'result',
                        'success': True,
                        'data': unnorm_action_np
                    })
                    
                except Exception as e:
                    import traceback
                    result_queue.put({
                        'type': 'result',
                        'success': False,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
        
    except Exception as e:
        import traceback
        result_queue.put({
            'type': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    finally:
        # Cleanup on shutdown
        if model is not None:
            del model
        if normalizer is not None:
            del normalizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[VLA Process] Cleaned up and exiting")


class HandReconstructionService:
    """Service wrapper for persistent hand reconstruction process"""
    
    def __init__(self, args):
        self.ctx = mp.get_context('spawn')
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        
        # Convert args to dict for pickling
        args_dict = {
            'hawor_model_path': args.hawor_model_path,
            'detector_path': args.detector_path,
            'moge_model_name': args.moge_model_name,
            'mano_path': args.mano_path,
        }
        
        # Start persistent process
        self.process = self.ctx.Process(
            target=_hand_reconstruction_worker,
            args=(args_dict, self.task_queue, self.result_queue)
        )
        self.process.start()
        
        # Wait for ready signal
        ready_msg = self.result_queue.get()
        if ready_msg['type'] == 'ready':
            print("Hand reconstruction service initialized")
        elif ready_msg['type'] == 'error':
            raise RuntimeError(f"Failed to initialize hand reconstruction: {ready_msg['error']}")
    
    def reconstruct(self, image_path):
        """Request hand reconstruction for an image"""
        self.task_queue.put({
            'type': 'reconstruct',
            'image_path': image_path
        })
        
        result = self.result_queue.get()
        if result['type'] == 'result' and result['success']:
            return result['data']
        else:
            raise RuntimeError(f"Hand reconstruction failed: {result.get('error', 'Unknown error')}")
    
    def shutdown(self):
        """Shutdown the persistent process"""
        self.task_queue.put({'type': 'shutdown'})
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


class VLAInferenceService:
    """Service wrapper for persistent VLA inference process"""
    
    def __init__(self, configs):
        self.ctx = mp.get_context('spawn')
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        
        # Start persistent process
        self.process = self.ctx.Process(
            target=_vla_inference_worker,
            args=(configs, self.task_queue, self.result_queue)
        )
        self.process.start()
        
        # Wait for ready signal
        ready_msg = self.result_queue.get()
        if ready_msg['type'] == 'ready':
            print("VLA inference service initialized")
        elif ready_msg['type'] == 'error':
            raise RuntimeError(f"Failed to initialize VLA model: {ready_msg['error']}")
    
    def predict(self, image, instruction, state, state_mask, action_mask, 
                fov, num_ddim_steps=10, cfg_scale=5.0, sample_times=1):
        """Request action prediction with state normalization and padding"""

        self.task_queue.put({
            'type': 'predict',
            'image': image,
            'instruction': instruction,
            'state': state,
            'state_mask': state_mask,
            'action_mask': action_mask,
            'fov': fov,
            'num_ddim_steps': num_ddim_steps,
            'cfg_scale': cfg_scale,
            'sample_times': sample_times,
        })
        
        result = self.result_queue.get()
        if result['type'] == 'result' and result['success']:
            # Return unnormalized action as numpy array
            return result['data']
        else:
            raise RuntimeError(f"VLA inference failed: {result.get('error', 'Unknown error')}")
    
    def shutdown(self):
        """Shutdown the persistent process"""
        self.task_queue.put({'type': 'shutdown'})
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' to avoid CUDA fork issues
    mp.set_start_method('spawn', force=True)
    main()