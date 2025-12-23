from robot_arm_ik import G1_29_ArmIK
import numpy as np

arm_ik = G1_29_ArmIK(faster=True)

left_arm_pose = np.eye(4)
left_arm_pose[:3, 3] = [0.25, 0.25, 0.1]  # 设置位置

right_arm_pose = np.eye(4)
right_arm_pose[:3, 3] = [0.25, -0.25, 0.1]  # 设置位置

# 关节状态参数 (14个关节: 左臂7个 + 右臂7个)
current_lr_arm_q = np.zeros(14)   # 当前关节角度 (弧度)
current_lr_arm_dq = np.zeros(14)  # 当前关节速度 (弧度/秒)

sol_q, sol_tauff  = arm_ik.solve_ik(left_arm_pose, right_arm_pose, current_lr_arm_q, current_lr_arm_dq)
print("Left and Right Arm Joint Angles (radians):", sol_q)
print("Left and Right Arm Joint Torques (Nm):", sol_tauff)