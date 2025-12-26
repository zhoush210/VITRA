import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import numpy as np

kPi = 3.141592654
kPi_2 = 1.57079632

class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29 # NOTE: Weight

class Custom:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.02  
        self.duration_ = 3.0   
        self.counter_ = 0
        self.weight = 0.
        self.weight_rate = 0.2
        self.kp = 60.
        self.kd = 1.5
        self.dq = 0.
        self.tau_ff = 0.
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.first_update_low_state = False
        self.crc = CRC()
        self.done = False

        self.target_pos = [
            0., kPi_2,  0., kPi_2, 0., 0., 0.,
            0., -kPi_2, 0., kPi_2, 0., 0., 0., 
            0, 0, 0
        ]

        self.arm_joints = [
          G1JointIndex.LeftShoulderPitch,  G1JointIndex.LeftShoulderRoll,
          G1JointIndex.LeftShoulderYaw,    G1JointIndex.LeftElbow,
          G1JointIndex.LeftWristRoll,      G1JointIndex.LeftWristPitch,
          G1JointIndex.LeftWristYaw,
          G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
          G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
          G1JointIndex.RightWristRoll,     G1JointIndex.RightWristPitch,
          G1JointIndex.RightWristYaw,
          G1JointIndex.WaistYaw,
          G1JointIndex.WaistRoll,
          G1JointIndex.WaistPitch
        ]

    def Init(self):
        # create publisher #
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.first_update_low_state == False:
            time.sleep(1)

        if self.first_update_low_state == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.first_update_low_state == False:
            self.first_update_low_state = True
        
    def LowCmdWrite(self):
        self.time_ += self.control_dt_

        if self.time_ < self.duration_ :
          # [Stage 1]: set robot to zero posture
          self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q =  1 # 1:Enable arm_sdk, 0:Disable arm_sdk
          for i,joint in enumerate(self.arm_joints):
            ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
            self.low_cmd.motor_cmd[joint].tau = 0. 
            self.low_cmd.motor_cmd[joint].q = (1.0 - ratio) * self.low_state.motor_state[joint].q 
            self.low_cmd.motor_cmd[joint].dq = 0. 
            self.low_cmd.motor_cmd[joint].kp = self.kp 
            self.low_cmd.motor_cmd[joint].kd = self.kd

        elif self.time_ < self.duration_ * 3 :
          # [Stage 2]: lift arms up
          for i,joint in enumerate(self.arm_joints):
              ratio = np.clip((self.time_ - self.duration_) / (self.duration_ * 2), 0.0, 1.0)
              self.low_cmd.motor_cmd[joint].tau = 0. 
              self.low_cmd.motor_cmd[joint].q = ratio * self.target_pos[i] + (1.0 - ratio) * self.low_state.motor_state[joint].q 
              self.low_cmd.motor_cmd[joint].dq = 0. 
              self.low_cmd.motor_cmd[joint].kp = self.kp 
              self.low_cmd.motor_cmd[joint].kd = self.kd

        elif self.time_ < self.duration_ * 6 :
          # [Stage 3]: set robot back to zero posture
          for i,joint in enumerate(self.arm_joints):
              ratio = np.clip((self.time_ - self.duration_*3) / (self.duration_ * 3), 0.0, 1.0)
              self.low_cmd.motor_cmd[joint].tau = 0. 
              self.low_cmd.motor_cmd[joint].q = (1.0 - ratio) * self.low_state.motor_state[joint].q
              self.low_cmd.motor_cmd[joint].dq = 0. 
              self.low_cmd.motor_cmd[joint].kp = self.kp 
              self.low_cmd.motor_cmd[joint].kd = self.kd

        elif self.time_ < self.duration_ * 7 :
          # [Stage 4]: release arm_sdk
          for i,joint in enumerate(self.arm_joints):
              ratio = np.clip((self.time_ - self.duration_*6) / (self.duration_), 0.0, 1.0)
              self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q =  (1 - ratio) # 1:Enable arm_sdk, 0:Disable arm_sdk
        
        else:
            self.done = True
  
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

    def init_armpos(self, init_left_arm, init_right_arm, duration=3.0):
        """把双臂平滑收回到零位；该函数会阻塞 `duration` 秒"""
        steps = int(duration / self.control_dt_)
        t = 0.0
        # 确保已经有 low_state
        while not self.first_update_low_state:
            time.sleep(0.01)

        # 启动 arm_sdk
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1

        '''
        right_arm
        array([-0.01813975, -0.15770234,  0.07567498,  0.27876395, -0.4835219 ,
       -0.40243876, -0.29535112, -0.28364822, -0.80444217, -0.15418975,
        0.37165838,  0.02983215,  0.36935064,  0.02350703], dtype=float32)
        '''
        for _ in range(steps):
            ratio = np.clip(t / duration, 0.0, 1.0)
            for i, joint in enumerate(self.arm_joints):
                
                self.low_cmd.motor_cmd[joint].tau = 0.
                #left arm 15-21
                if joint ==15:  # 左肩 Roll
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_left_arm[0]
                elif joint == 16:  # 左肩 Pitch
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_left_arm[1]
                elif joint == 17:  # 左肩 Yaw
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_left_arm[2]
                elif joint == 18:  # 左肘
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_left_arm[3]
                elif joint == 19:  # 左腕 Roll
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_left_arm[4]
                elif joint == 20:   # 左腕 Pitch
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_left_arm[5]
                elif joint == 21:   # 左腕 Yaw
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_left_arm[6]
                #right_arm
                elif joint == 22:  
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_right_arm[0]
                elif joint == 23:
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_right_arm[1]
                elif joint == 24:  
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_right_arm[2]
                elif joint == 25:  
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_right_arm[3]
                elif joint == 26:  
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_right_arm[4]
                elif joint == 27:  
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_right_arm[5]
                elif joint == 28:  
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q+ratio * init_right_arm[6]
                else:
                    self.low_cmd.motor_cmd[joint].q  = (1.0 - ratio) * self.low_state.motor_state[joint].q

                self.low_cmd.motor_cmd[joint].dq = 0.
                self.low_cmd.motor_cmd[joint].kp = self.kp
                self.low_cmd.motor_cmd[joint].kd = self.kd
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.arm_sdk_publisher.Write(self.low_cmd)

            t += self.control_dt_
            time.sleep(self.control_dt_)
    
    def release_arm(self, duration=1.0):
        """
        逐渐关闭 arm_sdk。
        调用后会阻塞 duration 秒，直到完全 release。
        """
        # 1. 确保至少收到一帧 low_state
        while not self.first_update_low_state:
            time.sleep(0.01)

        steps = int(duration / self.control_dt_)
        t = 0.0

        for _ in range(steps):
            ratio = np.clip(t / duration, 0.0, 1.0)   # 0→1
            # 仅需写 kNotUsedJoint
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0 - ratio
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.arm_sdk_publisher.Write(self.low_cmd)

            t += self.control_dt_
            time.sleep(self.control_dt_)
    
    def set_arm_pose(self, joint_angles, enable_sdk=True):
        """
        立即下发一帧关节角度指令。

        参数
        ----
        joint_angles : Sequence[float]
            长度需与 self.arm_joints 一致，对应各关节目标角度（弧度）。
        enable_sdk : bool, default=True
            是否同时启用 arm_sdk（写 kNotUsedJoint.q = 1.0）。如果想保留系统控制，可设 False。
        """
        if not self.first_update_low_state:
            print("[WARN] 尚未收到 LowState，指令已忽略")
            return

        # 1. 根据参数决定是否启用 arm_sdk
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0 if enable_sdk else 0.0

        # 2. 写入每个关节的目标角度
        for i, joint in enumerate(self.arm_joints):
            if i >= len(joint_angles):
                break
            cmd = self.low_cmd.motor_cmd[joint]
            cmd.q  = float(joint_angles[i])   # 目标角度（弧度）
            cmd.dq = 0.0                     # 速度目标可置 0
            cmd.tau = 0.0                    # 前馈力矩 0
            cmd.kp  = self.kp                # 刚度、阻尼保持默认
            cmd.kd  = self.kd

        # 3. 重新计算 CRC 并发送
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    # custom.Start()
    # custom.init_armpos()      # 例如先复位到零位
    
    # ---------- 等第一帧 low_state ----------
    while not custom.first_update_low_state:
        time.sleep(0.02)
    
    # ---------------- 复位到 0 位 ----------------
    zero_pose = [0.0] * len(custom.arm_joints)           # 与 self.arm_joints 等长
    hold_time = 3.0                                      # 持续发送 2 秒
    steps = int(hold_time / custom.control_dt_)

    for i in range(steps):
        ratio = np.clip((i+1)/ steps, 0.0, 1.0)
        target_pose = [
            (1.0 - ratio) * custom.low_state.motor_state[j].q
            for j in custom.arm_joints
        ]
        custom.set_arm_pose(target_pose, enable_sdk=True)  # 立即写 0 位
        time.sleep(custom.control_dt_)    
 

    custom.release_arm()      # 平滑释放 arm_sdk
    sys.exit(0)    
    # while True:        
    #     time.sleep(1)

    #     if custom.done: 
    #        print("Done!")
    #        sys.exit(-1)    